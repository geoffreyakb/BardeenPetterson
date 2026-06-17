#include "idefix.hpp"
#include "setup.hpp"
#include "analysis.hpp"

real epsilonGlob;
real alphaGlob;
real tiltGlob;
real densityFloorGlob;

Analysis *analysis;
void AnalysisFunction(DataBlock &data) {
    analysis->PerformAnalysis(data);
}

void MySoundSpeed(DataBlock &data, const real t, IdefixArray3D<real> &cs) {
    IdefixArray1D<real> r = data.x[IDIR];
    IdefixArray1D<real> th = data.x[JDIR];
    IdefixArray1D<real> phi = data.x[KDIR];
    real epsilon = epsilonGlob;

    idefix_for("MySoundSpeed",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    cs(k,j,i) = epsilon / sqrt(r(i));
                });
}

void MyViscosity(DataBlock &data, const real t, IdefixArray3D<real> &eta1, IdefixArray3D<real> &eta2) {
    IdefixArray4D<real> Vc = data.hydro->Vc;
    IdefixArray1D<real> r = data.x[IDIR];
    IdefixArray1D<real> th = data.x[JDIR];
    IdefixArray1D<real> phi = data.x[KDIR];
    real epsilon = epsilonGlob;
    real alpha = alphaGlob;

    idefix_for("MyViscosity",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    real cs = epsilon / sqrt(r(i));
                    eta1(k,j,i) = alpha * cs * epsilon * r(i) * Vc(RHO,k,j,i);
                    eta2(k,j,i) = ZERO_F;
              });
}

void InternalBoundary(Hydro *hydro, const real t) {
    auto *data = hydro->data;
    IdefixArray4D<real> Vc = hydro->Vc;
    real densityFloor = densityFloorGlob;

    idefix_for("InternalBoundary",
                0, data->np_tot[KDIR],
                0, data->np_tot[JDIR],
                0, data->np_tot[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    if(Vc(RHO,k,j,i) < densityFloor) {
                        Vc(RHO,k,j,i) = densityFloor;
                    }
                });
}

void ControlTerm(Hydro *hydro, const real t, const real dtin) {
    auto *data = hydro->data;
    IdefixArray4D<real> Vc = hydro->Vc;
    IdefixArray4D<real> Uc = hydro->Uc;
    IdefixArray1D<real> x1 = data->x[IDIR];
    IdefixArray1D<real> x2 = data->x[JDIR];
    IdefixArray1D<real> x3 = data->x[KDIR];
    real dt = dtin;

    real densityFloor = densityFloorGlob;

    idefix_for("ControlTerm",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        0, data->np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            real q = Vc(RHO,k,j,i) / densityFloor;
            real fact = 1/(1+exp(q*q)); // Smoothly goes to 0 for q>>1 and to 1 for q<<1
            Uc(MX1,k,j,i) -= dt * 10*fact * Vc(RHO,k,j,i) * Vc(VX1,k,j,i);
            Uc(MX2,k,j,i) -= dt * 10*fact * Vc(RHO,k,j,i) * Vc(VX2,k,j,i);
    });
}

void ComputeUserVars(DataBlock & data, UserDefVariablesContainer &variables) {
    DataBlockHost d(data);
    d.SyncFromDevice();

    IdefixArray3D<real> scrh("Scratch", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
    IdefixArray3D<real>::HostMirror scrhHost = Kokkos::create_mirror_view(scrh);
    Kokkos::deep_copy(scrhHost,scrh);

    IdefixHostArray3D<real> InvDt  = variables["InvDt"];

    for(int k = d.beg[KDIR]; k < d.end[KDIR] ; k++) {
        for(int j = d.beg[JDIR]; j < d.end[JDIR] ; j++) {
            for(int i = d.beg[IDIR]; i < d.end[IDIR] ; i++) {
                InvDt(k,j,i) = d.InvDt(k,j,i);
            }
        }
    }
}

void CoarsenFunction(DataBlock &data) {
    IdefixArray2D<int> coarseningLevel = data.coarseningLevel[KDIR];
    IdefixArray1D<real> th = data.x[JDIR];
    idefix_for("set_coarsening", 0, data.np_tot[JDIR], 0, data.np_tot[IDIR],
                KOKKOS_LAMBDA(int j, int i) {
                    int c = 1.0 / std::abs(sin(th(j)));
                    if(c > 6) c = 6;
                    coarseningLevel(j,i) = c;
                });
}

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    epsilonGlob = input.Get<real>("Setup", "epsilon", 0);
    alphaGlob = input.Get<real>("Setup", "alpha", 0);
    tiltGlob = input.Get<real>("Setup", "tilt", 0);
    densityFloorGlob = input.Get<real>("Setup", "densityFloor", 0);

    data.hydro->EnrollInternalBoundary(&InternalBoundary);
    data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
    data.hydro->viscosity->EnrollViscousDiffusivity(&MyViscosity);

    data.hydro->EnrollUserSourceTerm(&ControlTerm);
    analysis = new Analysis(input, grid, data);
    output.EnrollAnalysis(&AnalysisFunction);
    output.EnrollUserDefVariables(&ComputeUserVars);
    if(data.haveGridCoarsening) {
        data.EnrollGridCoarseningLevels(&CoarsenFunction);
    }
}

void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    real epsilon = epsilonGlob;
    real tilt = tiltGlob * M_PI / 180.0;    // Conversion in radians
    real tiltMax = tiltGlob * M_PI / 180.0;    // Conversion in radians
    real rWarp = 20.0;
    real rWidth = 5.0;

    real r, th, phi;
    real x, y, z;
    real xUnt, yUnt, zUnt;
    real rUnt, thUnt, phiUnt;

    real rhoUnt, VrUnt, VthUnt, VphiUnt;
    real er_ex, er_ey, er_ez;
    real eth_ex, eth_ey, eth_ez;
    real ephi_ex, ephi_ey, ephi_ez;
    real VxUnt, VyUnt, VzUnt;

    real Vx, Vy, Vz;
    real ex_er, ex_eth, ex_ephi;
    real ey_er, ey_eth, ey_ephi;
    real ez_er, ez_eth, ez_ephi;
    real Vr, Vth, Vphi;
    

    for(int k = 0; k < d.np_tot[KDIR]; k++) {
        for(int j = 0; j < d.np_tot[JDIR]; j++) {
            for(int i = 0; i < d.np_tot[IDIR]; i++) {                
                // Spherical coordinates
                r = d.x[IDIR](i);
                th = d.x[JDIR](j);
                phi = d.x[KDIR](k);
                // Cartesian coordinates
                x = r * sin(th) * cos(phi);
                y = r * sin(th) * sin(phi);
                z = r * cos(th);
                // Rotation around the y-axis (the -tilt is for a initial precession angle of 0)
                // real tilt = 0.5 * tiltMax * (tanh((r - rWarp)/rWidth) + 1);
                xUnt = cos(-tilt)*x + sin(-tilt)*z;
                yUnt = y;
                zUnt = -sin(-tilt)*x + cos(-tilt)*z;
                // Back to spherical coordinates
                rUnt = sqrt(xUnt*xUnt + yUnt*yUnt + zUnt*zUnt);
                thUnt = acos(zUnt/rUnt);
                phiUnt = atan2(yUnt,xUnt);

                // Useful parameters
                real R = rUnt * sin(thUnt);
                real Vk = 1.0 / sqrt(R);
                real cs2 = pow(epsilon / sqrt(R), 2);
                // Physical value in the untilted version of the disk
                rhoUnt = 1.0/(R * sqrt(R)) * exp(1.0/cs2 * (1/rUnt - 1/R));
                VrUnt = ZERO_F;
                VthUnt = ZERO_F;
                real grad_Phi = 1/pow(r,2);
                VphiUnt = sqrt(r * pow(sin(th), 2) * grad_Phi);

                // Expressing spherical unit vectors as cartesian ones (dot products)
                er_ex = sin(thUnt)*cos(phiUnt);
                er_ey = sin(thUnt)*sin(phiUnt);
                er_ez = cos(thUnt);
                eth_ex = cos(thUnt)*cos(phiUnt);
                eth_ey = cos(thUnt)*sin(phiUnt);
                eth_ez = -sin(thUnt);
                ephi_ex = -sin(phiUnt);
                ephi_ey = cos(phiUnt);
                ephi_ez = ZERO_F;
                // Cartesian untilted velocity
                VxUnt = VrUnt*er_ex + VthUnt*eth_ex + VphiUnt*ephi_ex;
                VyUnt = VrUnt*er_ey + VthUnt*eth_ey + VphiUnt*ephi_ey;
                VzUnt = VrUnt*er_ez + VthUnt*eth_ez + VphiUnt*ephi_ez;
                // Cartesian tilted velocity
                Vx = cos(tilt)*VxUnt + sin(tilt)*VzUnt;
                Vy = VyUnt;
                Vz = -sin(tilt)*VxUnt + cos(tilt)*VzUnt;      

                // Expressing cartesian unit vectors as spherical ones (dot products)
                ex_er = sin(th)*cos(phi);
                ex_eth = cos(th)*cos(phi);
                ex_ephi = -sin(phi);
                ey_er = sin(th)*sin(phi);
                ey_eth = cos(th)*sin(phi);
                ey_ephi = cos(phi);
                ez_er = cos(th);
                ez_eth = -sin(th);
                ez_ephi = ZERO_F;
                // Final spherical velocity
                Vr = Vx*ex_er + Vy*ey_er + Vz*ez_er;
                Vth = Vx*ex_eth + Vy*ey_eth + Vz*ez_eth;
                Vphi = Vx*ex_ephi + Vy*ey_ephi + Vz*ez_ephi;

                d.Vc(RHO,k,j,i) = rhoUnt;
                d.Vc(VX1,k,j,i) = Vr;
                d.Vc(VX2,k,j,i) = Vth;
                d.Vc(VX3,k,j,i) = Vphi;
            }
        }
    }

    d.SyncToDevice();
}
