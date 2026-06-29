#include "idefix.hpp"
#include "setup.hpp"
#include "analysis.hpp"

real epsilonGlob;
real alphaGlob;
real densityFloorGlob;
enum GravityPotential {Kepler, Einstein, PW};
GravityPotential gravityGlob;
enum Referential {Disc, BH};
Referential referentialGlob;
real tiltGlob;
real spinGlob;

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

void GravitomagneticTerm(Hydro *hydro, const real t, const real dtin) {
    auto *data = hydro->data;
    IdefixArray4D<real> Vc = hydro->Vc;
    IdefixArray4D<real> Uc = hydro->Uc;
    IdefixArray1D<real> x1 = data->x[IDIR];
    IdefixArray1D<real> x2 = data->x[JDIR];
    IdefixArray1D<real> x3 = data->x[KDIR];
    real dt = dtin;

    real spin = spinGlob;
    real tilt = tiltGlob * M_PI / 180;
    Referential referential = referentialGlob;
    real Sx;
    real Sy;
    real Sz;
    switch (referential) {
        case Referential::Disc:
            // -tilt so that the initial precession is zero
            Sx = spin * sin(-tilt);
            Sy = ZERO_F;
            Sz = spin * cos(-tilt);
        break;
        case Referential::BH:
            Sx = ZERO_F;
            Sy = ZERO_F;
            Sz = spin;
        break;
    }
    real densityFloor = densityFloorGlob;

    idefix_for("GravitomagneticTerm",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        0, data->np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            real r = x1(i);
            real th = x2(j);
            real phi = x3(k);
            real Vr = Vc(VX1,k,j,i);
            real Vth = Vc(VX2,k,j,i);
            real Vphi = Vc(VX3,k,j,i);

            real Sr = sin(th)*cos(phi)*Sx + sin(th)*sin(phi)*Sy + cos(th)*Sz;
            real Sth = cos(th)*cos(phi)*Sx + cos(th)*sin(phi)*Sy - sin(th)*Sz;
            real Sphi = - sin(phi)*Sx + cos(phi)*Sy;
            real hr = -4*Sr / pow(r,3);
            real hth = 2*Sth / pow(r,3);
            real hphi = 2*Sphi / pow(r,3);
            real Vcrossh_r = Vth*hphi - Vphi*hth;
            real Vcrossh_th = Vphi*hr - Vr*hphi;
            real Vcrossh_phi = Vr*hth - Vth*hr;

            Uc(MX1,k,j,i) += dt * Vc(RHO,k,j,i) * Vcrossh_r;
            Uc(MX2,k,j,i) += dt * Vc(RHO,k,j,i) * Vcrossh_th;
            Uc(MX3,k,j,i) += dt * Vc(RHO,k,j,i) * Vcrossh_phi;

            real q = Vc(RHO,k,j,i) / densityFloor;
            real fact = 1/(1+exp(q*q)); // Smoothly goes to 0 for q>>1 and to 1 for q<<1
            Uc(MX1,k,j,i) -= dt * 10*fact * Vc(RHO,k,j,i) * Vc(VX1,k,j,i);
            Uc(MX2,k,j,i) -= dt * 10*fact * Vc(RHO,k,j,i) * Vc(VX2,k,j,i);
    });
}

void EinsteinPotential(DataBlock &data, const real t, IdefixArray1D<real> &x1, IdefixArray1D<real> &x2, IdefixArray1D<real> &x3, IdefixArray3D<real> &phi) {
    idefix_for("EinsteinPotential",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            real r = x1(i);
            phi(k,j,i) = - 1/r - 3/pow(r,2);
    });
}

void PaczynskiWiitaPotential(DataBlock &data, const real t, IdefixArray1D<real> &x1, IdefixArray1D<real> &x2, IdefixArray1D<real> &x3, IdefixArray3D<real> &phi) {
    idefix_for("PaczynskiWiitaPotential",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            real r = x1(i);
            phi(k,j,i) = - 1/(r - 2);
    });
}

void KeplerPotential(DataBlock &data, const real t, IdefixArray1D<real> &x1, IdefixArray1D<real> &x2, IdefixArray1D<real> &x3, IdefixArray3D<real> &phi) {
    idefix_for("KeplerPotential",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            real r = x1(i);
            phi(k,j,i) = - 1/r;
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
    densityFloorGlob = input.Get<real>("Setup", "densityFloor", 0);
    if (input.Get<std::string>("Setup", "gravity", 0) == "Kepler") gravityGlob = GravityPotential::Kepler;
    else if (input.Get<std::string>("Setup", "gravity", 0) == "Einstein") gravityGlob = GravityPotential::Einstein;
    else if (input.Get<std::string>("Setup", "gravity", 0) == "PW") gravityGlob = GravityPotential::PW;
    else IDEFIX_ERROR("Input correct gravity potential.");
    if (input.Get<std::string>("Setup", "referential", 0) == "Disc") referentialGlob = Referential::Disc;
    else if (input.Get<std::string>("Setup", "referential", 0) == "BH") referentialGlob = Referential::BH;
    else IDEFIX_ERROR("Input correct referential.");
    tiltGlob = input.Get<real>("Setup", "tilt", 0);
    spinGlob = input.Get<real>("Setup", "spin", 0);

    data.hydro->EnrollInternalBoundary(&InternalBoundary);
    data.hydro->EnrollIsoSoundSpeed(&MySoundSpeed);
    data.hydro->viscosity->EnrollViscousDiffusivity(&MyViscosity);
    switch (gravityGlob) {
        case GravityPotential::Kepler:
            data.gravity->EnrollPotential(&KeplerPotential);
        break;
        case GravityPotential::Einstein:
            data.gravity->EnrollPotential(&EinsteinPotential);
        break;
        case GravityPotential::PW:
            data.gravity->EnrollPotential(&PaczynskiWiitaPotential);
        break;
    }
    data.hydro->EnrollUserSourceTerm(&GravitomagneticTerm);

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
    real tilt = tiltGlob * M_PI / 180;
    GravityPotential gravity = gravityGlob;
    Referential referential = referentialGlob;

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

    switch (referential) {
        case Referential::Disc:
            for(int k = 0; k < d.np_tot[KDIR]; k++) {
                for(int j = 0; j < d.np_tot[JDIR]; j++) {
                    for(int i = 0; i < d.np_tot[IDIR]; i++) {           
                        r = d.x[IDIR](i);
                        th = d.x[JDIR](j);

                        real R = r*sin(th);
                        real z = r*cos(th);
                        real Vk = 1.0/sqrt(R);
                        real cs = epsilon/sqrt(R);

                        d.Vc(RHO,k,j,i) = 1.0/(R * sqrt(R)) * exp(1.0/pow(cs,2) * (1/r - 1/R));
                        d.Vc(VX1,k,j,i) = ZERO_F;
                        d.Vc(VX2,k,j,i) = ZERO_F;

                        real grad_Phi;
                        switch (gravity) {
                            case GravityPotential::Kepler:
                                grad_Phi = 1/pow(r,2);
                            break;
                            case GravityPotential::Einstein:
                                grad_Phi = 1/pow(r,2) + 6/pow(r,3);
                            break;
                            case GravityPotential::PW:
                                grad_Phi = 1 / pow(r-2, 2);
                            break;
                        }
                        d.Vc(VX3,k,j,i) = sqrt(r * pow(sin(th), 2) * grad_Phi);
                    }
                }
            }
        break;
        case Referential::BH:
            // real tiltMax = tilt * M_PI / 180.0;    // Conversion in radians
            // real rWarp = 80.0;
            // real rWidth = 16.0;
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
                        // Rotation around the y-axis
                        // tilt = 0.5 * tiltMax * (tanh((r - rWarp)/rWidth) + 1);
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
        break;
    }     

    d.SyncToDevice();
}
