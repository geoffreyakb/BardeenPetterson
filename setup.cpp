#include "idefix.hpp"
#include "setup.hpp"
#include "analysis.hpp"

real epsilonGlob;
real alphaGlob;
real densityFloorGlob;
real rminGlob;
enum GravityPotential {Kepler, Einstein, PW};
GravityPotential gravityGlob;
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
    // -tilt so that the initial precession is zero
    real Sx = spin * sin(-tilt);
    real Sy = ZERO_F;
    real Sz = spin * cos(-tilt);

    real densityFloor = densityFloorGlob;

    idefix_for("GravitomagneticTerm",
        0, data->np_tot[KDIR],
        0, data->np_tot[JDIR],
        0, data->np_tot[IDIR],
        KOKKOS_LAMBDA (int k, int j, int i) {
            if (Vc(RHO,k,j,i) > densityFloor) {
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
            }
            else {
                real r = x1(i);
                // real omega = sqrt(pow(Vc(VX1,k,j,i), 2) + pow(Vc(VX2,k,j,i), 2) + pow(Vc(VX3,k,j,i), 2));
                real omega = 1/(2*dt);
                Uc(MX1,k,j,i) -= dt*omega * Vc(RHO,k,j,i) * Vc(VX1,k,j,i);
                Uc(MX2,k,j,i) -= dt*omega * Vc(RHO,k,j,i) * Vc(VX2,k,j,i);
                // if (dt*omega > 1.0) {
                //     idfx::cout << "Violating CFL condition probably" << std::endl; 
                // }
            }
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

void BodyForce(DataBlock &data, const real t, IdefixArray4D<real> &force) {
    idfx::pushRegion("BodyForce");
    IdefixArray4D<real> Vc = data.hydro->Vc;
    IdefixArray1D<real> r = data.x[IDIR];
    GravityPotential gravity = gravityGlob;
    real densityFloor = densityFloorGlob;

    idefix_for("BodyForce",
                data.beg[KDIR] , data.end[KDIR],
                data.beg[JDIR] , data.end[JDIR],
                data.beg[IDIR] , data.end[IDIR],
                KOKKOS_LAMBDA (int k, int j, int i) {
                    if (Vc(RHO,k,j,i) > densityFloor) {
                        switch (gravity) {
                            case GravityPotential::Kepler:
                                force(IDIR,k,j,i) = - 1/pow(r(i),2);
                            break;
                            case GravityPotential::Einstein:
                                force(IDIR,k,j,i) = - 1/pow(r(i),2) - 6/pow(r(i),3);
                            break;
                            case GravityPotential::PW:
                                force(IDIR,k,j,i) = - 1/pow(r(i)-2,2);
                            break;
                        }
                    }
                    else {
                        force(IDIR,k,j,i) = ZERO_F;
                    }
                    force(JDIR,k,j,i) = ZERO_F;
                    force(KDIR,k,j,i) = ZERO_F;
        });

    idfx::popRegion();
}

Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    epsilonGlob = input.Get<real>("Setup", "epsilon", 0);
    alphaGlob = input.Get<real>("Setup", "alpha", 0);
    densityFloorGlob = input.Get<real>("Setup", "densityFloor", 0);
    rminGlob = input.Get<real>("Grid", "X1-grid", 1);
    if (input.Get<std::string>("Setup", "gravity", 0) == "Kepler") gravityGlob = GravityPotential::Kepler;
    else if (input.Get<std::string>("Setup", "gravity", 0) == "Einstein") gravityGlob = GravityPotential::Einstein;
    else if (input.Get<std::string>("Setup", "gravity", 0) == "PW") gravityGlob = GravityPotential::PW;
    else IDEFIX_ERROR("Input correct gravity potential.");
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
    // data.gravity->EnrollBodyForce(BodyForce);
    data.hydro->EnrollUserSourceTerm(&GravitomagneticTerm);

    analysis = new Analysis(input, grid, data);
    output.EnrollAnalysis(&AnalysisFunction);
}

void Setup::InitFlow(DataBlock &data) {
    DataBlockHost d(data);
    real epsilon = epsilonGlob;
    GravityPotential gravity = gravityGlob;
    real r, th;

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

    d.SyncToDevice();
}
