from src import *
# ----------------------------------------------------------------------------------
# Plotting parameters
# ----------------------------------------------------------------------------------
# plt.rcParams.update({
#     "text.usetex": True,
#     'text.latex.preamble':r'\usepackage{amsmath}',
#     "font.family": "Fourier"
# })
w, l, l_log = 1.25, 10, 6
pad, lpad, shrink = 0.1, 5, 0.8
formats = tkr.FormatStrFormatter('%.1e')

# ----------------------------------------------------------------------------------
# Getting the input data
# ----------------------------------------------------------------------------------
conf = inifix.load("idefix.ini")
t_max = conf["TimeIntegrator"]["tstop"]
n_average = int(t_max / conf["Output"]["analysis"]) + 1
n_vtk = int(t_max / conf["Output"]["vtk"]) + 1
r_min = conf["Grid"]["X1-grid"][1]
r_max = conf["Grid"]["X1-grid"][-1]
n_r = conf["Grid"]["X1-grid"][2]
epsilon = conf["Setup"]["epsilon"]
alpha = conf["Setup"]["alpha"]
densityFloor = conf["Setup"]["densityFloor"]
gravity = conf["Setup"]["gravity"]
beta_0 = conf["Setup"]["tilt"]
beta_0 *= np.pi / 180
spin = conf["Setup"]["spin"]

# Reading the analysis files
t, M_tot = READ_BOX_AVERAGE()
r, Sigma, rho_mean, L, beta, gamma, LBH, betaBH, gammaBH, rho_Vr = READ_RADIAL_AVERAGE(n_average, n_r, beta_0)

# ----------------------------------------------------------------------------------
# Normalizations and useful quantities
# ----------------------------------------------------------------------------------
# Time
t_orbit = 2*np.pi*(r_min**1.5) / np.sqrt(1 - 2.5*epsilon**2)       # This is an input
omega_orbit = 1/t_orbit
# n_orbit = (t[-1] // t_orbit)
n_orbit = 500
t *= omega_orbit
wh_t_final = np.where(t >= n_orbit)[0][0]
T, R = np.meshgrid(t, r, indexing="ij")
# Radial averages
LBH_unit = np.zeros_like(LBH)
LBH_unit[:,:,0] = LBH[:,:,0] / np.sqrt(LBH[:,:,0]**2 + LBH[:,:,1]**2 + LBH[:,:,2]**2)
LBH_unit[:,:,1] = LBH[:,:,1] / np.sqrt(LBH[:,:,0]**2 + LBH[:,:,1]**2 + LBH[:,:,2]**2)
LBH_unit[:,:,2] = LBH[:,:,2] / np.sqrt(LBH[:,:,0]**2 + LBH[:,:,1]**2 + LBH[:,:,2]**2)
psi = R * np.sqrt(np.gradient(LBH_unit[:,:,0], r, axis=1)**2 + np.gradient(LBH_unit[:,:,1], r, axis=1)**2 + np.gradient(LBH_unit[:,:,2], r, axis=1)**2)      # Warp profile
zeta = np.gradient(betaBH, t, axis=0)      # Nodal precession frequency
eta = np.gradient(gammaBH, t, axis=0)      # Apsidal precession frequency
# Must implement for every gravitationnal potential ! --------------------------------------------------------------------------------------------------------------
if spin != 0:
    if gravity == "Kepler":
        omega2_E = 1/R**3 - 2*spin/R**(9/2)
        zeta_th = 1/omega2_E * (-4*spin) / (2*R**(3/2) - 4*spin)
        eta_th = 1/omega2_E * 3*spin / (2*R**(3/2) - 4*spin)
    elif gravity == "Einstein":
        omega2_E = 1/R**3 * (1 + 6/R) - 2*spin/R**(9/2)
        zeta_th = -2*spin / (omega2_E * R**(9/2))
        eta_th = -1/(2*omega2_E) * (6/R**4 - 3*spin/R**(9/2))
    t_visc = 1 / (alpha * np.sqrt(omega2_E)) * epsilon**(-2)       # To verify (especially that omega)
    M_dot_visc = Sigma[0,0]*(2*np.pi*R[0,0]**2*epsilon) / t_visc
    M_dot = 2*np.pi*R * Sigma * (-rho_Vr / rho_mean)
    M_dot /= M_dot_visc
    zeta /= zeta_th
    eta /= eta_th
else:
    omega_K = KEPLER(r)
    t_visc = 1 / (alpha * np.sqrt(omega_K)) * epsilon**(-2)       # To verify (especially that omega)
    M_dot_visc = Sigma[0,0]*(2*np.pi*R[0,0]**2*epsilon) / t_visc
    M_dot = 2*np.pi*R * Sigma * (-rho_Vr / rho_mean)
    M_dot /= M_dot_visc

# ----------------------------------------------------------------------------------
# angles color plot
# ----------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

ax = axs[0]
ax.set_title(r"$\beta$ [°]")
buff = np.max(np.abs(betaBH))
ticks = np.linspace(-buff, buff, 5)
pc0 = ax.pcolormesh(T, R, betaBH, cmap="berlin", vmin=-buff, vmax=buff)
cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
ax.set_facecolor("black")
ax.set_xlabel(r"$t\omega_0$ [$-$]")
ax.set_ylabel(r"$r$ [$R_g$]")
ax.set_xlim((0, n_orbit))
xplot = np.linspace(0, n_orbit, 5)
xl = [f"{i:.0f}" for i in xplot]
xl[0] = ""
xl[-1] = ""
ax.set_xticks(xplot,xl)
ax.set_ylim((0, r_max))
yplot = np.linspace(r_min, r_max, 5)
yl = [f"{i:.1f}" for i in yplot]
ax.set_yticks(yplot, yl)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
for spine in ax.spines.values():
    spine.set_linewidth(w)

ax = axs[1]
ax.set_title(r"$\psi$ [$-$]")
buff = np.max(np.abs(psi))
ticks = np.linspace(0, buff, 5)
pc0 = ax.pcolormesh(T, R, psi, cmap="inferno", vmin=0, vmax=buff)
cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
ax.set_facecolor("black")
ax.set_xlabel(r"$t\omega_0$ [$-$]")
ax.set_ylabel(r"$r$ [$R_g$]")
ax.set_xlim((0, n_orbit))
xplot = np.linspace(0, n_orbit, 5)
xl = [f"{i:.1f}" for i in xplot]
xl[0] = ""
xl[-1] = ""
ax.set_xticks(xplot,xl)
ax.set_ylim((r_min, r_max))
yplot = np.linspace(r_min, r_max, 5)
yl = ["" for i in yplot]
ax.set_yticks(yplot,yl)
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
for spine in ax.spines.values():
    spine.set_linewidth(w)

ax = axs[2]
ax.set_title(r"$\gamma$ [°]")
buff = np.max(np.abs(gammaBH))
ticks = np.linspace(-buff, buff, 5)
pc0 = ax.pcolormesh(T, R, gammaBH, cmap="berlin", vmin=-buff, vmax=buff)
cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
ax.set_facecolor("black")
ax.set_xlabel(r"$t\omega_0$ [$-$]")
ax.set_ylabel(r"$r$ [$R_g$]")
ax.set_xlim((0, n_orbit))
xplot = np.linspace(0, n_orbit, 5)
xl = [f"{i:.1f}" for i in xplot]
xl[0] = ""
xl[-1] = ""
ax.set_xticks(xplot,xl)
ax.set_ylim((r_min, r_max))
yplot = np.linspace(r_min, r_max, 5)
yl = [f"{i:.1f}" for i in yplot]
ax.set_yticks(yplot, yl)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.yaxis.set_ticks_position('both')
ax.xaxis.set_ticks_position('both')
for spine in ax.spines.values():
    spine.set_linewidth(w)

fig.tight_layout()
plt.savefig(f"./plots/angles.png", bbox_inches='tight', dpi=300)
plt.close()
# ----------------------------------------------------------------------------------
# angles movie
# ----------------------------------------------------------------------------------
angles_plots = []
for n in range(n_average):
    if ((n%int(conf["Output"]["vtk"]) == 0) or (n == wh_t_final)) and (n <= wh_t_final):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        ax = axs[0,0]
        ax.plot(r, betaBH[n,:], color="tab:blue")
        ax.set_xlim((0, r_max))
        ax.set_xlabel(r"$r$ [$R_g$]")
        mini = np.min(betaBH)*(1 - 0.05*np.sign(betaBH.min()))
        maxi = np.max(betaBH)*(1 + 0.05*np.sign(betaBH.max()))
        ax.set_ylim((mini, maxi))
        ax.set_ylabel(r"$\beta$ [°]")
        ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
        ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
        for spine in ax.spines.values():
                spine.set_linewidth(w)
        ax.grid()

        ax = axs[0,1]
        ax.plot(r, psi[n,:], color="tab:red")
        ax.set_xlim((0, r_max))
        ax.set_xlabel(r"$r$ [$R_g$]")
        mini = np.min(psi)*(1 - 0.05*np.sign(psi.min()))
        maxi = np.max(psi)*(1 + 0.05*np.sign(psi.max()))
        ax.set_ylim((mini, maxi))
        ax.set_ylabel(r"$\psi$ [$-$]")
        ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
        ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
        for spine in ax.spines.values():
                spine.set_linewidth(w)
        ax.grid()

        ax = axs[0,2]
        ax.plot(r, gammaBH[n,:], color="tab:green")
        ax.set_xlim((0, r_max))
        ax.set_xlabel(r"$r$ [$R_g$]")
        mini = np.min(gammaBH)*(1 - 0.05*np.sign(gammaBH.min()))
        maxi = np.max(gammaBH)*(1 + 0.05*np.sign(gammaBH.max()))
        ax.set_ylim((mini, maxi))
        ax.set_ylabel(r"$\gamma$ [°]")
        ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
        ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
        for spine in ax.spines.values():
                spine.set_linewidth(w)
        ax.grid()

        ax = axs[1,0]
        ax.plot(r, zeta[n,:], color="tab:blue")
        ax.set_xlim((0, r_max))
        ax.set_xlabel(r"$r$ [$R_g$]")
        mini = np.min(zeta)*(1 - 0.05*np.sign(zeta.min()))
        maxi = np.max(zeta)*(1 + 0.05*np.sign(zeta.max()))
        ax.set_ylim((mini, maxi))
        ax.set_ylabel(r"$\zeta/\zeta_\text{theory}$ [$-$]")
        ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
        ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
        for spine in ax.spines.values():
                spine.set_linewidth(w)
        ax.grid()

        ax = axs[1,1]
        ax.plot(r, Sigma[n,:]/Sigma[0,:], color="tab:red")
        ax.set_xlim((0, r_max))
        ax.set_xlabel(r"$r$ [$R_g$]")
        mini = 1e-4
        maxi = 1e2
        ax.set_ylim((mini, maxi))
        ax.set_yscale("log")
        ax.set_ylabel(r"$\Sigma/\Sigma(t=0)$ [$-$]")
        ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
        ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
        ax.tick_params(axis='y', which='minor', length=l_log, width=w)
        for spine in ax.spines.values():
                spine.set_linewidth(w)
        ax.grid()

        ax = axs[1,2]
        ax.plot(r, eta[n,:], color="tab:green")
        ax.set_xlim((0, r_max))
        ax.set_xlabel(r"$r$ [$R_g$]")
        mini = np.min(eta)*(1 - 0.05*np.sign(eta.min()))
        maxi = np.max(eta)*(1 + 0.05*np.sign(eta.max()))
        ax.set_ylim((mini, maxi))
        ax.set_ylabel(r"$\eta/\eta_\text{theory}$ [$-$]")
        ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
        ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
        for spine in ax.spines.values():
                spine.set_linewidth(w)
        ax.grid()

        fig.suptitle(r"$t\omega_0 =$ " + f"{t[n]:.0f}")
        fig.tight_layout()
        plt.savefig(f"./output/plots/angles_{n}.png", bbox_inches='tight', dpi=300)
        plt.close()
        angles_plots.append(f"./output/plots/angles_{n}.png")
MOVIE(angles_plots, "angles")

# ----------------------------------------------------------------------------------
# mass velocity and 3D movies
# ----------------------------------------------------------------------------------
mass_plots = []
mass_xz_plots = []
mass_zoom_xz_plots = []
mass_yz_plots = []
mass_zoom_yz_plots = []

velocity_xz_plots = []
velocity_zoom_xz_plots = []
velocity_yz_plots = []
velocity_zoom_yz_plots = []

sourceTerm_xz_plots = []
sourceTerm_zoom_xz_plots = []
sourceTerm_yz_plots = []
sourceTerm_zoom_yz_plots = []

gradP_xz_plots = []
gradP_zoom_xz_plots = []
gradP_yz_plots = []
gradP_zoom_yz_plots = []

vAdv_xz_plots = []
vAdv_zoom_xz_plots = []
vAdv_yz_plots = []
vAdv_zoom_yz_plots = []

Svisc_xz_plots = []
Svisc_zoom_xz_plots = []
Svisc_yz_plots = []
Svisc_zoom_yz_plots = []

rho_velocity_xz_plots = []
rho_velocity_zoom_xz_plots = []
rho_velocity_yz_plots = []
rho_velocity_zoom_yz_plots = []

rho_sourceTerm_xz_plots = []
rho_sourceTerm_zoom_xz_plots = []
rho_sourceTerm_yz_plots = []
rho_sourceTerm_zoom_yz_plots = []

rho_gradP_xz_plots = []
rho_gradP_zoom_xz_plots = []
rho_gradP_yz_plots = []
rho_gradP_zoom_yz_plots = []

rho_vAdv_xz_plots = []
rho_vAdv_zoom_xz_plots = []
rho_vAdv_yz_plots = []
rho_vAdv_zoom_yz_plots = []

rho_Svisc_xz_plots = []
rho_Svisc_zoom_xz_plots = []
rho_Svisc_yz_plots = []
rho_Svisc_zoom_yz_plots = []

velocityLoc_xz_plots = []
velocityLoc_zoom_xz_plots = []
velocityLoc_yz_plots = []
velocityLoc_zoom_yz_plots = []

rho_velocityLoc_xz_plots = []
rho_velocityLoc_zoom_xz_plots = []
rho_velocityLoc_yz_plots = []
rho_velocityLoc_zoom_yz_plots = []

rotation_plots = []
threeD_plots = []

for n in range(n_vtk):
    n_analysis = int(n * conf["Output"]["vtk"] // conf["Output"]["analysis"])
    r_vtk, theta_vtk, phi_vtk, rho, v_r, v_theta, v_phi = READ_VTK(n)
    PHI, TH, R = np.meshgrid(phi_vtk, theta_vtk, r_vtk, indexing="ij")

    T = epsilon / R
    c_s = np.sqrt(T)

    # mass inflow ---------------------------------------------------------------------------------------------------------------------------------------
    inflow = - rho * v_r
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    # gravitomagnetic source term -----------------------------------------------------------------------------------------------------------------------
    Sx = np.sin(-beta_0) * spin
    Sy = 0
    Sz = np.cos(-beta_0) * spin
    Sr = np.sin(TH)*np.cos(PHI)*Sx + np.sin(TH)*np.sin(PHI)*Sy + np.cos(TH)*Sz
    Sth = np.cos(TH)*np.cos(PHI)*Sx + np.cos(TH)*np.sin(PHI)*Sy - np.sin(TH)*Sz
    Sphi = - np.sin(PHI)*Sx + np.cos(PHI)*Sy
    hr = -4*Sr / R**3
    hth = 2*Sth / R**3
    hphi = 2*Sphi / R**3

    Vcrossh_r = v_theta*hphi - v_phi*hth
    Vcrossh_th = v_phi*hr - v_r*hphi
    Vcrossh_phi = v_r*hth - v_theta*hr
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    # presssure gradient --------------------------------------------------------------------------------------------------------------------------------
    P = rho * T
    gradP_r = np.gradient(P, r_vtk, axis=2)
    gradP_th = 1/R * np.gradient(P, theta_vtk, axis=1)
    gradP_phi = 1/(R*np.sin(TH)) * np.gradient(P, phi_vtk, axis=0)

    gradPsi_r = np.zeros_like(v_r)
    if gravity == "Kepler":
        gradPsi_r = 1/R**2
    elif gravity == "Einstein":
        gradPsi_r = 1/R**2 + 6/R**3
    elif gravity == "PW":
        gradPsi_r = 1/(R - 2)**2
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    # velocity self-advection ---------------------------------------------------------------------------------------------------------------------------
    dr_v_r = np.gradient(v_r, r_vtk, axis=2)
    dr_v_th = np.gradient(v_theta, r_vtk, axis=2)
    dr_v_phi = np.gradient(v_phi, r_vtk, axis=2)
    dth_v_r = np.gradient(v_r, theta_vtk, axis=1)
    dth_v_th = np.gradient(v_theta, theta_vtk, axis=1)
    dth_v_phi = np.gradient(v_phi, theta_vtk, axis=1)
    dphi_v_r = np.gradient(v_r, phi_vtk, axis=0)
    dphi_v_th = np.gradient(v_theta, phi_vtk, axis=0)
    dphi_v_phi = np.gradient(v_phi, phi_vtk, axis=0)

    v_adv_r = v_r*dr_v_r + v_theta/R*dth_v_r + v_phi/(R*np.sin(TH))*dphi_v_r - (v_theta**2 + v_phi**2)/R
    v_adv_th = v_r*dr_v_th + v_theta/R*dth_v_th + v_phi/(R*np.sin(TH))*dphi_v_th + v_theta*v_r/R - v_phi**2/(R*np.tan(TH))
    v_adv_phi = v_r*dr_v_phi + v_theta/R*dth_v_phi + v_phi/(R*np.sin(TH))*dphi_v_phi + v_phi*v_r/R + v_phi*v_theta/(R*np.tan(TH))
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    # viscous tensor ------------------------------------------------------------------------------------------------------------------------------------
    eta_1, eta_2 = alpha * np.sqrt(T) * epsilon * R * rho, np.zeros_like(v_r)
    div_v = 1/R**2 * np.gradient(R**2*v_r, r_vtk, axis=2) + 1/(R*np.sin(TH)) * np.gradient(v_theta*np.sin(TH), theta_vtk, axis=1) + 1/(R*np.sin(TH)) * np.gradient(v_phi, phi_vtk, axis=0)

    rr = 2*eta_1 * (dr_v_r) + (eta_2 - 2/3*eta_1) * div_v
    dr_rr = np.gradient(rr, r_vtk, axis=2)
    dth_rr = np.gradient(rr, theta_vtk, axis=1)
    dphi_rr = np.gradient(rr, phi_vtk, axis=0)
    rth = eta_1 * (1/R*dth_v_r - v_theta/R + dr_v_th)
    dr_rth = np.gradient(rth, r_vtk, axis=2)
    dth_rth = np.gradient(rth, theta_vtk, axis=1)
    dphi_rth = np.gradient(rth, phi_vtk, axis=0)
    rphi = eta_1 * (1/(R*np.sin(TH))*dphi_v_r - v_phi/R + dr_v_phi)
    dr_rphi = np.gradient(rphi, r_vtk, axis=2)
    dth_rphi = np.gradient(rphi, theta_vtk, axis=1)
    dphi_rphi = np.gradient(rphi, phi_vtk, axis=0)
    
    thth = 2*eta_1 * (1/R*dth_v_th + 1*v_r/R) + (eta_2 - 2/3*eta_1) * div_v
    dr_thth = np.gradient(thth, r_vtk, axis=2)
    dth_thth = np.gradient(thth, theta_vtk, axis=1)
    dphi_thth = np.gradient(thth, phi_vtk, axis=0)
    thphi = eta_1 * (1/(R*np.sin(TH))*dphi_v_th - v_phi/(R*np.tan(TH)) + 1/R*dth_v_phi)
    dr_thphi = np.gradient(thphi, r_vtk, axis=2)
    dth_thphi = np.gradient(thphi, theta_vtk, axis=1)
    dphi_thphi = np.gradient(thphi, phi_vtk, axis=0)
    
    phiphi = 2*eta_1 * (1/(R*np.sin(TH))*dphi_v_phi + 1/(np.tan(TH))*v_theta/R + v_r/R) + (eta_2 - 2/3*eta_1) * div_v
    dr_phiphi = np.gradient(phiphi, r_vtk, axis=2)
    dth_phiphi = np.gradient(phiphi, theta_vtk, axis=1)
    dphi_phiphi = np.gradient(phiphi, phi_vtk, axis=0)

    Svisc_r = - (dr_rr + 2*rr/R + 1/R*dth_rth + 1/(R*np.tan(TH))*rth + 1/(R*np.sin(TH))*dphi_rphi - 1/R*(thth + phiphi))
    Svisc_th = - (dr_rth + 2*rth/R + 1/R*dth_thth + 1/(R*np.tan(TH))*thth + 1/(R*np.sin(TH))*dphi_thphi + rth/R - 1/(R*np.tan(TH))*phiphi)
    Svisc_phi = - (dr_rphi + 2*rphi/R + 1/R*dth_thphi + 1/(R*np.sin(TH))*dphi_phiphi + rphi/R + 1/(R*np.tan(TH))*(thphi + thphi))
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    # rotation curve ------------------------------------------------------------------------------------------------------------------------------------
    # See Ogilvie (1999) for these transformations in warped coordinates
    X = np.sin(TH)*np.cos(PHI) * R + np.cos(TH)*np.cos(PHI) * TH - np.sin(PHI) * PHI
    Y = np.sin(TH)*np.sin(PHI) * R + np.cos(TH)*np.sin(PHI) * TH + np.cos(PHI) * PHI
    Z = np.cos(TH) * R - np.sin(TH) * TH
    v_x = np.sin(TH)*np.cos(PHI) * v_r + np.cos(TH)*np.cos(PHI) * v_theta - np.sin(PHI) * v_phi
    v_y = np.sin(TH)*np.sin(PHI) * v_r + np.cos(TH)*np.sin(PHI) * v_theta + np.cos(PHI) * v_phi
    v_z = np.cos(TH) * v_r - np.sin(TH) * v_theta
    beta_Loc = np.zeros_like(v_r)
    gamma_Loc = np.zeros_like(v_r)
    for k in range(phi_vtk.size):
        for j in range(theta_vtk.size):
            beta_Loc[k,j,:] = beta[n_analysis,:]
            gamma_Loc[k,j,:] = gamma[n_analysis,:]
    TH_Loc = (TH - beta_Loc) % np.pi
    PHI_Loc = (PHI - gamma_Loc) % (2*np.pi)
    a11 = np.sin(TH_Loc)*(np.cos(PHI_Loc)*np.cos(beta_Loc)*np.cos(gamma_Loc) - np.sin(PHI_Loc)*np.sin(gamma_Loc)) + np.cos(TH_Loc)*np.sin(beta_Loc)*np.cos(gamma_Loc)
    a12 = np.sin(TH_Loc)*(np.cos(PHI_Loc)*np.cos(beta_Loc)*np.sin(gamma_Loc) + np.sin(PHI_Loc)*np.cos(gamma_Loc)) + np.cos(TH_Loc)*np.sin(beta_Loc)*np.sin(gamma_Loc)
    a13 = -np.sin(TH_Loc)*np.cos(PHI_Loc)*np.sin(beta_Loc) + np.cos(TH_Loc)*np.cos(beta_Loc)
    a21 = np.cos(TH_Loc)*(np.cos(PHI_Loc)*np.cos(beta_Loc)*np.cos(gamma_Loc) - np.sin(PHI_Loc)*np.sin(gamma_Loc)) - np.sin(TH_Loc)*np.sin(beta_Loc)*np.cos(gamma_Loc)
    a22 = np.cos(TH_Loc)*(np.cos(PHI_Loc)*np.cos(beta_Loc)*np.sin(gamma_Loc) + np.sin(PHI_Loc)*np.cos(gamma_Loc)) - np.sin(TH_Loc)*np.sin(beta_Loc)*np.sin(gamma_Loc)
    a23 = -np.cos(TH_Loc)*np.cos(PHI_Loc)*np.sin(beta_Loc) - np.sin(TH_Loc)*np.cos(beta_Loc)
    a31 = -np.sin(PHI_Loc)*np.cos(beta_Loc)*np.cos(gamma_Loc) - np.cos(PHI_Loc)*np.sin(gamma_Loc)
    a32 = -np.sin(PHI_Loc)*np.cos(beta_Loc)*np.sin(gamma_Loc) + np.cos(PHI_Loc)*np.cos(gamma_Loc)
    a33 = np.sin(PHI_Loc)*np.sin(beta_Loc)
    v_r_Loc = a11*v_x + a12*v_y + a13*v_z
    v_th_Loc = a21*v_x + a22*v_y + a23*v_z
    v_phi_Loc = a31*v_x + a32*v_y + a33*v_z

    omega = np.zeros(n_r)
    # Regular grid assumed...
    dth = np.pi / conf["Grid"]["X2-grid"][2]
    dphi = 2*np.pi / conf["Grid"]["X3-grid"][2]
    for i in range(r_vtk.size):
        omega[i] = np.sum(rho[:,:,i] * v_phi_Loc[:,:,i] * np.sin(TH[:,:,i]) * dth * dphi) / (4*np.pi*r_vtk[i]*rho_mean[n_analysis,i])
    kappa2 = 4*omega**2 + 2*r*omega * np.gradient(omega, r_vtk)
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    # mass plots ----------------------------------------------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ax = axs[0]
    Sigma /= Sigma[0,:]
    ax.plot(r, Sigma[n_analysis,:], color="tab:red")
    ax.set_xlim((0, r_max))
    xplot = np.linspace(0, r_max , 7)
    xl = [f"{i:.0f}" for i in xplot]
    xl[0] = ""
    xl[-1] = ""
    ax.set_xticks(xplot, xl)
    ax.set_xlabel(r"$r$ [$R_g$]")
    ax.set_ylim((1e-4, 1e2))
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Sigma/\Sigma(t=0)$ [$-$]")
    ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
    ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
    ax.tick_params(axis='y', which='minor', length=l_log, width=w)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
            spine.set_linewidth(w)
    ax.grid()

    ax = axs[1]
    ax.plot(r, M_dot[n_analysis,:], color="tab:orange")
    ax.set_xlim((0, r_max))
    xplot = np.linspace(0, r_max , 7)
    xl = [f"{i:.0f}" for i in xplot]
    xl[0] = ""
    xl[-1] = ""
    ax.set_xticks(xplot, xl)
    ax.set_xlabel(r"$r$ [$R_g$]")
    ax.set_ylim((1e-4, 1e2))
    ax.set_yscale("log")
    ax.set_ylabel(r"$\dot{M}/\dot{M}_\text{visc}$ [$-$]")
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position("right")
    ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
    ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
    ax.tick_params(axis='y', which='minor', length=l_log, width=w)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
            spine.set_linewidth(w)
    ax.grid()

    fig.suptitle(r"$t\omega_0 =$ " + f"{t[n_analysis]:.0f}")
    fig.tight_layout()
    plt.savefig(f"./output/plots/mass_{n}.png", bbox_inches='tight', dpi=300)
    plt.close()
    mass_plots.append(f"./output/plots/mass_{n}.png")

    quantities = {
        "densityFloor": densityFloor,
        "rho": rho,
        "inflow": inflow/c_s,
        "beta_0": beta_0
    }
    MASS_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, mass_xz_plots, f"mass_xz_{n}", t[n_analysis])
    MASS_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, mass_zoom_xz_plots, f"mass_zoom_xz_{n}", t[n_analysis])
    MASS_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, mass_yz_plots, f"mass_yz_{n}", t[n_analysis])
    MASS_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, mass_zoom_yz_plots, f"mass_zoom_yz_{n}", t[n_analysis])
    # ---------------------------------------------------------------------------------------------------------------------------------------------------

    # velocity plots ------------------------------------------------------------------------------------------------------------------------
    quantities = {
        "q_r": v_r/c_s,
        "q_th": v_theta/c_s,
        "q_phi": v_phi/c_s,
        "title_r": r"$v_r/c_s$ [$-$]",
        "title_th": r"$v_\theta/c_s$ [$-$]",
        "title_phi": r"$v_\varphi/c_s$ [$-$]",
        "beta_0": beta_0
    }
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, velocity_xz_plots, f"velocity_xz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, velocity_zoom_xz_plots, f"velocity_zoom_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, velocity_yz_plots, f"velocity_yz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, velocity_zoom_yz_plots, f"velocity_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": Vcrossh_r/(omega_orbit * c_s),
        "q_th": Vcrossh_th/(omega_orbit * c_s),
        "q_phi": Vcrossh_phi/(omega_orbit * c_s),
        "title_r": r'$(v\times h)_r/(\omega_0c_s)$ [$-$]',
        "title_th": r'$(v\times h)_\theta/(\omega_0c_s)$ [$-$]',
        "title_phi": r'$(v\times h)_\varphi/(\omega_0c_s)$ [$-$]',
        "beta_0": beta_0
    }
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, sourceTerm_xz_plots, f"sourceTerm_xz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, sourceTerm_zoom_xz_plots, f"sourceTerm_zoom_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, sourceTerm_yz_plots, f"sourceTerm_yz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, sourceTerm_zoom_yz_plots, f"sourceTerm_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": - gradP_r/(rho * omega_orbit * c_s) - gradPsi_r/(omega_orbit * c_s),
        "q_th": - gradP_th/(rho * omega_orbit * c_s),
        "q_phi": - gradP_phi/(rho * omega_orbit * c_s),
        "title_r": r'$- (\frac{1}{\rho}\nabla_r P + \nabla_r\Psi)(\omega_0c_s)$ [$-$]',
        "title_th": r'$- \frac{1}{\rho}\nabla_\theta P(\omega_0c_s)$ [$-$]',
        "title_phi": r'$- \frac{1}{\rho}\nabla_\varphi P(\omega_0c_s)$ [$-$]',
        "beta_0": beta_0
    }
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, gradP_xz_plots, f"gradP_xz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, gradP_zoom_xz_plots, f"gradP_zoom_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, gradP_yz_plots, f"gradP_yz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, gradP_zoom_yz_plots, f"gradP_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": v_adv_r/(omega_orbit * c_s),
        "q_th": v_adv_th/(omega_orbit * c_s),
        "q_phi": v_adv_phi/(omega_orbit * c_s),
        "title_r": r'$- (v \cdot \nabla) v_r/(\omega_0c_s)$ [$-$]',
        "title_th": r'$- (v \cdot \nabla) v_\theta/(\omega_0c_s)$ [$-$]',
        "title_phi": r'$- (v \cdot \nabla) v_\varphi/(\omega_0c_s)$ [$-$]',
        "beta_0": beta_0
    }
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, vAdv_xz_plots, f"vAdv_xz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, vAdv_zoom_xz_plots, f"vAdv_zoom_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, vAdv_yz_plots, f"vAdv_yz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, vAdv_zoom_yz_plots, f"vAdv_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": Svisc_r/(rho * omega_orbit * c_s),
        "q_th": Svisc_th/(rho * omega_orbit * c_s),
        "q_phi": Svisc_phi/(rho * omega_orbit * c_s),
        "title_r": r'$\text{S}_{\text{visc},r}/(\omega_0c_s)$ [$-$]',
        "title_th": r'$\text{S}_{\text{visc},\theta}/(\omega_0c_s)$ [$-$]',
        "title_phi": r'$\text{S}_{\text{visc},\varphi}/(\omega_0c_s)$ [$-$]',
        "beta_0": beta_0
    }
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, Svisc_xz_plots, f"Svisc_xz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, Svisc_zoom_xz_plots, f"Svisc_zoom_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, Svisc_yz_plots, f"Svisc_yz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, Svisc_zoom_yz_plots, f"Svisc_zoom_yz_{n}", t[n_analysis])

    # momentum plots ------------------------------------------------------------------------------------------------------------------------
    quantities = {
        "q_r": rho*v_r/c_s,
        "q_th": rho*v_theta/c_s,
        "q_phi": rho*v_phi/c_s,
        "title_r": r"$(\rho v_r)/(\rho c_s)$ [$-$]",
        "title_th": r"$(\rho v_\theta)/(\rho c_s)$ [$-$]",
        "title_phi": r"$(\rho v_\varphi)/(\rho c_s)$ [$-$]",
        "beta_0": beta_0
    }
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, rho_velocity_xz_plots, f"rho_velocity_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, rho_velocity_zoom_xz_plots, f"rho_velocity_zoom_xz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, rho_velocity_yz_plots, f"rho_velocity_yz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, rho_velocity_zoom_yz_plots, f"rho_velocity_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": rho*Vcrossh_r/(omega_orbit * c_s),
        "q_th": rho*Vcrossh_th/(omega_orbit * c_s),
        "q_phi": rho*Vcrossh_phi/(omega_orbit * c_s),
        "title_r": r'$\rho (v\times h)_r/(\rho_0 \omega_0c_s)$ [$-$]',
        "title_th": r'$\rho (v\times h)_\theta/(\rho_0 \omega_0c_s)$ [$-$]',
        "title_phi": r'$\rho (v\times h)_\varphi/(\rho_0 \omega_0c_s)$ [$-$]',
        "beta_0": beta_0
    }
    # # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, rho_sourceTerm_xz_plots, f"rho_sourceTerm_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, rho_sourceTerm_zoom_xz_plots, f"rho_sourceTerm_zoom_xz_{n}", t[n_analysis])
    # # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, rho_sourceTerm_yz_plots, f"rho_sourceTerm_yz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, rho_sourceTerm_zoom_yz_plots, f"rho_sourceTerm_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": - gradP_r/(omega_orbit * c_s) - rho*gradPsi_r/(omega_orbit * c_s),
        "q_th": - gradP_th/(omega_orbit * c_s),
        "q_phi": - gradP_phi/(omega_orbit * c_s),
        "title_r": r'$- (\nabla_r P + \rho \nabla_r\Psi)/(\rho_0 \omega_0c_s)$ [$-$]',
        "title_th": r'$- \nabla_\theta P/(\rho_0 \omega_0c_s)$ [$-$]',
        "title_phi": r'$- \nabla_\varphi P/(\rho_0 \omega_0c_s)$ [$-$]',
        "beta_0": beta_0
    }
    # # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, rho_gradP_xz_plots, f"rho_gradP_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, rho_gradP_zoom_xz_plots, f"rho_gradP_zoom_xz_{n}", t[n_analysis])
    # # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, rho_gradP_yz_plots, f"rho_gradP_yz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, rho_gradP_zoom_yz_plots, f"rho_gradP_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": rho*v_adv_r/(omega_orbit * c_s),
        "q_th": rho*v_adv_th/(omega_orbit * c_s),
        "q_phi": rho*v_adv_phi/(omega_orbit * c_s),
        "title_r": r'$- \rho (v \cdot \nabla) v_r/(\rho_0 \omega_0c_s)$ [$-$]',
        "title_th": r'$- \rho (v \cdot \nabla) v_\theta/(\rho_0 \omega_0c_s)$ [$-$]',
        "title_phi": r'$- \rho (v \cdot \nabla) v_\varphi/(\rho_0 \omega_0c_s)$ [$-$]',
        "beta_0": beta_0
    }
    # # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, rho_vAdv_xz_plots, f"rho_vAdv_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, rho_vAdv_zoom_xz_plots, f"rho_vAdv_zoom_xz_{n}", t[n_analysis])
    # # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, rho_vAdv_yz_plots, f"rho_vAdv_yz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, rho_vAdv_zoom_yz_plots, f"rho_vAdv_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": Svisc_r/(omega_orbit * c_s),
        "q_th": Svisc_th/(omega_orbit * c_s),
        "q_phi": Svisc_phi/(omega_orbit * c_s),
        "title_r": r'$\rho \text{S}_{\text{visc},r}/(\rho_0 \omega_0c_s)$ [$-$]',
        "title_th": r'$\rho \text{S}_{\text{visc},\theta}/(\rho_0 \omega_0c_s)$ [$-$]',
        "title_phi": r'$\rho \text{S}_{\text{visc},\varphi}/(\rho_0 \omega_0c_s)$ [$-$]',
        "beta_0": beta_0
    }
    # # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, rho_Svisc_xz_plots, f"rho_Svisc_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, rho_Svisc_zoom_xz_plots, f"rho_Svisc_zoom_xz_{n}", t[n_analysis])
    # # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, rho_Svisc_yz_plots, f"rho_Svisc_yz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, rho_Svisc_zoom_yz_plots, f"rho_Svisc_zoom_yz_{n}", t[n_analysis])

    # Local velocity plots ----------------------------------------------------------------------------------------------------------------------------
    quantities = {
        "q_r": v_r_Loc/c_s,
        "q_th": v_th_Loc/c_s,
        "q_phi": v_phi_Loc/c_s,
        "title_r": r"$v_{r,\text{Warped}}/c_s$ [$-$]",
        "title_th": r"$v_{\theta,\text{Warped}}/c_s$ [$-$]",
        "title_phi": r"$v_{\varphi,\text{Warped}}/c_s$ [$-$]",
        "beta_0": beta_0
    }
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, velocityLoc_xz_plots, f"velocityLoc_xz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, velocityLoc_zoom_xz_plots, f"velocityLoc_zoom_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, velocityLoc_yz_plots, f"velocityLoc_yz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, velocityLoc_zoom_yz_plots, f"velocityLoc_zoom_yz_{n}", t[n_analysis])

    quantities = {
        "q_r": rho*v_r_Loc/c_s,
        "q_th": rho*v_th_Loc/c_s,
        "q_phi": rho*v_phi_Loc/c_s,
        "title_r": r"$\rho v_{r,\text{Warped}}/(\rho_0 c_s)$ [$-$]",
        "title_th": r"$\rho v_{\theta,\text{Warped}}/(\rho_0 c_s)$ [$-$]",
        "title_phi": r"$\rho v_{\varphi,\text{Warped}}/(\rho_0 c_s)$ [$-$]",
        "beta_0": beta_0
    }
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, rho_velocityLoc_xz_plots, f"rho_velocityLoc_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, rho_velocityLoc_zoom_xz_plots, f"rho_velocityLoc_zoom_xz_{n}", t[n_analysis])
    # VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, rho_velocityLoc_yz_plots, f"rho_velocityLoc_yz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, rho_velocityLoc_zoom_yz_plots, f"rho_velocityLoc_zoom_yz_{n}", t[n_analysis])

    # rotation curve plots ----------------------------------------------------------------------------------------------------------------------------
    omega_K = KEPLER(r_vtk)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ax = axs[0]
    ax.plot(r, omega**2/omega_K**2, color="tab:brown")
    ax.set_xlim((0, r_max))
    xplot = np.linspace(0, r_max , 7)
    xl = [f"{i:.0f}" for i in xplot]
    xl[0] = ""
    xl[-1] = ""
    ax.set_xticks(xplot, xl)
    ax.set_xlabel(r"$r$ [$R_g$]")
    ax.set_ylim((1e-4, 1e2))
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Omega^2/\Omega_K^2$ [$-$]")
    ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
    ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
    ax.tick_params(axis='y', which='minor', length=l_log, width=w)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
            spine.set_linewidth(w)
    ax.grid()

    ax = axs[1]
    ax.plot(r, kappa2/omega_K**2, color="tab:brown")
    ax.set_xlim((0, r_max))
    xplot = np.linspace(0, r_max , 7)
    xl = [f"{i:.0f}" for i in xplot]
    xl[0] = ""
    xl[-1] = ""
    ax.set_xticks(xplot, xl)
    ax.set_xlabel(r"$r$ [$R_g$]")
    ax.set_ylim((1e-4, 1e2))
    ax.set_yscale("log")
    ax.set_ylabel(r"$\kappa^2/\Omega_K^2$ [$-$]")
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position("right")
    ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
    ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
    ax.tick_params(axis='y', which='minor', length=l_log, width=w)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
            spine.set_linewidth(w)
    ax.grid()
    
    fig.suptitle(r"$t\omega_0 =$ " + f"{t[n_analysis]:.0f}")
    fig.tight_layout()
    plt.savefig(f"./output/plots/rotation_{n}.png", bbox_inches='tight', dpi=300)
    plt.close()
    rotation_plots.append(f"./output/plots/rotation_{n}.png")
    # -------------------------------------------------------------------------------------------------------------------------------------------------

    # 3D plots ----------------------------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')

    r_3D = np.zeros(r_vtk.size + 1)
    r_3D[1:-1] = (r_vtk[:-1] + r_vtk[1:]) / 2
    r_3D[0] = r_min
    r_3D[-1] = r_max
    theta_3D = np.zeros(theta_vtk.size + 1)
    theta_3D[1:-1] = (theta_vtk[:-1] + theta_vtk[1:]) / 2
    theta_3D[0] = 0
    theta_3D[-1] = np.pi
    phi_3D = np.zeros(phi_vtk.size + 1)
    phi_3D[1:-1] = (phi_vtk[:-1] + phi_vtk[1:]) / 2
    phi_3D[0] = 0
    phi_3D[-1] = 2*np.pi

    PHI, TH, R = np.meshgrid(phi_3D, theta_3D, r_3D, indexing="ij")
    X, Y, Z = R*np.sin(TH)*np.cos(PHI), R*np.sin(TH)*np.sin(PHI), R*np.cos(TH)

    wh = rho > 0.05
    facecolors = np.where(wh, 'tab:blue', 'none')
    ax.voxels(X,Y,Z, wh, facecolors=facecolors)

    x = r_min * np.outer(np.cos(phi_vtk), np.sin(theta_vtk))
    y = r_min * np.outer(np.sin(phi_vtk), np.sin(theta_vtk))
    z = r_min * np.outer(np.ones(np.size(phi_vtk)), np.cos(theta_vtk))
    ax.plot_surface(x,y,z, color="black")

    if (beta_0 != 0):
        wh = X > 1/np.sqrt(1 + 1/np.sin(-beta_0)**2)
        ax.plot(X[wh], np.zeros_like(Y)[wh], X[wh]/np.sin(-beta_0), color="white")
        wh = X < -1/np.sqrt(1 + 1/np.sin(-beta_0)**2)
        ax.plot(X[wh], np.zeros_like(Y)[wh], X[wh]/np.sin(-beta_0), color="white")
    else:
        ax.plot(np.zeros_like(X), np.zeros_like(Y), Z, color="white")

    ax.set_xlim((-5,5))
    ax.set_ylim((-5,5))
    ax.set_zlim((-5,5))
    ax.set_axis_off()
    ax.set_facecolor("black")
    ax.view_init(elev=10, azim=90, roll=beta_0*180/np.pi)

    fig.tight_layout()
    plt.savefig(f"./output/plots/3D_{n}.png", bbox_inches='tight', dpi=300)
    plt.close()
    threeD_plots.append(f"./output/plots/3D_{n}.png")
    # -------------------------------------------------------------------------------------------------------------------------------------------------
MOVIE(mass_plots, "mass")
MOVIE(mass_xz_plots, "mass_xz")
MOVIE(mass_zoom_xz_plots, "mass_zoom__xz")
MOVIE(mass_yz_plots, "mass_yz")
MOVIE(mass_zoom_yz_plots, "mass_zoom_yz")

MOVIE(velocity_xz_plots, "velocity_xz")
# MOVIE(velocity_zoom_xz_plots, "velocity_zoom_xz")
MOVIE(velocity_yz_plots, "velocity_yz")
# MOVIE(velocity_zoom_yz_plots, "velocity_zoom_yz")

MOVIE(sourceTerm_xz_plots, "sourceTerm_xz")
# MOVIE(sourceTerm_zoom_xz_plots, "sourceTerm_zoom_xz")
MOVIE(sourceTerm_yz_plots, "sourceTerm_yz")
# MOVIE(sourceTerm_zoom_yz_plots, "sourceTerm_zoom_yz")

MOVIE(gradP_xz_plots, "gradP_xz")
# MOVIE(gradP_zoom_xz_plots, "gradP_zoom_xz")
MOVIE(gradP_yz_plots, "gradP_yz")
# MOVIE(gradP_zoom_yz_plots, "gradP_zoom_yz")

MOVIE(vAdv_xz_plots, "vAdv_xz")
# MOVIE(vAdv_zoom_xz_plots, "vAdv_zoom_xz")
MOVIE(vAdv_yz_plots, "vAdv_yz")
# MOVIE(vAdv_zoom_yz_plots, "vAdv_zoom_yz")

MOVIE(Svisc_xz_plots, "Svisc_xz")
# MOVIE(Svisc_zoom_xz_plots, "Svisc_zoom_xz")
MOVIE(Svisc_yz_plots, "Svisc_yz")
# MOVIE(Svisc_zoom_yz_plots, "Svisc_zoom_yz")

# MOVIE(rho_velocity_xz_plots, "rho_velocity_xz")
MOVIE(rho_velocity_zoom_xz_plots, "rho_velocity_zoom_xz")
# MOVIE(rho_velocity_yz_plots, "rho_velocity_yz")
MOVIE(rho_velocity_zoom_yz_plots, "rho_velocity_zoom_yz")

# MOVIE(rho_sourceTerm_xz_plots, "rho_sourceTerm_xz")
MOVIE(rho_sourceTerm_zoom_xz_plots, "rho_sourceTerm_zoom_xz")
# MOVIE(rho_sourceTerm_yz_plots, "rho_sourceTerm_yz")
MOVIE(rho_sourceTerm_zoom_yz_plots, "rho_sourceTerm_zoom_yz")

# MOVIE(rho_gradP_xz_plots, "rho_gradP_xz")
MOVIE(rho_gradP_zoom_xz_plots, "rho_gradP_zoom_xz")
# MOVIE(rho_gradP_yz_plots, "rho_gradP_yz")
MOVIE(rho_gradP_zoom_yz_plots, "rho_gradP_zoom_yz")

# MOVIE(rho_vAdv_xz_plots, "rho_vAdv_xz")
MOVIE(rho_vAdv_zoom_xz_plots, "rho_vAdv_zoom_xz")
# MOVIE(rho_vAdv_yz_plots, "rho_vAdv_yz")
MOVIE(rho_vAdv_zoom_yz_plots, "rho_vAdv_zoom_yz")

# MOVIE(rho_Svisc_xz_plots, "rho_Svisc_xz")
MOVIE(rho_Svisc_zoom_xz_plots, "rho_Svisc_zoom_xz")
# MOVIE(rho_Svisc_yz_plots, "rho_Svisc_yz")
MOVIE(rho_Svisc_zoom_yz_plots, "rho_Svisc_zoom_yz")

MOVIE(velocityLoc_xz_plots, "velocityLoc_xz")
# MOVIE(velocityLoc_zoom_xz_plots, "velocityLoc_zoom_xz")
MOVIE(velocityLoc_yz_plots, "velocityLoc_yz")
# MOVIE(velocityLoc_zoom_yz_plots, "velocityLoc_zoom_yz")

# MOVIE(rho_velocityLoc_xz_plots, "rho_velocityLoc_xz")
MOVIE(rho_velocityLoc_zoom_xz_plots, "rho_velocityLoc_zoom_xz")
# MOVIE(rho_velocityLoc_yz_plots, "rho_velocityLoc_yz")
MOVIE(rho_velocityLoc_zoom_yz_plots, "rho_velocityLoc_zoom_yz")

MOVIE(rotation_plots, "rotation")
MOVIE(threeD_plots, "3D")
