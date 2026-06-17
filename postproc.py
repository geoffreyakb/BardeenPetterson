from src import *
# ----------------------------------------------------------------------------------
# Plotting parameters
# ----------------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'text.latex.preamble':r'\usepackage{amsmath}',
    'text.latex.preamble':r'\usepackage{fourier}'
})
w, l, l_log = 1.25, 10, 6
pad, lpad, shrink = 0.12, 5, 0.8

# ----------------------------------------------------------------------------------
# Getting the input data
# ----------------------------------------------------------------------------------
conf = inifix.load("idefix.ini")
n_average = int(conf["TimeIntegrator"]["tstop"] / conf["Output"]["analysis"]) + 1
n_vtk = int(conf["TimeIntegrator"]["tstop"] / conf["Output"]["vtk"]) + 1
r_min = conf["Grid"]["X1-grid"][1]
r_max = conf["Grid"]["X1-grid"][-1]
n_r = conf["Grid"]["X1-grid"][2]
epsilon = conf["Setup"]["epsilon"]
alpha = conf["Setup"]["alpha"]
densityFloor = conf["Setup"]["densityFloor"]
beta_0 = conf["Setup"]["tilt"] * np.pi/180

# Reading the analysis files
t, M_tot = READ_BOX_AVERAGE()
r, Sigma, Tilt, Precession, L = READ_RADIAL_AVERAGE(n_average, n_r)

# Calculating the normalizators
r_norm = r_min
t_orbit = 2*np.pi*(r_norm**1.5) / np.sqrt(1 - 2.5*epsilon**2)
n_orbit = t[-1]/t_orbit
t /= t_orbit

Tilt_plots = []
for n in range(n_average):
    if n % 10 == 0:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        ax = axs[0]
        ax.plot(r, Tilt[n,:], color="black")
        ax.set_xlim((r_min, r_max))
        xplot = np.linspace(r_min, r_max, 5)
        xl = [f"{i:.1f}" for i in xplot]
        ax.set_xticks(xplot, xl)
        ax.set_ylim((0,beta_0*180/np.pi))
        ax.set_xlabel(r"$r$ $[$Code Units$]$")
        ax.set_ylabel(r"$\beta$ $[$°$]$")
        ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
        ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
        for spine in ax.spines.values():
                spine.set_linewidth(w)
        ax.grid()

        ax = axs[1]
        ax.plot(r, Precession[n,:], color="black")
        ax.set_xlim((r_min, r_max))
        xplot = np.linspace(r_min, r_max, 5)
        xl = [f"{i:.1f}" for i in xplot]
        ax.set_xticks(xplot, xl)
        ax.set_ylim((-180,180))
        ax.set_xlabel(r"$r$ $[$Code Units$]$")
        ax.set_ylabel(r"$\gamma$ $[$°$]$")
        ax.tick_params(axis='y', which='both', direction='in', right=True, width=w, length=l)
        ax.tick_params(axis='x', which='both', direction='in', top=True, width=w, length=l)
        for spine in ax.spines.values():
                spine.set_linewidth(w)
        ax.grid()

        fig.suptitle(r"$t\omega_\mathrm{orbit} =$ " + rf"{t[n]:.0f}")
        fig.tight_layout()
        plt.savefig(f"./output/plots/Tilt_{n}.png", bbox_inches='tight', dpi=200)
        Tilt_plots.append(f"./output/plots/Tilt_{n}.png")
        plt.close()

MOVIE(Tilt_plots, "tilt")

mass_plots = []
mass_zoom_plots = []
velocity_xz_plots = []
velocity_zoom_xz_plots = []
velocity_yz_plots = []
velocity_zoom_yz_plots = []
InvDT_plots = []
InvDT_zoom_plots = []

for n in range(n_vtk):
    n_analysis = int(n * conf["Output"]["vtk"] // conf["Output"]["analysis"])
    r_vtk, theta_vtk, phi_vtk, rho, v_r, v_theta, v_phi, InvDT = READ_VTK(n)
    PHI, TH, R = np.meshgrid(phi_vtk, theta_vtk, r_vtk, indexing="ij")

    c_s = epsilon / np.sqrt(R)

    # mass plots ------------------------------------------------------------------------------------------------------------------------
    quantities = {
        "densityFloor": densityFloor,
        "rho": rho,
        "beta_0": beta_0,
        "r_norm": r_norm
    }
    MASS_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, False, mass_plots, f"mass_{n}", t[n_analysis])
    MASS_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, True, mass_zoom_plots, f"mass_zoom_{n}", t[n_analysis])

    # velocity plots ------------------------------------------------------------------------------------------------------------------------
    quantities = {
        "q_r": v_r,
        "q_th": v_theta,
        "q_phi": v_phi,
        "buff_r": np.max(np.abs(v_r)),
        "buff_th": np.max(np.abs(v_theta)),
        "buff_phi": np.max(np.abs(v_phi)),
        "title_r": r"$v_r$ $[$Code Units$]$",
        "title_th": r"$v_\theta$ $[$Code Units$]$",
        "title_phi": r"$v_\varphi$ $[$Code Units$]$",
        "beta_0": beta_0,
        "r_norm": r_norm,
        "rho": rho,
        "densityFloor": densityFloor
    }
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", False, velocity_xz_plots, f"velocity_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "xz", True, velocity_zoom_xz_plots, f"velocity_zoom_xz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", False, velocity_yz_plots, f"velocity_yz_{n}", t[n_analysis])
    VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, "yz", True, velocity_zoom_yz_plots, f"velocity_zoom_yz_{n}", t[n_analysis])

    if n > 0:
        quantities = {
            "InvDT": InvDT,
            "beta_0": beta_0,
            "r_norm": r_norm
        }
        INVDT_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, False, InvDT_plots, f"InvDT_{n}", t[n_analysis])
        INVDT_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, True, InvDT_zoom_plots, f"InvDT_zoom_{n}", t[n_analysis])

MOVIE(mass_plots, "mass")
MOVIE(mass_zoom_plots, "mass_zoom")
MOVIE(velocity_xz_plots, "velocity_xz")
MOVIE(velocity_zoom_xz_plots, "velocity_zoom_xz")
MOVIE(velocity_yz_plots, "velocity_yz")
MOVIE(velocity_zoom_yz_plots, "velocity_zoom_yz")
MOVIE(InvDT_plots, "InvDT")
MOVIE(InvDT_zoom_plots, "InvDT_zoom")
