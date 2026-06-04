import os
import sys
sys.path.append(os.getenv("IDEFIX_DIR"))
import csv
import numpy as np
import matplotlib.pyplot as plt
from pytools.vtk_io import readVTK
import inifix
import cv2
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors

w, l, l_log = 1.25, 10, 6
pad, lpad, shrink = 0.05, 5, 0.8
formats = tkr.FormatStrFormatter('%.1e')

def READ_BOX_AVERAGE():
    fid = open("output/analysis/globalAverage.dat", "r")
    varnames = fid.readline().split()
    fid.close()
    data = np.loadtxt("output/analysis/globalAverage.dat",skiprows=1)
    V = {}
    i = 0
    for name in varnames:
        V[name] = data[:,i]           
        i += 1

    return V["t"], V["Mtot"]

def READ_RADIAL_AVERAGE(n_average, n_r, beta_0):
    Sigma = np.zeros((n_average, n_r))
    rho = np.zeros((n_average, n_r))
    L = np.zeros((n_average, n_r, 3))
    beta = np.zeros((n_average, n_r))      # Tilt angle
    gamma = np.zeros((n_average, n_r))       # Precession angle
    LBH = np.zeros((n_average, n_r, 3))
    betaBH = np.zeros((n_average, n_r))
    gammaBH = np.zeros((n_average, n_r))
    rho_Vr = np.zeros((n_average, n_r))

    current_number = 0
    for i in range(n_average):
        if i < 10:
            current_number = f"000{i}"
        elif i < 100:
            current_number = f"00{i}"
        elif i < 1000:
            current_number = f"0{i}"
        else:
            current_number = i

        fid=open(f"output/analysis/radialAverage_{current_number}.dat","r")
        varnames=fid.readline().split()
        fid.close()
        data=np.loadtxt(f"output/analysis/radialAverage_{current_number}.dat",skiprows=1)
        V={}
        j=0
        for name in varnames:
            V[name]=data[:,j]
            j=j+1

        Sigma[i,:] = V['Sigma']
        rho[i,:] = Sigma[i,:] / (2*V["r"])
        L[i,:,0] = V['Lx']
        L[i,:,1] = V['Ly']
        L[i,:,2] = V['Lz']
        LBH[i,:,0] = np.cos(beta_0)*L[i,:,0] + np.sin(beta_0)*L[i,:,2]
        LBH[i,:,1] = L[i,:,1]
        LBH[i,:,2] = -np.sin(beta_0)*L[i,:,0] + np.cos(beta_0)*L[i,:,2]

        norm = np.sqrt(L[i,:,0]**2 + L[i,:,1]**2 + L[i,:,2]**2)
        beta[i,:] = np.arccos(L[i,:,2] / norm) * 180/np.pi
        gamma[i,:] = np.arctan2(L[i,:,1], L[i,:,0]) * 180/np.pi
        norm = np.sqrt(LBH[i,:,0]**2 + LBH[i,:,1]**2 + LBH[i,:,2]**2)
        betaBH[i,:] = np.arccos(LBH[i,:,2] / norm) * 180/np.pi
        gammaBH[i,:] = np.arctan2(LBH[i,:,1], LBH[i,:,0]) * 180/np.pi

        rho_Vr[i,:] = V['rho_Vr']
    
    wh = (beta < 1e-9)
    gamma[wh] *= 0
    wh = (betaBH < 1e-9)
    gammaBH[wh] *= 0
        
    return V["r"], Sigma, rho, L, beta, gamma, LBH, betaBH, gammaBH, rho_Vr

def READ_VTK(n_vtk):
    NVAR = 4
    RHO = 0
    VX1 = 1
    VX2 = 2
    VX3 = 3

    if n_vtk >= 1000:
        current_number = str(n_vtk)
    elif n_vtk >= 100:
        current_number = '0' + str(n_vtk)
    elif n_vtk >= 10:
        current_number = '00' + str(n_vtk)
    else:
        current_number = '000' + str(n_vtk)
    current_VTK = readVTK('output/vtk/data.' + current_number + '.vtk', geometry='spherical')    
    
    r_vtk = current_VTK.r
    theta_vtk = current_VTK.theta
    phi_vtk = current_VTK.phi

    vtk = np.zeros((NVAR, phi_vtk.size, theta_vtk.size, r_vtk.size), dtype=float)
    vtk[RHO,:,:,:] = np.moveaxis(current_VTK.data['RHO'], [0, 2], [2, 0])
    vtk[VX1,:,:,:] = np.moveaxis(current_VTK.data['VX1'], [0, 2], [2, 0])
    vtk[VX2,:,:,:] = np.moveaxis(current_VTK.data['VX2'], [0, 2], [2, 0])
    vtk[VX3,:,:,:] = np.moveaxis(current_VTK.data['VX3'], [0, 2], [2, 0])

    rho = vtk[RHO,:,:,:]
    v_r = vtk[VX1,:,:,:]
    v_theta = vtk[VX2,:,:,:]
    v_phi = vtk[VX3,:,:,:]

    return r_vtk, theta_vtk, phi_vtk, rho, v_r, v_theta, v_phi

def MOVIE(plots, name):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20
    images = plots
    height, width, _ = cv2.imread(images[0]).shape
    video_summary = cv2.VideoWriter(filename=f"./plots/{name}.mp4", fourcc=fourcc, fps=fps, frameSize=(width, height))
    for image in images:
        video_summary.write(cv2.imread(image))
    # cv2.destroyAllWindows()
    video_summary.release()

def MASS_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, zoom, list_plots, plots_name, time):
    densityFloor = quantities["densityFloor"]
    rho = quantities["rho"]
    v_r = quantities["v_r"]
    v_theta = quantities["v_theta"]

    R, TH = np.meshgrid(r_vtk, theta_vtk)
    phi_cut_plus = np.where(phi_vtk >= 0)[0][0]
    X_cut_plus, Z = R*np.sin(TH)*np.cos(phi_vtk[phi_cut_plus]), R*np.cos(TH)
    x_label = r"$x$ [$R_g$]"
    y_label = r"$z$ [$R_g$]"

    fig, axs = plt.subplots(1, figsize=(5, 12))
   
    ax = axs
    ax.set_title(r'$\log(\rho/\rho_0)$ $[-]$')
    ticks = np.linspace(np.log10(densityFloor), 0, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, np.log10(rho[phi_cut_plus,:,:]), cmap="inferno", vmin=np.log10(densityFloor), vmax=0)
    formats = tkr.FormatStrFormatter('%.0f')
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    ax.set_facecolor("dimgray")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if zoom == True:
        ax.set_xlim((0,4*r_min))
        xplot = np.linspace(0, 4*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-4*r_min, 4*r_min))
        yplot = np.linspace(-4*r_min, 4*r_min, 5)
        ax.set_yticks(yplot)
    elif zoom == False:
        ax.set_xlim((0,r_max))
        xplot = np.linspace(0, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        yplot = np.linspace(-r_max, r_max, 5)
        ax.set_yticks(yplot)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    # ax = axs[1]
    # ax.set_title(r'$E_K / (\rho_0 c_s^2)$ $[-]$')
    # buff = np.max(np.abs(E_K))
    # ticks = np.linspace(0, buff, 5)
    # pc0 = ax.pcolormesh(X_cut_plus, Z, E_K[phi_cut_plus,:,:], cmap="inferno", vmin=0, vmax=buff)
    # formats = tkr.FormatStrFormatter('%.1f')
    # cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    # ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    # ax.set_facecolor("dimgray")
    # ax.set_xlabel(x_label)
    # if zoom == True:
    #     ax.set_xlim((0,4*r_min))
    #     xplot = np.linspace(0, 4*r_min, 5)
    #     xl = [f"{i:.1f}" for i in xplot]
    #     xl[0] = ""
    #     xl[-1] = ""
    #     ax.set_xticks(xplot,xl)
    #     ax.set_ylim((-4*r_min, 4*r_min))
    #     yplot = np.linspace(-4*r_min, 4*r_min,5)
    #     yl = ["" for i in yplot]
    #     ax.set_yticks(yplot, yl)
    # elif zoom == False:
    #     ax.set_xlim((0,r_max))
    #     xplot = np.linspace(0, r_max, 5)
    #     xl = [f"{i:.0f}" for i in xplot]
    #     xl[0] = ""
    #     xl[-1] = ""
    #     ax.set_xticks(xplot,xl)
    #     ax.set_ylim((-r_max, r_max))
    #     yplot = np.linspace(-r_max, r_max, 5)
    #     yl = ["" for i in yplot]
    #     ax.set_yticks(yplot, yl)
    # ax.yaxis.set_ticks_position('both')
    # ax.xaxis.set_ticks_position('both')
    # for spine in ax.spines.values():
    #     spine.set_linewidth(w)

    # ax = axs[2]
    # ax.set_title(r'$L^2 / (\rho_0^2 R_g^2 c_s^2)$ $[-]$')
    # buff = np.max(np.abs(L2))
    # ticks = np.linspace(0, buff, 5)
    # pc0 = ax.pcolormesh(X_cut_plus, Z, L2[phi_cut_plus,:,:], cmap="inferno", vmin=0, vmax=buff)
    # formats = tkr.FormatStrFormatter('%.1f')
    # cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    # ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    # ax.set_facecolor("dimgray")
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
    # if zoom == True:
    #     ax.set_xlim((0,4*r_min))
    #     xplot = np.linspace(0, 4*r_min, 5)
    #     xl = [f"{i:.1f}" for i in xplot]
    #     xl[0] = ""
    #     xl[-1] = ""
    #     ax.set_xticks(xplot,xl)
    #     ax.set_ylim((-4*r_min, 4*r_min))
    #     yplot = np.linspace(-4*r_min, 4*r_min,5)
    #     yl = ["" for i in yplot]
    #     ax.set_yticks(yplot, yl)
    # elif zoom == False:
    #     ax.set_xlim((0,r_max))
    #     xplot = np.linspace(0, r_max, 5)
    #     xl = [f"{i:.0f}" for i in xplot]
    #     xl[0] = ""
    #     xl[-1] = ""
    #     ax.set_xticks(xplot,xl)
    #     ax.set_ylim((-r_max, r_max))
    #     yplot = np.linspace(-r_max, r_max, 5)
    #     yl = ["" for i in yplot]
    #     ax.set_yticks(yplot, yl)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    # ax.yaxis.set_ticks_position('both')
    # ax.xaxis.set_ticks_position('both')
    # for spine in ax.spines.values():
    #     spine.set_linewidth(w)

    fig.suptitle(r"$t\omega_\mathrm{orbit} =$ " + f"{time:.0f}", x=0.5, y=0.98)
    fig.tight_layout()
    plt.savefig(f"./output/plots/{plots_name}.png", bbox_inches='tight', dpi=300)
    plt.close()
    list_plots.append(f"./output/plots/{plots_name}.png")

def VELOCITY_PLOT(r_vtk, r_min, r_max, theta_vtk, phi_vtk, quantities, zoom, list_plots, plots_name, time):
    q_r = quantities["q_r"]
    q_th = quantities["q_th"]
    q_phi = quantities["q_phi"]
    title_r = quantities["title_r"]
    title_th = quantities["title_th"]
    title_phi = quantities["title_phi"]

    R, TH = np.meshgrid(r_vtk, theta_vtk)
    phi_cut_plus = np.where(phi_vtk >= 0)[0][0]
    X_cut_plus, Z = R*np.sin(TH)*np.cos(phi_vtk[phi_cut_plus]), R*np.cos(TH)
    x_label = r"$x$ [$R_g$]"
    y_label = r"$z$ [$R_g$]"

    fig, axs = plt.subplots(1, 3, gridspec_kw={'wspace': 0.01}, figsize=(15, 12))

    ax = axs[0]
    ax.set_title(title_r)
    buff = np.max(np.abs(q_r))
    ticks = np.linspace(-buff, buff, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, q_r[phi_cut_plus,:,:], cmap="berlin", vmin=-buff, vmax=buff)
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    ax.set_facecolor("dimgray")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if zoom == True:
        ax.set_xlim((0,4*r_min))
        xplot = np.linspace(0, 4*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-4*r_min, 4*r_min))
        yplot = np.linspace(-4*r_min, 4*r_min, 5)
        ax.set_yticks(yplot)
    elif zoom == False:
        ax.set_xlim((0,r_max))
        xplot = np.linspace(0, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        yplot = np.linspace(-r_max, r_max, 5)
        ax.set_yticks(yplot)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    ax = axs[1]
    ax.set_title(title_th)
    buff = np.max(np.abs(q_th))
    ticks = np.linspace(-buff, buff, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, q_th[phi_cut_plus,:,:], cmap="berlin", vmin=-buff, vmax=buff)
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    ax.set_facecolor("dimgray")
    ax.set_xlabel(x_label)
    if zoom == True:
        ax.set_xlim((0,4*r_min))
        xplot = np.linspace(0, 4*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-4*r_min, 4*r_min))
        yplot = np.linspace(-4*r_min, 4*r_min,5)
        yl = ["" for i in yplot]
        ax.set_yticks(yplot, yl)
    elif zoom == False:
        ax.set_xlim((0,r_max))
        xplot = np.linspace(0, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        yplot = np.linspace(-r_max, r_max, 5)
        yl = ["" for i in yplot]
        ax.set_yticks(yplot, yl)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    ax = axs[2]
    ax.set_title(title_phi)
    buff = np.max(np.abs(q_phi))
    ticks = np.linspace(-buff, buff, 5)
    pc0 = ax.pcolormesh(X_cut_plus, Z, q_phi[phi_cut_plus,:,:], cmap="berlin", vmin=-buff, vmax=buff)
    cbar = fig.colorbar(pc0, ax=ax, location="bottom", pad=pad, shrink=shrink, format=formats, ticks=ticks)
    ax.tick_params(axis='both', direction='in', color='white', width=w, length=l, pad=lpad)
    ax.set_facecolor("dimgray")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if zoom == True:
        ax.set_xlim((0,4*r_min))
        xplot = np.linspace(0, 4*r_min, 5)
        xl = [f"{i:.1f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-4*r_min, 4*r_min))
        yplot = np.linspace(-4*r_min, 4*r_min,5)
        yl = ["" for i in yplot]
        ax.set_yticks(yplot, yl)
    elif zoom == False:
        ax.set_xlim((0,r_max))
        xplot = np.linspace(0, r_max, 5)
        xl = [f"{i:.0f}" for i in xplot]
        xl[0] = ""
        xl[-1] = ""
        ax.set_xticks(xplot,xl)
        ax.set_ylim((-r_max, r_max))
        yplot = np.linspace(-r_max, r_max, 5)
        yl = ["" for i in yplot]
        ax.set_yticks(yplot, yl)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    fig.suptitle(r"$t\omega_\mathrm{orbit} =$ " + f"{time:.0f}", x=0.5, y=0.915)
    fig.tight_layout()
    plt.savefig(f"./output/plots/{plots_name}.png", bbox_inches='tight', dpi=300)
    plt.close()
    list_plots.append(f"./output/plots/{plots_name}.png")

def THEORETICAL_ROTATION_CURVE(R, Tilt_init, spin, gravity, v_r, dr_v_r, r_vtk):
    tilt = Tilt_init * np.pi / 180

    a = R
    b = 2 * np.cos(tilt) * spin / R**2
    if gravity == "Kepler":
        c = - 1/R**2
    elif gravity == "PW":
        c = - 1/(R - 2)**2
    elif gravity == "Einstein":
        c = - 1/R**2 - 6/R**3
    c -= v_r * dr_v_r

    delta = b**2 - 4*a*c
    omega_th_p = (-b + np.sqrt(delta)) / (2*a)
    omega_th_m = (-b - np.sqrt(delta)) / (2*a)
    
    kappa_2_th_p = 4*omega_th_p**2 + 2*R*omega_th_p*np.gradient(omega_th_p, r_vtk)
    kappa_2_th_m = 4*omega_th_m**2 + 2*R*omega_th_m*np.gradient(omega_th_m, r_vtk)

    return omega_th_p, omega_th_m, kappa_2_th_p, kappa_2_th_m

def KEPLER(r_vtk):
    return np.sqrt(1/r_vtk**3)
