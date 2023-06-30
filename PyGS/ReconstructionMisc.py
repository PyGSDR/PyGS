"""
This file includes several functions needed for the GS reconstruction.
Mainly translated from the original GSR codes in Matlab, 
i.e., prefr.m, hu12n.m, hu34n.m, Kr_homo.m, Ap_homo, etc.

Authors: Yu Chen & Qiang Hu
Affiliation: CSPAR & Department of Space Science @ The University of Alabama in Huntsville

References: 
- Hu, Q., & Sonnerup, B. U. Ö. 2002, JGR, 107, 1142
- Hu, Q., & Dasgupta, B. 2005. GRL, 32: L12109
- Teh, W.-L. 2018, EP&S, 70, 1
- Chen, Y., Hu, Q., Zhao, L., Kasper, J. C., & Huang, J. 2021, ApJ, 914, 108

Version 0.0.1 @ June 2023

"""
from __future__ import division
import os
import pickle
import math
import time
import sympy
import numpy as np
import pandas as pd
from numpy import linalg as la
import pandas as pd
import scipy
from scipy import signal
from scipy import integrate
from scipy import interpolate
from datetime import datetime
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from PyGS import FluxRopeDetection as detect
from PyGS import obtainAxis

def plotXSeries(X, A, B, P1, P2, PT, Beta):
    """Plot parameters along the spacecraft path,
    which include A', the magnetic field in the flux rope frame,
    three pressures with a prime symbol, and plasma/proton beta.
    Will be presented when adjustAxis = True. """

    fig, scpath = plt.subplots(4, 1, figsize=(6, 6))
    plt.subplots_adjust(hspace = 0.5)

    # Plot A' along s/c path
    plot_A = scpath[0]
    plot_A.grid(color=(0.9,0.9,0.9), zorder=0)
    plot_A.plot(X, A, color='k')
    plot_A.set_ylabel("$A'$")
    
    # Plot B_inFR along s/c path
    # Note that Bx, By, Bz that have lowercase subscripts
    # represent three components in FR frame.
    plot_B = scpath[1]
    plot_B.grid(color=(0.9,0.9,0.9),zorder=0)
    plot_B.plot(X, B[0]/1e-9, color='r', label='$B_x$')
    plot_B.plot(X, B[1]/1e-9, color='g', label='$B_y$')
    plot_B.plot(X, B[2]/1e-9, color='b', label='$B_z$')
    plot_B.set_ylabel('$B_{inFR}$ (nT)')
    plot_B.legend(loc='center left', prop={'size':7}, bbox_to_anchor=(1.01, 0.5))

    # Plot pressures along s/c path
    # PB' = magnetic pressure'
    # Ppe' = thermal pressure' including electron data if available
    # Pt' = transverse pressure'
    plot_P = scpath[2]
    plot_P.grid(color=(0.9,0.9,0.9), zorder=0)
    plot_P.plot(X, P1, color='r', label="$P_B'$")
    plot_P.plot(X, P2, color='g', label="$P_{pe}'$")
    plot_P.plot(X, PT, color='b', label="$P_t'$")
    plot_P.set_ylabel("$Pressure'$")
    plot_P.legend(loc='center left', prop={'size':7}, bbox_to_anchor=(1.01, 0.5))

    # Plot plasma beta along s/c path
    # If electron is unavailable or not included,
    # This beta is proton beta only.
    plot_beta = scpath[3]
    plot_beta.grid(color=(0.9,0.9,0.9), zorder=0)
    plot_beta.plot(X, Beta, color='k')
    plot_beta.set_ylabel(r'$\beta$')

    fig.savefig(rootDir + 'axis_parameters_alongsc.png', format='png', dpi=450)
    # Save figure.
    if saveFig:
        fig.savefig(rootDir + spacecraftID + startTime.strftime('%Y%m%d%H%M%S') 
            + '_parameters_alongsc.png', format='png', dpi=450)
    return

def plotPressureA(A, P, B, P1, P2, PtA_coeff, **kwargs):
    # Plot four pressures versus A'.
    
    fig1, versusA = plt.subplots(2, 2, figsize=(7, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.27)
    
    # Plot Pt' versus A'.
    plot_Pt_vs_A = versusA[0][0]
    plot_Pt_vs_A.grid(color=(0.9,0.9,0.9), zorder=0)
    plot_Pt_vs_A.plot(A, P, '--', color='k')
    PtA2, = plot_Pt_vs_A.plot(A, PtA_coeff, '-', color='tab:blue')
    PtA0, = plot_Pt_vs_A.plot(A[0], P[0], marker='d', markersize=5, color='tab:red')
    PtA1, = plot_Pt_vs_A.plot(A[-1], P[-1], marker='X', markersize=6, color='g')
    plot_Pt_vs_A.set_xlabel("$A'$")
    plot_Pt_vs_A.set_ylabel("$P_t'$")
    plot_Pt_vs_A.legend(handles=[PtA0, PtA1, PtA2], 
        labels=['start','end','fitting'], loc='best',prop={'size':8})
    
    # Plot Bz' versus A'.
    # Bz = the axial magnetic field'
    plot_Bz_vs_A = versusA[0][1]
    plot_Bz_vs_A.grid(color=(0.9,0.9,0.9),zorder=0)
    plot_Bz_vs_A.plot(A, B[2], color='k')
    plot_Bz_vs_A.set_xlabel("$A'$")
    plot_Bz_vs_A.set_ylabel(r'$(1-\alpha)B_z$')

    # Plot Ppe' versus A'.
    # Include electron data if available.
    plot_Ppe_vs_A = versusA[1][0]
    plot_Ppe_vs_A.grid(color=(0.9,0.9,0.9),zorder=0)
    plot_Ppe_vs_A.plot(A, P1, color='k')
    plot_Ppe_vs_A.set_xlabel("$A'$")
    plot_Ppe_vs_A.set_ylabel(r'$(1-\alpha)p$')

    # Plot PB' versus A'.
    # PB' = the axial magnetic pressure
    plot_PB_vs_A = versusA[1][1]
    plot_PB_vs_A.grid(color=(0.9,0.9,0.9),zorder=0)
    plot_PB_vs_A.plot(A, P2, color='k')
    plot_PB_vs_A.set_xlabel("$A'$")
    plot_PB_vs_A.set_ylabel(r'$(1-\alpha)^2B^2_z/2\mu_0$')
    
    ##########################################
    """ This function controls the selection 
    and plot of the boundary of A', i.e., Ab.
    When get_Ab = 0, users select Ab, 
    which will be presented and saved automatically.
    When get_Ab = +/- 1, load saved Ab from saved file. 
    Note this Ab value is with the prime symbol. """
    
    global Ab
    plotAb = True
    if 'plotAb' in kwargs:
        plotAb = kwargs['plotAb']
    if plotAb == True:
        if get_Ab == 0: 
            print("\n---------------------------------------------")
            print('\nClick to select Ab...')
            select_Ab = plt.ginput(1)
            Ab = select_Ab[0][0]
            print("Ab = ", Ab)
            # Plot Ab and save.
            ymin, ymax = plot_Bz_vs_A.get_ylim()
            saved_Ab = plot_Bz_vs_A.plot([Ab,Ab], [ymin,ymax], '--', color='c', linewidth=1.5)
            plot_Bz_vs_A.legend(saved_Ab,["$A' = A_b$"], loc='best', prop={'size':8})
            np.savetxt(rootDir + 'Ab.dat', [Ab], fmt="%.8f")
        if (get_Ab == 1) or (get_Ab == -1):
            # Load Ab data and plot it.
            Ab = np.loadtxt(rootDir + 'Ab.dat')
            ymin, ymax = plot_Bz_vs_A.get_ylim()
            saved_Ab = plot_Bz_vs_A.plot([Ab,Ab],[ymin,ymax], color='c',linewidth=1.5)
            plot_Bz_vs_A.legend(saved_Ab,["$A' = A_b$"], loc='best', prop={'size':8})

    return

def MVAB(B_DataFrame, **kwargs):
    """This function obtains MVAB frame with 
    the magnetic field data as input as well as provides hodograms. """

    Bhat = np.array(B_DataFrame)
    M = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            M[i,j] = np.mean(Bhat[:,i]*Bhat[:,j]) \
            - np.mean(Bhat[:,i])*np.mean(Bhat[:,j])
    D, X = scipy.linalg.eigh(M)

    limda, In = sorted(D), np.argsort(D)

    # MVAB frame
    # Note that the eigenvectors may be given with different signs
    # from Matlab results.
    X = np.array([X[:,In[2]], X[:,In[1]], X[:,In[0]]]).T
    limda_flip = np.flipud(limda)
    aBxi = [np.mean(B_DataFrame@X[:,0]),
            np.mean(B_DataFrame@X[:,1]),
            np.mean(B_DataFrame@X[:,2])]

    Bi = np.array(B_DataFrame@X)
    a = (Bi.mean(axis=0))**2
    C = [sum(a),
         -a[0]*(limda_flip[1]+limda_flip[2]) - a[1]*(limda_flip[0]+limda_flip[2]) - a[2]*(limda_flip[0]+limda_flip[1]),
         a[0]*limda_flip[1]*limda_flip[2] + a[1]*limda_flip[0]*limda_flip[2] + a[2]*limda_flip[0]*limda_flip[1]]
    Lim = np.sort(np.roots(C))
    Limmin = Lim[0]
    Gama = 1/np.sqrt(sum(a/((limda_flip-Limmin)**2)))
    ni = Gama*Bi.mean(axis=0)/(limda_flip-Limmin)
    normal = X@ni

    ######################################################################
    ############################# Hodogram ###############################
    ######################################################################
    # plotSwitch controls whether the Hodogram needs to be hidden.
    # plotHodogram_temp is a temporary controller that will work 
    # if this MVAB function will be called independently. 
    # If using the full GSR, it will read the setting in the main command line.
    plotSwitch = True
    if 'plotFigure' in kwargs: 
        plotSwitch = kwargs['plotFigure']
    plotHodogram_temp = False
    if 'plotHodogram' in kwargs: 
        plotHodogram_temp = kwargs['plotHodogram']
    else:
        plotHodogram_temp = plotHodogram

    if (plotHodogram_temp & plotSwitch):
        # On Hodograms: subscripts 1, 2, and 3 correspond to the maximum, 
        # intermediate, and minimum variance in the magnetic field.
        fig1, hodogram = plt.subplots(1, 2, figsize=(6, 4))
        plt.subplots_adjust(wspace = 0.3)

        # Plot B1 vs B2 - maximum vs intermediate
        plot_B1B2 = hodogram[0]
        plot_B1B2.plot(Bi[:,1], Bi[:,0], color='tab:blue', linewidth=1.0)
        B1B20, = plot_B1B2.plot(Bi[0,1], Bi[0,0], marker='d', markersize=5, color='tab:red')
        B1B21, = plot_B1B2.plot(Bi[-1,1], Bi[-1,0], marker='X', markersize=6, color='g')
        plot_B1B2.grid(color=(0.9,0.9,0.9),zorder=0)
        plot_B1B2.set_title(r'$\lambda_1/\lambda_2\approx$'
            +str(math.ceil(limda_flip[0]/limda_flip[1])))
        plot_B1B2.set_ylabel('B$_1$ (nT)')
        plot_B1B2.set_xlabel('B$_2$ (nT)')
        plot_B1B2.legend(handles=[B1B20,B1B21], 
            labels=['start','end'], loc='best',prop={'size':8})

        # Plot B1 vs B3 - maximum vs minimum
        plot_B1B3 = hodogram[1]
        plot_B1B3.plot(Bi[:,2], Bi[:,0], color='tab:blue',linewidth=1.0)
        B1B30, = plot_B1B3.plot(Bi[0,2], Bi[0,0], marker='d', markersize=5, color='tab:red')
        B1B31, = plot_B1B3.plot(Bi[-1,2], Bi[-1,0], marker='X', markersize=6, color='g')
        plot_B1B3.grid(color=(0.9,0.9,0.9), zorder=0)
        plot_B1B3.set_title(r'$\lambda_1/\lambda_3\approx$'
            +str(math.ceil(limda_flip[0]/limda_flip[2])))
        plot_B1B3.set_ylabel('B$_1$ (nT)')
        plot_B1B3.set_xlabel('B$_3$ (nT)')
        plot_B1B3.legend(handles=[B1B30,B1B31], 
            labels=['start','end'], loc='best', prop={'size':8})
        
    return X

def findVHT(GS_DataFrame, **kwargs):
    """This function corresponds to prefr.m from the original GS reconstruction,
    which includes the following:
    (1) calculate the constant vector VHT, 
    (2) time-varying VHT_varying_t,
    (3) check the quality of an HT frame with the corresponding cc values and figure checkHT,
    (4) plot the time-series data from the spacecraft dataset, and
    (5) plot the Walen relation between the remaining flow and Alfven velocities.

    *Reference: Hu, Q., & Sonnerup, B. U. Ö. 2002, JGR, 107, 1142 """

    # Physical constants
    k_Boltzmann, miu = 1.3806488e-23, 4.0*np.pi*1e-7 
    
    # Read other settings
    # Subscript "temp" means temporary settings,
    # in case these need to be adjusted.
    # Otherwise, it will read settings from the main command line.
    if 'spacecraftID' in kwargs:
        spacecraftID_temp = kwargs['spacecraftID']
    else:
        spacecraftID_temp = spacecraftID
    func = []
    if 'func' in kwargs:
        func = kwargs['func']
    # plotSwitch: Additional controller if one needs 
    # to call optcloud without showing figures
    plotSwitch = True 
    if 'plotFigure' in kwargs: plotSwitch = kwargs['plotFigure']
    checkHT_temp = False
    if 'checkHT' in kwargs: 
        checkHT_temp = kwargs['checkHT']
    else:
        checkHT_temp = checkHT

    # Read the magnetic field and plasma parameters
    B_DataFrame = GS_DataFrame.iloc[:,0:3].astype(float)
    Vsw_DataFrame = GS_DataFrame.iloc[:,3:6].astype(float)
    
    # fill nan for reconstruction
    Vsw_DataFrame.fillna(method='ffill', inplace=True)
    Vsw_DataFrame.fillna(method='bfill', inplace=True)
    
    if 'Np' in GS_DataFrame.keys():
        Np_DataFrame = GS_DataFrame.loc[:,['Np']].astype(float)
        Np_DataFrame.fillna(method='ffill', inplace=True)
        Np_DataFrame.fillna(method='bfill', inplace=True)
        Tp_DataFrame = GS_DataFrame.loc[:,['Tp']].astype(float)
        Tp_DataFrame.fillna(method='ffill', inplace=True)
        Tp_DataFrame.fillna(method='bfill', inplace=True)
    if 'Te' in GS_DataFrame.keys():
        Te_DataFrame = GS_DataFrame.loc[:,['Te']].astype(float)
    if 'Ne' in GS_DataFrame.keys():
        Ne_DataFrame = GS_DataFrame.loc[:,['Ne']].astype(float)

    # Calculate time step
    dt_temp = (GS_DataFrame.index[1]-GS_DataFrame.index[0]).seconds
    dt = np.append(0, np.ones(len(GS_DataFrame)-1)*dt_temp)
    ttime_temp = np.linspace(0, dt[1]*len(GS_DataFrame), len(GS_DataFrame)+1)
    ttime = ttime_temp[0:len(GS_DataFrame)]

    ######################################################################
    ####################### (1) Calculate VHT ############################
    ######################################################################
    N = len(B_DataFrame)
    B_square = np.square(B_DataFrame).sum(axis=1) 
    KN = np.zeros((N,3,3)) #((layer, row, column)).
    KN_temp = np.array([[B_DataFrame.iloc[:,1]**2+B_DataFrame.iloc[:,2]**2, -B_DataFrame.iloc[:,0]*B_DataFrame.iloc[:,1],
                         -B_DataFrame.iloc[:,0]*B_DataFrame.iloc[:,2]],
                        [-B_DataFrame.iloc[:,1]*B_DataFrame.iloc[:,0], B_DataFrame.iloc[:,0]**2+B_DataFrame.iloc[:,2]**2, 
                         -B_DataFrame.iloc[:,1]*B_DataFrame.iloc[:,2]],
                        [-B_DataFrame.iloc[:,2]*B_DataFrame.iloc[:,0], -B_DataFrame.iloc[:,2]*B_DataFrame.iloc[:,1], 
                         B_DataFrame.iloc[:,0]**2+B_DataFrame.iloc[:,1]**2]])
    KN = KN_temp.T
    K = np.mean(KN, axis=0)
    KVN = np.zeros((N,3))
    Vmat = np.array([Vsw_DataFrame.iloc[:,0], Vsw_DataFrame.iloc[:,1], Vsw_DataFrame.iloc[:,2]])
    KVN = np.array([np.matmul(KN[i], Vmat.T[i]) for i in range(N)])
    KV = np.mean(KVN, axis=0) # RHS1
    VHT = np.dot(np.linalg.inv(K), KV) # Constant vector

    ######################################################################
    ############ (2) Calculate the time-varying frame velocity ###########
    ######################################################################
    RHS2 = np.mean(np.array([KVN[i] * ttime[i] for i in range(len(ttime))]), axis=0)
    K1 = np.mean(np.array([KN[i] * ttime[i] for i in range(len(ttime))]), axis=0)
    K2 = np.mean(np.array([KN[i] * ttime[i]**2 for i in range(len(ttime))]), axis=0)
    VHT0aHT_K = np.row_stack((np.column_stack((K,K1)),np.column_stack((K1,K2))))
    VHT0aHT_RHS = np.append(KV, RHS2)
    VHT0aHT = la.inv(VHT0aHT_K) @ VHT0aHT_RHS
    VHT0, aHT = VHT0aHT[0:3], VHT0aHT[3:6]
    VHT_varying_t = np.array([VHT0 + aHT * ttime[i] for i in range(N)])

    ######################################################################
    ############### (3) Check the quality of an HT frame #################
    ######################################################################
    E = -np.cross(Vsw_DataFrame, B_DataFrame)
    EHT = -np.cross(VHT, B_DataFrame)
    EHT_varying_t = -np.cross(VHT_varying_t, B_DataFrame)

    E_EHT = np.row_stack((E[0,:], EHT[0,:]))
    E_EHT_varying_t = np.row_stack((E[0,:], EHT_varying_t[0,:]))
    for i in range(1,N):
        E_EHT = np.column_stack((E_EHT, np.row_stack((E[i,:], EHT[i,:]))))
        E_EHT_varying_t = np.column_stack((E_EHT_varying_t,np.row_stack((E[i,:], EHT_varying_t[i,:]))))
    S, aS = np.corrcoef(E_EHT), np.corrcoef(E_EHT_varying_t)
    CCoeff, aCCoef = S[0][1], aS[0][1]

    isVerbose = True 
    if 'isVerbose' in kwargs: 
        isVerbose = kwargs['isVerbose']
    if isVerbose:
        print("\nThe HT frame is good when the correlation coefficient is close to 1.")
        print("For the current time interval, it is {}.".format(CCoeff))
    
    # Warning of the poor HT frame
    if CCoeff <= 0.90:
        print("Warning! The HT frame might have poor quality.")
        print("Please set checkHT = True to see the corresponding figure.")
    
    if (checkHT_temp & plotSwitch):
        fig2, plotHT = plt.subplots(1, 1, figsize=(4.5, 4.5))

        # Find the axis limit
        plotHT_upper = math.ceil(max(abs(EHT/1000).max(), abs(EHT_varying_t/1000).max())/5.0)*5.0
        plotHT_lower = plotHT_upper * np.sign(min(EHT.min()/1000, EHT_varying_t.min()/1000))
        plotHT.set_xlim((plotHT_lower, plotHT_upper))
        plotHT.set_ylim((plotHT_lower, plotHT_upper)) 
        plotHT.grid(color=(0.9,0.9,0.9), zorder=0)
        # Plot
        plotHT.scatter(EHT/1000,E/1000, marker='o', c='none', 
            edgecolors='tab:blue', label=r'Const V$_{HT}$', zorder=5)
        plotHT.scatter(EHT_varying_t/1000,E/1000, marker='*', 
            c='none', edgecolors='tab:red', label=r'Accel. V$_{HT}$', zorder=5)
        plotHT.plot([plotHT_lower, plotHT_upper], [plotHT_lower, plotHT_upper], 
            '--', color='k', zorder=10)
        plotHT.legend(loc='best', prop={'size':9})
        plotHT.set_xlabel(r'-V$_{HT} \times B~(mV/m)$')
        plotHT.set_ylabel(r'-V$_{HT}(t) \times B~(mV/m)$')

        # Set title
        if (GS_DataFrame.index[0].strftime('%m/%d/%Y') == GS_DataFrame.index[-1].strftime('%m/%d/%Y')):
            pltTitleStart = GS_DataFrame.index[0].strftime('%m/%d/%Y %H:%M:%S')
            pltTitleEnd = GS_DataFrame.index[-1].strftime('%H:%M:%S')
        else:
            pltTitleStart = GS_DataFrame.index[0].strftime('%m/%d/%Y %H:%M:%S')
            pltTitleEnd = GS_DataFrame.index[-1].strftime('%m/%d/%Y %H:%M:%S')
        plotHT.set_title(r'$-V_{HT} \times B$ vs $-V_{HT}(t) \times B$',fontsize = 10)

    # If only needs VHT result
    if (func == 'simple_VHT'): return VHT

    # Get MVAB matrix here
    # If only needs the result of HT frame,
    # skip calculating MVAB frame.
    if (func == 'HT_frame'):
        pass
    else:
        X = MVAB(B_DataFrame, plotFigure=plotSwitch)

    ######################################################################
    ################### (4) Plot the original data #######################
    ######################################################################
    # Calculate some parameters for time-series plot & the Walen relation
    rho = np.array(Np_DataFrame['Np'])*1.673*1e-27*1e6
    # Alfven velocity
    VA = np.array([B_DataFrame.iloc[:,0]*1e-9/np.sqrt(miu*rho)/1000,
          B_DataFrame.iloc[:,1]*1e-9/np.sqrt(miu*rho)/1000,
          B_DataFrame.iloc[:,2]*1e-9/np.sqrt(miu*rho)/1000]).T
    # Alfven speed
    VA_norm = la.norm(VA, axis=1)
    # The average Alfven speed
    VA_mean = VA_norm.mean()
    # Vsw - VHT, the remaining flow velocity via the constant VHT vector
    V_remain_const = np.array(Vsw_DataFrame) - VHT
    V_remain_norm = la.norm(V_remain_const, axis=1) 
    # The average Alfven Mach number
    # Notice this is in the original frame
    MachAvg = np.nanmean(V_remain_norm/VA_norm)
    # Vsw - VHT_varying_t, the remaining flow velocity via the varying VHT vector
    V_remain_varying_t = np.array(Vsw_DataFrame) - VHT_varying_t
    V_remain_varying_norm = la.norm(V_remain_varying_t, axis=1) 

    # If needs the result of HT frame, 
    # findVHT returns VHT and average Alfven Mach number
    if (func == 'HT_frame'): 
        return VHT, MachAvg

    # Plot time-series of the original spacecraft parameters,
    # which include the magnetic field, number density & temperature,
    # and Alfven Mach number as well as V_remaining_varyting/VA.
    if (plotTimeSeries & plotSwitch):
        fig, spacecraft_data = plt.subplots(3, 1, figsize=(6, 6))
        plt.subplots_adjust(hspace = 0.5)

        # Plot the magnetic field
        plot_B = spacecraft_data[0]
        plot_B.grid(color=(0.9,0.9,0.9), zorder=0)
        B0, = plot_B.plot(B_DataFrame.index, B_DataFrame.iloc[:,0]/1e-9, color='r')
        B1, = plot_B.plot(B_DataFrame.index, B_DataFrame.iloc[:,1]/1e-9, color='g')
        B2, = plot_B.plot(B_DataFrame.index, B_DataFrame.iloc[:,2]/1e-9, color='b')
        # ACE & WIND use GSE coordinates
        if (spacecraftID_temp == 'ACE') or (spacecraftID_temp == 'WIND'):
            plot_B.legend(handles=[B0,B1,B2], labels=['$B_X$','$B_Y$','$B_Z$'], 
                loc='center left', prop={'size':8}, bbox_to_anchor=(1.01, 0.5))
        # ULYSSES, PSP, and SolO use RTN coordinates
        if (spacecraftID_temp == 'ULYSSES') or (spacecraftID_temp == 'PSP') or (spacecraftID_temp == 'SOLARORBITER'):
            plot_B.legend(handles=[B0,B1,B2], labels=['$B_R$','$B_T$','$B_N$'], 
                loc='center left', prop={'size':8}, bbox_to_anchor=(1.01, 0.5))
        plot_B.set_ylabel('B (nT)')
        plot_B.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Plot the number density and temperature
        plot_NpTp = spacecraft_data[1]
        plot_NpTp.grid(color=(0.9,0.9,0.9),zorder=0)
        plot_NpTp.plot(Np_DataFrame.index, Np_DataFrame['Np'], color='k',label = r'$N_p$')
        if 'Ne' in GS_DataFrame.keys():
            plot_NpTp.plot(Ne_DataFrame.index, Ne_DataFrame['Ne'], 
                color='k', linestyle='dashed', label = r'$N_e$')
        plot_NpTp.set_ylabel('Density (#/cc)')
        plot_NpTp.legend(loc = 'upper left',prop={'size':5})
        plot_NpTp_twin = plot_NpTp.twinx()
        plot_NpTp_twin.set_ylabel('Temperature (10$^6$K)')
        plot_NpTp_twin.tick_params(axis='y', colors='tab:blue')
        plot_NpTp_twin.yaxis.label.set_color('tab:blue')
        plot_NpTp_twin.plot(Tp_DataFrame.index, Tp_DataFrame['Tp']/1e6, 
            color='tab:blue',label = r'$T_p$')
        if 'Te' in GS_DataFrame.keys():
            plot_NpTp_twin.plot(Te_DataFrame.index, Te_DataFrame['Te']/1e6, 
                color='tab:blue', linestyle='dashed', label = r'$T_e$')
        plot_NpTp_twin.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plot_NpTp_twin.legend(loc = 'upper right',prop={'size':5})

        # Plot the Alfven Mach number, which is V_remaining/V_A
        plot_MA = spacecraft_data[2]
        plot_MA.grid(color=(0.9,0.9,0.9),zorder=0)
        plot_MA.plot(B_DataFrame.index, V_remain_norm/VA_norm, color='k', label=r'$M_A$')
        plot_MA.set_ylabel('$M_A$')
        plot_MA_twin = plot_MA.twinx()
        # Plot the ratio between the remaining flow and the Alfven speeds
        plot_MA_twin.plot(B_DataFrame.index, V_remain_varying_norm/VA_norm, 
            color='tab:blue', label=r'$V_{r}/{V_A}$')
        plot_MA_twin.set_ylabel('$V_{remaining}(t)/{V_A}$')
        plot_MA_twin.tick_params(axis='y', colors='tab:blue')
        plot_MA_twin.yaxis.label.set_color('tab:blue')
        plot_MA_twin.set_title(r'$\langle$M$_A\rangle$ = '
            +str(round(np.mean(V_remain_norm/VA_norm),3)), fontsize=10)
        plot_MA_twin.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.tight_layout()

    ######################################################################
    ######################### (5) Walen Relation #########################
    ######################################################################
    # NOTE! This is in the original frame
    VA_V_remain = np.row_stack((VA[0,:],V_remain_const[0,:]))
    VA_V_remain_varying= np.row_stack((VA[0,:],V_remain_varying_t[0,:]))
    for i in range(1,N):
        VA_V_remain = np.column_stack((VA_V_remain,np.row_stack((VA[i,:],V_remain_const[i,:]))))
        VA_V_remain_varying = np.column_stack((VA_V_remain_varying,np.row_stack((VA[i,:],V_remain_varying_t[i,:]))))
    
    # Polyfit can't handle nan
    index_nan = np.isfinite(VA_V_remain[0]) & np.isfinite(VA_V_remain[1])
    VA_V_remain_cc = np.polyfit(VA_V_remain[0][index_nan], VA_V_remain[1][index_nan], 1)
    index_nan1 = np.isfinite(VA_V_remain_varying[0]) & np.isfinite(VA_V_remain_varying[1])
    VA_V_remain_varying_cc = np.polyfit(VA_V_remain_varying[0][index_nan1], 
        VA_V_remain_varying[1][index_nan1], 1)
    
    # Plot the Walen Relation
    if (plotWalenRelation & plotSwitch):
        fig3, walen_plot = plt.subplots(1, 1, figsize=(4.5, 4))
        walen_plot.grid(color=(0.9,0.9,0.9), zorder=0)
        walen_plot.scatter(VA[:,0], V_remain_const[:,0], s=50, 
            marker='o', c='none', edgecolors='r', zorder=5)
        walen_plot.scatter(VA[:,1], V_remain_const[:,1], s=50, 
            marker='o', c='none', edgecolors='g', zorder=5)
        walen_plot.scatter(VA[:,2], V_remain_const[:,2], s=50 ,
            marker='o', c='none', edgecolors='b', zorder=5)
        walen_plot.set_xlabel('$V_{A}$ (km/s)')
        walen_plot.set_ylabel('$V_{sw}$ - $V_{HT}$ (km/s)')
        axes_upper = math.ceil(max(abs(VA).max(), abs(V_remain_const).max())/50.0)*50.0
        axes_lower = axes_upper * np.sign(min(VA.min(), V_remain_const.min()))
        walen_plot.set_xlim((axes_lower, axes_upper))
        walen_plot.set_ylim((axes_lower, axes_upper)) 
        walen_plot.plot([-axes_upper, axes_upper], 
            [np.polyval(VA_V_remain_cc, -axes_upper), 
            np.polyval(VA_V_remain_cc, axes_upper)], '--', color='k', zorder=10)
        walen_plot.set_title('Walen slope: '+str(round(VA_V_remain_cc[0], 3)))
        walen_plot.set_aspect('equal')
        
        if saveFig:
            fig3.savefig(rootDir + spacecraftID_temp + startTime.strftime('%Y%m%d%H%M%S') 
                + '_' + endTime.strftime('%Y%m%d%H%M%S') + '_walen_relation.png', format='png', dpi=450)
            
    return VHT, VA_mean, MachAvg

def clouddata(timeStart, timeEnd, FR_DataFrame, **kwargs):
    """The clouddata in the original GS reconstruction saves data 
    into a matrix, which is read and processed from the original
    spacecraft dataset within the selected interval.
    This version combines the clouddata function with several 
    features of hu12n.m to output the necessary parameters.
    It includes different outputs for different purposes."""
    
    # Physical constants
    k_Boltzmann, miu = 1.3806488e-23, 4.0*np.pi*1e-7 
    
    # Read other settings
    # Subscript "temp" means temporary settings,
    # in case these need to be adjusted.
    # Otherwise, it will read settings from the main command line.

    if 'func' in kwargs:
        func = kwargs['func']
    if 'spacecraftID' in kwargs:
        spacecraftID_temp = kwargs['spacecraftID']
    else:
        spacecraftID_temp = spacecraftID
    if func == 'HT_frame': 
        includeNe_temp, includeTe_temp = False, False
    elif func == 'mvab':
        pass
    else:
        includeNe_temp, includeTe_temp = includeNe, includeTe
    pressureSwitch_temp = 1
    if 'pressureSwitch' in kwargs: 
        pressureSwitch_temp = kwargs['pressureSwitch']
    elif (func == 'HT_frame') or (func == 'mvab'):
        pass
    else:
        pressureSwitch_temp = pressureSwitch
    polyOrder_temp = 3
    if 'polyOrder' in kwargs: 
        polyOrder_temp = kwargs['polyOrder']
    elif (func == 'HT_frame') or (func == 'mvab'):
        pass
    else:
        polyOrder_temp = polyOrder

    # Trim the preprocessed data
    selectedRange_mask = (FR_DataFrame.index >= timeStart) & (FR_DataFrame.index <= timeEnd)
    GS_DataFrame = FR_DataFrame.iloc[selectedRange_mask]

    # Calculate the time step
    dt_temp = (GS_DataFrame.index[1]-GS_DataFrame.index[0]).seconds
    dt = np.append(0,np.ones(len(GS_DataFrame)-1)*dt_temp)
    
    # Read the magnetic field and plasma data
    B_DataFrame = GS_DataFrame.iloc[:,0:3].astype(float)
    Vsw_DataFrame = GS_DataFrame.iloc[:,3:6].astype(float)
    B_Tesla = B_DataFrame*1e-9 
    B_Mag_T = np.array(np.sqrt(np.square(B_Tesla).sum(axis=1)))
    
    # If only need MVAB, 
    # return the processed magnetic field data.
    if func == 'mvab': 
        return B_DataFrame

    # Calculate pressures PBB & Pp & Pe
    if 'Np' in GS_DataFrame.keys():
        # fill nan for reconstruction
        Np = GS_DataFrame.loc[:,['Np']].astype(float)
        Np.fillna(method='ffill', inplace=True)
        Np.fillna(method='bfill', inplace=True)
        Tp = GS_DataFrame.loc[:,['Tp']].astype(float)
        Tp.fillna(method='ffill', inplace=True)
        Tp.fillna(method='bfill', inplace=True)
        Pp = np.array(Np['Np']) * 1e6 * k_Boltzmann * np.array(Tp['Tp'])
    else:
        print('\nError! No Np or Tp data.')
        print('Cannot calculate the HT frame.')
        print('Please re-select the time interval.')
        exit()
    # If both Ne and Te are available and users wish do include both,
    # Pe = Ne kB Te
    if ('Ne' in GS_DataFrame.keys()) & includeNe_temp & ('Ne' in GS_DataFrame.keys()) & includeNe_temp:
        Pe = np.array(GS_DataFrame['Ne']) * 1e6 * k_Boltzmann * np.array(GS_DataFrame['Te'])
    # If decides not to include Ne data or Ne is unavailable
    # Pe = Np kB Te
    elif ('Te' in GS_DataFrame.keys()) & includeTe_temp & ((not includeNe_temp) or ('Ne' not in GS_DataFrame.keys())):
        Pe = np.array(GS_DataFrame['Np']) * 1e6 * k_Boltzmann * np.array(GS_DataFrame['Te'])
    # If both are unavailable or not included, Pe = 0
    else:
        Pe = 0

    # For magnetic flux calculation, only needs Np
    # to estimate the mass.
    if func == 'flux_calculation': 
        return Np
    # If called when obtainting flux rope axis,
    # needs to return VHT related results.
    if (func == 'optcloud_simple') or (func == 'optcloud_full'):
        VHT, VA_mean, MachAvg = findVHT(GS_DataFrame, plotFigure=False, isVerbose=False)
    # If only need the HT frame analysis,
    # returns VHT and average Alfven Mach number with personalized settings.
    elif func == 'HT_frame':
        VHT, MachAvg = findVHT(GS_DataFrame, spacecraftID=spacecraftID_temp, 
            checkHT=False,func='HT_frame')
    # The default output without settings.
    else:
        VHT, VA_mean, MachAvg = findVHT(GS_DataFrame)

    AlphaMach = MachAvg**2
    # The total magnetic pressure
    PBB = B_Mag_T**2/(2*miu)
    # Thermal pressure
    Ppe = (Pe + Pp)*(1 - AlphaMach)
    # Plasma beta
    Beta = (Pe + Pp)/PBB

    # If need in an earlier step when obtainting flux rope axis
    if func == 'optcloud_simple':
        return B_Tesla, Ppe, PBB, dt, VHT, AlphaMach
    
    # Obtain the FR frame and prepare for calculating Pt'
    # If use axis from optcloud or obtainAxis.py
    Z_reverse = False
    if 'fluxropeAxis' in kwargs:
        # Translated from hu12n.m
        Z_UnitVector = kwargs['fluxropeAxis']
        Vza = (VHT@Z_UnitVector)*Z_UnitVector
        VHTsv = VHT-Vza
        VHTs = np.array(np.sqrt(np.square(VHTsv).sum()))
        X_UnitVector = -VHTsv/VHTs
        Y_UnitVector = np.cross(Z_UnitVector,X_UnitVector)
        Bx_inFR,By_inFR,Bz_inFR = B_Tesla@X_UnitVector,B_Tesla@Y_UnitVector,B_Tesla@Z_UnitVector
        B_inFR = B_Tesla @ np.array([X_UnitVector,Y_UnitVector,Z_UnitVector]).T
        # HT frame with a given z-axis
        matrix_transToFR = np.array([X_UnitVector, Y_UnitVector, Z_UnitVector]).T
    # If use axis from GS detection results
    elif 'FR_list' in kwargs: 
        SFR_detection_list = kwargs['FR_list']
        if 'eventNum' in kwargs:
            eventNum = kwargs['eventNum']
            theta, phi = SFR_detection_list['theta_deg'].iloc[eventNum], SFR_detection_list['phi_deg'].iloc[eventNum]
            # HT frame with a given z-axis
            matrix_transToFR = detect.angle2matrix(theta, phi, VHT)
            B_inFR = B_Tesla.dot(matrix_transToFR)
    
    # If only need HT frame
    if func == 'HT_frame': 
        return matrix_transToFR
    
    # Calculate parameters in the FR frame
    # The axial magnetic pressure'
    PB = (1-AlphaMach)**2 * np.array(B_inFR[2])**2 / (2.0*miu)
    # The transverse pressure'
    Pt = PB + pressureSwitch_temp*Ppe + pressureSwitch_temp*AlphaMach*(1-AlphaMach)*PBB
    # Calculate the remaining flow velocity in FR frame
    V_remaining = (Vsw_DataFrame-VHT)*1e3
    V_rmn_FR = V_remaining.dot(matrix_transToFR)
    # Calculate A'
    VHT_inFR = VHT.dot(matrix_transToFR)
    ds = - VHT_inFR[0] * 1000.0 * (GS_DataFrame.index[1]-GS_DataFrame.index[0]).seconds
    A1 = integrate.cumtrapz(-(1 - AlphaMach) * B_inFR[1], dx=ds, initial=0)
    ds1 = - VHT_inFR[0] * 1000.0 * dt
    Xa = np.cumsum(ds1)
    Pt_A1_polyCoef = np.polyfit(A1, Pt, polyOrder_temp)
    Pt_A1_fit = np.polyval(Pt_A1_polyCoef, A1)

    # After getting axis from optcloud or obtainAxis,
    # need to check on two sets of figures.
    if func == 'optcloud_full':
        plotXSeries(Xa, A1, B_inFR, PBB, Ppe, Pt, Beta)
        plotPressureA(A1, Pt, B_inFR, Ppe, PB, Pt_A1_fit, plotAb=False)
        return
    # The same set of figure needed in reconstruction
    if func == 'hu34n':
        plotPressureA(A1, Pt, B_inFR, pressureSwitch_temp*Ppe, PB, Pt_A1_fit)
        return Pt, VA_mean, V_rmn_FR, B_inFR, A1, Xa, AlphaMach, Ab

def var_interp2(x1, y1, x2, y2, F):
    # Interpolation 2D matrix/array
    
    func_interp = interpolate.interp2d(x1, y1, F, kind='cubic')
    X = func_interp(x2, y2)
    return X

def var_resample(A, B, Pt, V):
    # Normalized by unit & resample to nx points
    
    # Default value of nx = 15
    global A0, B_magni_max, PB_max, L0
    A0 = max(abs(A))
    B_magni_max = np.array(np.sqrt(np.square(B).sum(axis=1))).max()
    L0 = A0/B_magni_max
    PB_max = B_magni_max**2/miu
    
    An = A/A0
    Bxn = B[0]/B_magni_max 
    Byn = B[1]/B_magni_max
    Bzn = B[2]/B_magni_max
    Ptn = Pt/PB_max
    Vx, Vy = V[0], V[1]
    
    Bxn1 = signal.resample_poly(Bxn, nx, len(Bxn))
    Byn1 = signal.resample_poly(Byn, nx, len(Byn))
    Bzn1 = signal.resample_poly(Bzn, nx, len(Bzn))
    Ptn1 = signal.resample_poly(Ptn, nx, len(Ptn))
    An1 = signal.resample_poly(An, nx, len(An))
    Vx1 = signal.resample_poly(Vx, nx, len(Vx))
    Vy1 = signal.resample_poly(Vy, nx, len(Vy))

    return Bxn1, Byn1, Ptn1, An1, Vx1, Vy1

def fitPtA(A, Pt, polyOrder, **kwargs):
    # Get fitting parameters for Pt'(A')
    # A is obtained via cumsum & normalized & resample
    # Pt is actually Pt_prime after normalized & resample
    
    # Fit P't versus A'
    # PtA_co: coefficient
    PtA_co, PtA_res, PtA_rank, PtA_sin_val, PtA_rcond = np.polyfit(A, Pt, polyOrder, full=True)
    res = np.sqrt(PtA_res/len(A))/(max(Pt)-min(Pt))     # residue
    # Calculated from fitting curve
    Ptf = np.polyval(PtA_co,A) # Pt values from fitting
    dPtdA = np.polyder(PtA_co)

    # left extrapolation part Pt versus A
    Al = min(A) + dAl0 # dAl = left percentage
    PtL = np.polyval(PtA_co, Al)
    dPtL = np.polyval(dPtdA, Al)
    if PtL <= 0:
        print("\nError! Bz' or Pt' at the left boundary is non-positive.")
        print("Please adjust dAl0 and try again.")
        exit()
    co1 = dPtL/PtL
    co2 = math.log(PtL) - co1*Al
    Alp = np.arange(min(A)-0.4,Al+0.1,0.1) # All Al points
    exptail = np.exp(co1 * Alp + co2)

    # right extrapolation part Pt versus A
    Ar = max(A) - dAr0 # dAr = right percentage
    PtR = np.polyval(PtA_co, Ar)
    dPtR = np.polyval(dPtdA, Ar)
    co1r = dPtR/PtR
    if PtR <= 0:
        print("\nError! Bz' or Pt' at the right boundary is non-positive.")
        print("Please adjust dAr0 and try again.")
        exit()
    co2r = math.log(PtR) - co1r*Ar
    Arp = np.arange(Ar,max(A)+0.5,0.1) # All Ar points
    rexptail = np.exp(co1r*Arp + co2r)


    # Save file for Bz(A) results 
    # Re-run will need to call these parameters
    if pressureSwitch == 0:
        global Pt_A_fit_coeffz, Al0z, co1z, co2z, Ar0z, co1rz, co2rz
        Pt_A_fit_coeffz, Al0z, co1z, co2z, Ar0z, co1rz, co2rz = PtA_co, Al, co1, co2, Ar, co1r, co2r
        if polyOrder == 3:
            np.savetxt(rootDir + 'bz2fit.dat', [PtA_co[0], PtA_co[1], PtA_co[2], PtA_co[3], 
                Al, co1, co2, Ar, co1r, co2r], fmt="%.8f")
        elif polyOrder == 2:
            np.savetxt(rootDir + 'bz2fit.dat', [PtA_co[0], PtA_co[1], PtA_co[2], 
                Al, co1, co2, Ar, co1r, co2r], fmt="%.8f")

    # If would like to check whether dAl or dAr are selected appropriately.
    if checkPtAFitting:
        fig5, plotFit = plt.subplots(1, 1, figsize=(4.5,4))
        PtAF0, = plotFit.plot(A, Pt, color='tab:red',marker='o')
        PtAF1, = plotFit.plot(A, Ptf,color='g')
        PtAF2, = plotFit.plot(Alp, exptail, color='gold')
        PtAF3, = plotFit.plot(Arp, rexptail, color='tab:blue');
        leftPer, rightPer = 'dAl0 = '+str(dAl0), 'dAr0 = '+str(dAr0)
        plotFit.legend(handles=[PtAF0,PtAF1,PtAF2,PtAF3],
            labels=[r"$normalized~P_{t}'(A')$","$normalized~P_{t}'(A')~fitting$",leftPer,rightPer], 
            loc='best', prop={'size':8})
        plotFit.set_xlabel("$A'/A_0$")
        plotFit.set_ylabel("$P_t'/p_0$")
        fig5.tight_layout()

    return PtA_co, dPtdA, Al, Ar, co1, co2, co1r, co2r, res

def Ap_calculation(A, Bx, By, Ptp):
    """This function calculates the vector potential A'.
    With A', both Bx' and By' will also be calculated."""

    # Initialize grid parameters
    global x, y, hx, hy, Pt_A_fit_coeff, residue
    global dPtdA, Al0, Ar0, co1, co2, co1r, co2r

    hx = Xi1[1]-Xi1[0] # increment of 15 points along y = 0
    py = 1.0*0.1/1
    hy = py*hx
    mid = round(ny/2) - dmid
    x = Xi1
    y = np.array([(j-mid)*hy for j in range(1,ny+1)])
    
    # Get fitting parameters from function fitPtA
    Pt_A_fit_coeff, dPtdA, Al0, Ar0, co1, co2, co1r, co2r, residue = fitPtA(A, Ptp, polyOrder)

    # Initialize 1D & 2D arrays
    # Ap_contour: A' contour on the cross-section map
    # dAdy has default size: 131 x 15
    Ap_contour, dAdy, dAdx = np.zeros((ny, nx)), np.zeros((ny, nx)), np.zeros((ny-1, nx))
    dPt, rhs = np.zeros((ny-1, nx)), np.zeros((ny-1, nx))
    Ap_contour_temp, dAdy_temp = np.zeros((ny, nx)), np.zeros((ny, nx))
    dA2dy2 = np.zeros(nx)
    
    # The mid y = 0 plane has the original resampled A values
    # Now dAdy & dAdx only have half plane - 56 x 15
    mid = round(ny/2) - dmid
    Ap_contour[mid-1,:] = A
    dAdy[mid-1,:] = Bx
    dAdx[mid-1,:] = -By
    
    #######################################################################
    ######################### upper half plane ############################
    #######################################################################
    nl, nr = 1, nx # nr = 15
    for j in range(mid-1, ny-1): # j = 56:130 upper plane
        for i in range(nl-1, nr):
            # handle boundaries of A first
            # dPt = dPt/dA
            if Ap_contour[j,i] < Al0: 
                # for portion that is smaller than left percentage
                # extrapolation; dPt = fitting curve
                dPt[j,i] = co1*np.exp(co1*Ap_contour[j,i] + co2) 
            elif Ap_contour[j,i] > Ar0: 
                # for portion that is larger than right percentage
                # extrapolation;
                dPt[j,i] = co1r*np.exp(co1r*Ap_contour[j,i] + co2r) 
            else:
                # fitting curve of Pt(A)
                dPt[j,i] = np.polyval(dPtdA,Ap_contour[j,i])
            rhs[j,i] = -dPt[j,i] # rhs = - dPtdA, note this is after normalized

        kk = 1.00 
        yfactor = min(0.7,(y[j]-y[mid-1])/(y[ny-1]-y[mid-1]))
        k1, k2, k3 = kk*yfactor, 3.0-2.0*kk*yfactor, kk*yfactor
        for i in range(nl-1, nr):
            Ap_contour_temp[j,i] = Ap_contour[j,i]
            dAdy_temp[j,i] = dAdy[j,i]
            if ((i > nl-1) & (i < nr-1)): # i in (1, 15)
                Ap_contour_temp[j,i] = (k1*Ap_contour[j,i+1] 
                    + k2*Ap_contour[j,i]
                    + k3*Ap_contour[j,i-1])/3;
                dAdy_temp[j,i] = (k1*dAdy[j,i+1] + k2*dAdy[j,i] + k3*dAdy[j,i-1])/3;
        Ap_contour[j,:] = Ap_contour_temp[j,:].copy()
        dAdy[j,:] = dAdy_temp[j,:]

        for i in range(nl-1, nr):
            if (i  ==  nl-1): # bottom boundary
                dA2dx2 = (1.0/(hx**2))*(-5.0*Ap_contour[j,i+1] + 
                                        2.0*Ap_contour[j,i] + 4.0*Ap_contour[j,i+2] - Ap_contour[j,i+3]);
            if (i  ==  nr-1): # upper boundary
                dA2dx2 = (1.0/(hx**2))*(-5.0*Ap_contour[j,i-1] + 
                                        2.0*Ap_contour[j,i] + 4.0*Ap_contour[j,i-2] - Ap_contour[j,i-3]);
            if ((i > nl-1) & (i < nr-1)): # in between
                dA2dx2 = (1.0/(hx**2))*(Ap_contour[j,i+1]- 2.0*Ap_contour[j,i] + Ap_contour[j,i-1]);
            # dA2/dy2 = -dA2/dx2 - miu dPtdA
            # rhs = - dPtdA
            dA2dy2 = rhs[j,i] - dA2dx2
            # dpt - dA2/dx2
            # A(x, y+dy) = A(x,y) + dAdy*(dy) + 0.5*(dA2/dy2)*(dy)^2
            Ap_contour[j+1,i] = Ap_contour[j,i] + dAdy[j,i]*(y[j+1] - y[j])\
                + 0.5*dA2dy2*((y[j+1] - y[j])**2)
            # dAdy = Bx
            # Bx(x, y+dy) = B(x,y) + dA2/dy2 * dy
            dAdy[j+1,i] = dAdy[j,i] + dA2dy2*(y[j+1] - y[j]) # Taylor expansion
    
        if j is not (mid-1):
            for i in range(nl-1, nr):
                if (i == nl-1):
                    dAdx_temp = (1.0/hx)*(Ap_contour[j,i+1] - Ap_contour[j,i])
                elif  (i == nr-1):
                    dAdx_temp = (1.0/hx)*(Ap_contour[j,i] - Ap_contour[j,i-1])
                else:
                    dAdx_temp = (1.0/(2.0*hx))*(Ap_contour[j,i+1] - Ap_contour[j,i-1])
                dAdx[j,i] = dAdx_temp

    # ny = 131, handle boundary
    for i in range(nl-1, nr): # 1:15
        Ap_contour_temp[ny-1,i] = Ap_contour[ny-1,i]
        dAdy_temp[ny-1,i] = dAdy[ny-1,i]
        if ((i > nl-1) & (i < nr-1)):
            Ap_contour_temp[ny-1,i] = (k1*Ap_contour[ny-1,i+1] + 
              k2*Ap_contour[ny-1,i] + k3*Ap_contour[ny-1,i-1])/3
            dAdy_temp[ny-1,i] = (k1*dAdy[ny-1,i+1] + k2*dAdy[ny-1,i] + k3*dAdy[ny-1,i-1])/3
    Ap_contour[ny-1,:] = Ap_contour_temp[ny-1,:]
    dAdy[ny-1,:] = dAdy_temp[ny-1,:].copy()

    #######################################################################
    ######################### lower half plane ############################
    #######################################################################f
    nl, nr = 1, nx # nr = 15
    for j in range(mid-1,0,-1):
        for i in range(nl-1, nr):
            if Ap_contour[j,i] < Al0:
                dPt[j,i] = co1*np.exp(co1*Ap_contour[j,i] + co2)
            elif Ap_contour[j,i]> Ar0:
                dPt[j,i] = co1r*np.exp(co1r*Ap_contour[j,i] + co2r)
            else:
                dPt[j,i] = np.polyval(dPtdA,Ap_contour[j,i])
            rhs[j,i] = -dPt[j,i]

        kk = 1.00
        yfactor = min(0.7,(y[j]-y[mid-1])/(y[0]-y[mid-1]))
        k1, k2, k3 = kk*yfactor, 3.0-2.0*kk*yfactor, kk*yfactor
        for i in range(nl-1, nr): # 1:15
            Ap_contour_temp[j,i] = Ap_contour[j,i]
            dAdy_temp[j,i] = dAdy[j,i]
            if ((i > nl-1) & (i < nr-1)): # i within (1, 15)
                Ap_contour_temp[j,i] = (k1*Ap_contour[j,i+1] 
                                       + k2*Ap_contour[j,i]
                                       + k3*Ap_contour[j,i-1])/3
                dAdy_temp[j,i] = (k1*dAdy[j,i+1] + k2*dAdy[j,i] + k3*dAdy[j,i-1])/3
        Ap_contour[j,:] = Ap_contour_temp[j,:]
        dAdy[j,:] = dAdy_temp[j,:]

        for i in range(nl-1, nr):
            if (i  ==  nl-1): # bottom boundary
                dA2dx2 = (1.0/(hx**2))*(-5.0*Ap_contour[j,i+1] + 2.0*Ap_contour[j,i] + 4.0*Ap_contour[j,i+2]-Ap_contour[j,i+3])
            if (i  ==  nr-1): # upper boundary
                dA2dx2 = (1.0/(hx**2))*(-5.0*Ap_contour[j,i-1] + 2.0*Ap_contour[j,i] + 4.0*Ap_contour[j,i-2]-Ap_contour[j,i-3])
            if ((i > nl-1) & (i < nr-1)): # in between
                dA2dx2 = (1.0/(hx**2))*(Ap_contour[j,i+1] - 2.0*Ap_contour[j,i] + Ap_contour[j,i-1])
            # dA2/dy2 = -dA2/dx2 - miu dPtdA
            dA2dy2 = rhs[j,i] - dA2dx2
            # A(x, y-dy) = A(x,y) + dAdy*(dy) + 0.5*(dA2/dy2)*(dy)^2
            Ap_contour[j-1,i] = Ap_contour[j,i] + dAdy[j,i]*(y[j-1] - y[j])\
                + 0.5*dA2dy2*((y[j-1] - y[j])**2)
            # Bx(x, y-dy) = B(x,y) + dA2/dy2 * dy
            dAdy[j-1,i] = dAdy[j,i] + dA2dy2*(y[j-1] - y[j]) #Taylor expansion
        if j is not (mid-1):
            for i in range(nl-1, nr):
                if (i == nl-1):
                    dAdx_temp = (1.0/hx)*(Ap_contour[j,i+1] - Ap_contour[j,i])
                elif  (i == nr-1):
                    dAdx_temp = (1.0/hx)*(Ap_contour[j,i] - Ap_contour[j,i-1])
                else:
                    dAdx_temp = (1.0/(2.0*hx))*(Ap_contour[j,i+1] - Ap_contour[j,i-1])
                dAdx[j,i] = dAdx_temp

    # ny = 0, handle boundary
    for i in range(nl-1, nr): # 1:15
        Ap_contour_temp[0,i] = Ap_contour[0,i]
        dAdy_temp[0,i] = dAdy[0,i]
        if ((i > nl-1) & (i < nr-1)):
            Ap_contour_temp[0,i] = (k1*Ap_contour[0,i+1] + k2*Ap_contour[0,i] + k3*Ap_contour[0,i-1])/3
            dAdy_temp[0,i] = (k1*dAdy[0,i+1] + k2*dAdy[0,i] + k3*dAdy[0,i-1])/3
    Ap_contour[0,:] = Ap_contour_temp[0,:]
    dAdy[0,:] = dAdy_temp[0,:]

    dPt_temp = dPt.copy()
    dPt_temp[0] = dPt_temp[1]
    dPt = np.row_stack((dPt_temp,dPt_temp[-1]))

    # Calculate Bx' and By' from A' matrix
    # Note they are now back to physical meanings in nT
    Bx_from_A_contour = dAdy * B_magni_max/1e-9/(1-AlphaMach)
    By_from_A_contour_temp = -dAdx*B_magni_max/1e-9/(1-AlphaMach)
    By_from_A_contour_temp[0] = By_from_A_contour_temp[1] 
    By_from_A_contour = np.row_stack((By_from_A_contour_temp,By_from_A_contour_temp[-1]))

    return Ap_contour, dPt, Bx_from_A_contour, By_from_A_contour

def Bz_calculation(Ap_contour, B, Ab):
    # This function calculates Bz'

    # pressureSwitch = 1, this is for the second run
    if pressureSwitch == 1:
        bz2fit_parameter = np.loadtxt(rootDir + 'bz2fit.dat')
        global Pt_A_fit_coeffz,Al0z,co1z,co2z,Ar0z, co1rz, co2rz
        if polyOrder == 3:
            Pt_A_fit_coeffz = [bz2fit_parameter[0],bz2fit_parameter[1], bz2fit_parameter[2], bz2fit_parameter[3]]
            Al0z, co1z, co2z = bz2fit_parameter[4], bz2fit_parameter[5], bz2fit_parameter[6]
            Ar0z, co1rz, co2rz = bz2fit_parameter[7], bz2fit_parameter[8], bz2fit_parameter[9]
        elif polyOrder == 2:
            Pt_A_fit_coeffz = [bz2fit_parameter[0],bz2fit_parameter[1], bz2fit_parameter[2]]
            Al0z, co1z, co2z = bz2fit_parameter[3], bz2fit_parameter[4], bz2fit_parameter[5]
            Ar0z, co1rz, co2rz = bz2fit_parameter[6], bz2fit_parameter[7], bz2fit_parameter[8]
    
    Bz2 = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            if Ap_contour[j,i] < Al0z: # less than left boundary
                Bz2[j,i] = np.exp(co1z*Ap_contour[j,i] + co2z)
            elif Ap_contour[j,i] > Ar0z: # greater than right boundary
                Bz2[j,i] = np.exp(co1rz*Ap_contour[j,i] + co2rz)
            else: # in between, from fitting curve
                Bz2[j,i] = np.polyval(Pt_A_fit_coeffz, Ap_contour[j,i])

    Bz_from_A = np.sqrt(Bz2*2) # Bz2 is Fz2
    Bz_from_A_max = Bz_from_A.max()
    Bz_from_A_max_index = np.argwhere(Bz_from_A == Bz_from_A_max)

    global J0, I0
    if np.size(Bz_from_A_max_index) == 0:
        print("\nError! A' and Bz have too many nan.")
        print("Please adjust parameters to try again.")
        exit()

    J0, I0 = Bz_from_A_max_index[0][0], Bz_from_A_max_index[0][1]
        
    if Ab/A0 < Al0z: Bz2b_from_Ab_pri = np.exp(co1z*Ab/A0+co2z)
    elif Ab/A0 > Ar0z: Bz2b_from_Ab_pri = np.exp(co1rz*Ab/A0+co2rz)
    else: Bz2b_from_Ab_pri = np.polyval(Pt_A_fit_coeffz, Ab/A0)
    Bzb_from_Ab_pri = np.sqrt(Bz2b_from_Ab_pri*2)*B_magni_max/1e-9
    
    Bz_inFR = np.array(B[2])
    Bz_inFR_max = abs(Bz_inFR).max()
    index_Bz_inFR_max = np.argmax(Bz_inFR)

    # Bz_from_A_phy is back to its physical value in nT
    Bz_from_A = (Bz_inFR[index_Bz_inFR_max]/Bz_inFR_max)*Bz_from_A*B_magni_max/1e-9
    Bz_from_A_phy = Bz_from_A/(1-AlphaMach)
   
    return Bz_from_A_phy

def Jz_calculation(dPt):
    # This function calculates Jz'
    # Jz_from_fit_physical is back to its physical value

    J_norm_unit = B_magni_max/(miu*L0)
    Jz_from_fit = dPt*J_norm_unit # dPt = dPt/dA
    Jz_from_fit_physical = Jz_from_fit/(1-AlphaMach)
    
    return Jz_from_fit_physical

def flux_calculation(Xi1, Ap, Bz, Jz, **kwargs):
    """This function estimates the poloidal and toroidal magnetic fluxes,
    as well as the current and total mass.
    The poloidal flux is obtained by multiplying Am and 1 AU length,
    where Am is the extreme value of poloidal flux per unit length.
    The toroidal flux is obtained by summing the product of Bz and area.

    # Input
    A_pri_contour = A'
    Bz & Jz are from fitting & without prime symbol
    """

    # Prepare calculation
    # Flux calculation needs finer grid, default sizes 131 x 131
    # Here, calculate all parameters in a finer grid first.
    mn = ny # to have a matrix [ny X ny]
    mid = round(ny/2) - dmid
    yi_finer = np.array([(j-mid)*hy for j in range(1,ny+1)])
    xi_finer = np.linspace(Xi1[0], Xi1[nx-1], num=ny)
    hx_finer = xi_finer[1]-xi_finer[0]
    hy_finer = yi_finer[1]-yi_finer[0]
    Ap_finer = var_interp2(x,y,xi_finer,yi_finer,Ap)
    Bz_finer = var_interp2(Xi1,yi_finer,xi_finer,yi_finer,Bz)
    Jz_finer = var_interp2(Xi1,yi_finer,xi_finer,yi_finer,Jz)
    Ap_finer_phy = Ap_finer*A0/(1-AlphaMach)
    xe_finer = xi_finer*L0/AU
    ye_finer = yi_finer*L0/AU
    Bx_A_finer = np.zeros((Ap_finer.shape))
    By_A_finer = np.zeros((Ap_finer.shape))
    for j in range(mn):
        for i in range(mn):
            if i == 0:
                By_A_finer[j,i] = -(1./hx_finer)*(Ap_finer[j,i+1] - Ap_finer[j,i])*B_magni_max/1e-9;
            elif i == mn-1: 
                By_A_finer[j,i] = -(1./hx_finer)*(Ap_finer[j,i] - Ap_finer[j,i-1])*B_magni_max/1e-9;
            else:
                By_A_finer[j,i] = -(1./(2.*hx_finer))*(Ap_finer[j,i+1] - Ap_finer[j,i-1])*B_magni_max/1e-9;
            if j == 0:
                Bx_A_finer[j,i] = (1./hy_finer)*(Ap_finer[j+1,i] - Ap_finer[j,i])*B_magni_max/1e-9;
            elif j == mn-1:
                Bx_A_finer[j,i] = (1./hy_finer)*(Ap_finer[j,i] - Ap_finer[j-1,i])*B_magni_max/1e-9;
            else:
                Bx_A_finer[j,i] = (1./(2.*hy_finer))*(Ap_finer[j+1,i] - Ap_finer[j-1,i])*B_magni_max/1e-9;

    # Get physical values
    Bx_A_finer_phy = Bx_A_finer/(1-AlphaMach)
    By_A_finer_phy = By_A_finer/(1-AlphaMach)
    Ab_physical = Ab/(1-AlphaMach)
    Ax_physical, Am = Ax_Am_calc(Ap_finer_phy)

    # Prepare flux calculation
    Pht_A, Php_A = np.ones(len(Ax_physical)), np.zeros(len(Ax_physical)) 
    Iz_A, Bz_A = np.ones(len(Ax_physical)), np.zeros(len(Ax_physical)) 
    Jz_A, Kr_A = np.ones(len(Ax_physical)), np.zeros(len(Ax_physical)) 
    Tau_A = np.zeros((4, len(Ax_physical)))
    garea = hx_finer*L0*hy_finer*L0

    for k in range(0,len(Ax_physical)):
        flux = 0.0
        current = 0.0
        
        if Ax_physical[k]/A0 < Al0z/(1-AlphaMach):
            Bz2b = np.exp(co1z*Ax_physical[k]/A0+co2z)
        elif Ax_physical[k]/A0 > Ar0z/(1-AlphaMach):
            Bz2b = np.exp(co1rz*Ax_physical[k]/A0+co2rz)
        else:
            Bz2b = np.polyval(Pt_A_fit_coeffz, Ax_physical[k]/A0)
        Bz_A[k] = np.sqrt(Bz2b*2)*B_magni_max/1e-9 # back to nT
        
        if Ax_physical[k]/A0<Al0/(1-AlphaMach):
            Jz_A[k] = co1*np.exp(co1*Ax_physical[k]/A0+co2)
        elif Ax_physical[k]/A0>Ar0/(1-AlphaMach):
            Jz_A[k] = co1r*np.exp(co1r*Ax_physical[k]/A0+co2r)
        else:
            Jz_A[k] = np.polyval(dPtdA,Ax_physical[k]/A0); # fit p(A)
        
        for j in range(len(ye_finer)):
            for i in range(len(xe_finer)):
                if get_Ab == 1:
                    if Ap_finer_phy[j,i] > Ax_physical[k]:
                        flux = flux+Bz_finer[j,i]*1e-9*garea
                        current = current+Jz_finer[j,i]*garea
                elif get_Ab == -1:
                    if Ap_finer_phy[j,i] < Ax_physical[k]:
                        flux = flux+Bz_finer[j,i]*1e-9*garea
                        current = current+Jz_finer[j,i]*garea
                else:
                    flux = flux+Bz_finer[j,i]*1e-9*garea
                    current = current+Jz_finer[j,i]*garea
        Pht_A[k] = flux
        Iz_A[k] = current

    toroidal_flux = flux
    Jz_A = Jz_A*Jz[J0][I0]
    ix = np.argmin(abs(xe_finer-X0))
    iy = np.argmin(abs(ye_finer-Y0))
    Php_A = abs(Am-Ax_physical)*AU
    phi_pA = abs((Ab_physical-Ap_finer_phy[iy,ix])*1e9/AU)
    phi_p_1AU = phi_pA*1e-9*AU*AU
    poloidal_flux = phi_p_1AU
    phi_p_2AU = phi_pA*1e-9*AU*AU*2 # 2AU length!! for historical reasons
    phi_p1AUophi_t = phi_p_2AU/2/flux
    # rough estimate of mass below
    m_proton = 1.67262192e-24 # grams
    Np = clouddata(startTime,endTime,alldata,func='flux_calculation')
    Mass_grams = (xe_finer[-1]-xe_finer[0])*(ye_finer[-1]-ye_finer[0])*AU**3*np.mean(Np['Np'])*1e6*m_proton
    
    return toroidal_flux, poloidal_flux, current

def Ap_homo(Bz, xe, ye, mn, fine):
    """Translated from Ap_homo.m in the original GS reconstruction
    The integral of 2D vector is calculated with slight differences
    from that obtained by quadv function in Matlab."""

    ny, nx = np.size(Bz,0), np.size(Bz,1)
    Ax, Ay = np.zeros((ny, nx)), np.zeros((ny, nx))
    Bz_max_index = np.argwhere(Bz == Bz.max())
    j0, i0 = Bz_max_index[0][0], Bz_max_index[0][1]
    x0, y0 = (xe[i0]+xe[i0+1])/2, ye[j0]
    xe_temp, ye_temp = xe-x0, ye-y0
    xi = np.linspace(xe_temp.min(),xe_temp.max(),mn)
    yi = np.linspace(ye_temp.min(),ye_temp.max(),mn)
    xe_new, ye_new = np.meshgrid(xe_temp,ye_temp);
    xi_new, yi_new = np.meshgrid(xi,yi)
    
    Bzz2 = interpolate.griddata((xe_new.ravel(), ye_new.ravel()), 
        Bz.ravel(), (xi_new, yi_new), method='cubic')
    f = lambda x: x*interpolate.RectBivariateSpline(xi, yi, Bzz2.T)(x*xe_temp, x*ye_temp).T
    F, err = integrate.quad_vec(f, 0, 1)
    Ax, Ay =-ye_new * F, xe_new * F

    return Ax, Ay

def curl_calculation(x, y, Fx, Fy):
    # Calcuate curl of a vector
    dx, dy = x[0,:], y[:,0]
    dFxdy = np.gradient(Fx, dy, axis=0)
    dFydx = np.gradient(Fy, dx, axis=1)
    curlz = dFydx - dFxdy
    
    return curlz

def helicity_calculation(A, Bx, By, Bz, **kwargs):
    """This function is migrated from Kr_homo.m, which 
    is one of functions included in the GS reconstruction package.
    Calculation of the gauge-invariant relarive helicity follows
    Hu and Dasgupta [2005].

    # Input
    A = A_contour_finer_physical - T dot m
    Bz = Bz_from_fit_physical - nT
    Bx = Bx_from_A - nT
    By = By_from_A - nT
    """
    
    garea = hx*L0*hy*L0/AU/AU
    Ax, Ay = Ap_homo(Bz, xe, ye, 256, 1) # in nT AU since Bz ~ nT, xe ~ AU
    X1, Y1 = np.meshgrid(xe*1000,ye*1000)
    curlz_temp = curl_calculation(X1, Y1, Ax, Ay)
    curlz = curlz_temp*1000
    relative_helicity_temp, relative_helicity_l_temp, A0B0 = 0.0, 0.0, 1e9/AU
    area, difff, count = 0.0, 0.0, 0
    Ab_physical = Ab/(1-AlphaMach)

    # Calculate the relative helicity
    # Hm = integral 2A' dot Bt dV (eq.3 in Hu and Dasgupta 2005)
    # relative_helicity_temp = Hm/2 = integral A' dot Bt dV
    for j in range(len(ye)-2):
        for i in range(len(xe)):
            if get_Ab == 1:
                if A[j,i] > Ab_physical:
                    relative_helicity_temp = relative_helicity_temp+(Ax[j,i]*Bx[j,i]+Ay[j,i]*By[j,i])*garea
                    relative_helicity_l_temp = relative_helicity_l_temp+(A[j,i]-Ab_physical)*Bz[j,i]*garea
                    area = area+garea
                    difff = difff+(Bz[j,i]-curlz[j,i])**2 
                    count = count+1
            elif get_Ab == -1:
                if A[j,i] < Ab_physical:
                    relative_helicity_temp = relative_helicity_temp+(Ax[j,i]*Bx[j,i]+Ay[j,i]*By[j,i])*garea
                    relative_helicity_l_temp = relative_helicity_l_temp+(A[j,i]-Ab_physical)*Bz[j,i]*garea
                    area = area+garea
                    difff = difff+(Bz[j,i]-curlz[j,i])**2
                    count = count+1    
            else:
                relative_helicity_temp = relative_helicity_temp+(Ax[j,i]*Bx[j,i]+Ay[j,i]*By[j,i])*garea
                relative_helicity_l_temp = relative_helicity_l_temp+(A[j,i]-Ab_physical)*Bz[j,i]*garea
                area = area+garea
                difff = difff+(Bz[j,i]-curlz[j,i])**2
                count = count+1
    
    # The relative helicity
    # nT^2 AU AU^2 --- Ax in nT AU, Bx in nT, and garea in AU^2
    # The relative helicity per unit length
    relative_helicity = 2*relative_helicity_temp
    # T m nT AU^2 = 1e-9 m T^2 AU^2 = T^2 AU^3 (1e-9/AU)
    # The second way to estimate
    relative_helicity_l = 2*relative_helicity_l_temp*A0B0
    # The relative helicity per volume
    relative_helicity_perVolume = relative_helicity/area
    # sqrt(Chi^2) evulates the deviation b/t the original Bz from GSR
    # and that recovered by two different apporaches in least-squares sense
    chi = np.sqrt(difff/count)
    
    return relative_helicity

def Ax_Am_calc(A):
    # Linspace A values based on different chiralities 
    # A = A_contour_finer_physical
    Ab_physical = Ab/(1-AlphaMach)
    if get_Ab == 1: 
        Am = A.max()
        Ax_physical = np.linspace(Am, Ab_physical, nx)
    elif get_Ab == -1: 
        Am = A.min()
        Ax_physical = np.linspace(Am, Ab_physical, nx)
    else: 
        Am = abs(A).max()
        Ax_physical = 0

    return Ax_physical, Am

def plotReconstruction(Ap, Bz, VA_mean, Bx, By, Ap_nr, Ptp_nr, Vx_rmn, Vy_rmn, **kwargs):
    """This function includes three figures of the GSR results:
    (1) cross-sectional map, (2) Pt' versus A', and (3) map of jz.
    Typically, the first two figures are required outputs while the
    third plot is optional.
    *Corresponds to part of hu34n.m in the original GS reconstruction
    
    # Input:
    A: A_contour
    Bz: Bz_from_fit_physical
    Bx & By: components in FR frame after normalization & resample
    Ap_nr & Ptp_nr: A' & Pt' after normalization & resample
    Vx_rmn & Vy_rmn: V_remaining velocity's X & Y components in FR frame
    """

    # Title of figure
    fig2, crossSection = plt.subplots(1, 1, figsize=(4.5, 4))
    if (startTime.strftime('%m/%d/%Y') == endTime.strftime('%m/%d/%Y')):
        pltTitleStart = startTime.strftime('%m/%d/%Y %H:%M:%S')
        pltTitleEnd = endTime.strftime('%H:%M:%S')
    else:
        pltTitleStart = startTime.strftime('%m/%d/%Y %H:%M:%S')
        pltTitleEnd = endTime.strftime('%m/%d/%Y %H:%M:%S')
    pltTitleStr = pltTitleStart + ' - ' + pltTitleEnd + ' UT'
    crossSection.set_title(pltTitleStr)
    crossSection.set_xlabel('$x~(10^{-3}$AU)',fontsize=9)
    crossSection.set_ylabel('$y~(10^{-3}$AU)',fontsize=9)

    ######################################################################
    ##################### (1) Cross-sectional map ########################
    ######################################################################
    levels = np.linspace(Bz.min(), Bz.max(), 50)
    crossSection_Bz = plt.contourf(xe*1000,ye*1000,Bz,50,levels=levels,cmap='RdYlBu_r')
    colorBar_ticks = np.arange(round(Bz.min()), round(Bz.max()))
    colorbarp = crossSection.get_position()
    colorbar_position = fig2.add_axes([colorbarp.x0+colorbarp.x1-0.135, 
                                       colorbarp.y0, 0.03, colorbarp.height])
    colorBar = plt.colorbar(crossSection_Bz,cax=colorbar_position,
        ticks=colorBar_ticks,spacing='uniform',fraction=0.043, pad=0.03)
    tick_locator = ticker.MaxNLocator(nbins=8)
    colorBar.locator = tick_locator
    colorBar.update_ticks()
    crossSection.text(1.02,-0.075, '$B_z~(nT)$', fontsize=9, transform=crossSection.transAxes)
    
    # Plot A contours
    # Notice the colormap scales from light to dark corresponding to - to + values
    # or vice versa. To have a clear schematic, using different colors for 
    # A values inside Ab with +/- signs, no guaranteed effects.
    num_Ap_lt0 = (Ap[abs(Ap)<abs(Ab)]<0).sum()
    num_Ap_gt0 = (Ap[abs(Ap)<abs(Ab)]>0).sum()
    if (num_Ap_lt0 > num_Ap_gt0):
        crossSection.contour(xe*1000,ye*1000,Ap,50,cmap='Greys_r',linewidths=0.8, zorder=1)
    else:
        crossSection.contour(xe*1000,ye*1000,Ap,50,cmap='Greys',linewidths=0.8, zorder=1)
    # Plot Ab contour
    crossSection.contour(xe*1000,ye*1000,Ap,[Ab],cmap='Greys',linewidths=2.0)
    # Plot Bz_max
    crossSection.plot(X0*1000, Y0*1000, marker = '.',color='w');

    # Plot remaining flow velocity & Bt vectors
    arrow_scale = (round(VA_mean/50)+1)*100
    crossSection.quiver(np.append(1000*xe,xe[-4]*1000), 
               np.append(np.zeros(len(xe)),ye[9]*1000),
               np.append(Bx,1e-8/B_magni_max),
               np.append(By,0),
               color='w',minshaft=1.5,headwidth=3,scale=10)
    crossSection.quiver(np.append(1000*xe,xe[1]*1000), 
               np.append(np.zeros(len(xe)),ye[115]*1000), 
               np.append(Vx_rmn/1000,VA_mean), 
               np.append(Vy_rmn/1000,0),
               color='yellowgreen',minshaft=1.5,scale=arrow_scale,
               scale_units='inches',headwidth=3);
    crossSection.quiver(xe[1]*1000,ye[115]*1000, 
               VA_mean,0,color='tab:green',minshaft=1.5,
               width=0.01,headwidth=3,zorder=3,
               scale=arrow_scale,scale_units='inches');

    str_VA = r'$V_A$ = ' + str(round(VA_mean)) + ' km/s'
    crossSection.text((xe[-4])*1000, ye[13]*1000, '10 nT',fontsize=12, color='white')
    crossSection.text((xe[1])*1000, ye[119]*1000, str_VA,fontsize=10, color='tab:green') 
    crossSection.set_aspect('equal')

    # Save figure
    if saveFig:
        fig2.savefig(rootDir + spacecraftID + startTime.strftime('%Y%m%d%H%M%S') 
            + '_' + endTime.strftime('%Y%m%d%H%M%S') + '.png', 
                format='png', dpi=450)

    ######################################################################
    ###################### (2) Pt' versus A' #############################
    ######################################################################
    Amin, Amax = Ap.min()/A0, Ap.max()/A0
    Al, Ar = np.linspace(Amin, Al0, 20), np.linspace(Ar0, Amax, 20)
    Aall = np.linspace(Al0, Ar0, 100)
    PtA = np.polyval(Pt_A_fit_coeff, Aall)
    Ptl, Ptr = np.exp(co1*Al + co2), np.exp(co1r*Ar + co2r)
    dPtA = np.polyval(dPtdA, Aall)
    dPtl, dPtr = co1*np.exp(co1*Al+co2), co1r*np.exp(co1r*Ar+co2r)
    index_Ap_nr = np.argmax(abs(Ap_nr))

    # Figure setting
    fig4, plot_Pt_versus_A = plt.subplots(1, 1, figsize=(4.1, 4))
    plot_Pt_versus_A.set_title(pltTitleStr)
    plot_Pt_versus_A.set_xlabel(r"$A'~(T \cdot m)$",fontsize=10)
    plot_Pt_versus_A.set_ylabel("$P'~(nPa)$",fontsize=10)
    # Plot the first branch of Pt versus A
    PtvA1, = plot_Pt_versus_A.plot(Ap_nr[0:index_Ap_nr+1]*A0,
            PB_max*np.array(Ptp_nr[0:index_Ap_nr+1])/1e-9,
            marker = 'o',linestyle='--',color='tab:blue',linewidth=2)
    # Plot the second branch of Pt versus A
    PtvA2, = plot_Pt_versus_A.plot(Ap_nr[index_Ap_nr+1:len(Ap_nr)]*A0,
            PB_max*np.array(Ptp_nr[index_Ap_nr+1:len(Ap_nr)])/1e-9,
            marker = '*',markersize=8,linestyle='--',color='tab:red',linewidth=2)
    # Plot all three fitting curves
    plot_Pt_versus_A.plot(Al*A0, Ptl*PB_max/1e-9, color='k',linewidth=2)
    plot_Pt_versus_A.plot(Ar*A0, Ptr*PB_max/1e-9, color='k',linewidth=2)
    fittingCurve, = plt.plot(Aall*A0, PtA*PB_max/1e-9, color='k',linewidth=2)

    # Plot Ab
    plot_Pt_versus_A.axvline(x=Ab, color='k',linewidth=1.0)
    plot_Pt_versus_A.grid(linewidth = 0.5,color=(0.9,0.9,0.9),zorder=0)
    plot_Pt_versus_A.legend(handles=[PtvA1,PtvA2,fittingCurve],
        labels=["$Pt'(A')_{1st}$","$Pt'(A')_{2nd}$",
        '$R_f$ = '+str(format(residue[0],'.2f'))],
            loc='best',prop={'size':10})
    fig4.tight_layout()
    
    if saveFig:
        fig4.savefig(rootDir + spacecraftID + startTime.strftime('%Y%m%d%H%M%S') 
            + '_' + endTime.strftime('%Y%m%d%H%M%S') + '_PtA.png', 
                format='png', dpi=450)

    ######################################################################
    ########################### (3) Jz map ###############################
    ######################################################################
    if plotJz:
        Jz = kwargs['Jz']
        fig3, plot_Jz = plt.subplots(1, 1, figsize=(4.5, 4))
        # Figure setting
        plot_Jz.set_title(pltTitleStr)
        plot_Jz.set_xlabel('$x~(10^{-3}$AU)',fontsize=9)
        plot_Jz.set_ylabel('$y~(10^{-3}$AU)',fontsize=9)

        # Plot Jz map
        levels = np.linspace(Jz.min(), Jz.max(), 50)
        plot_Jz_contour = plt.contourf(xe*1000,ye*1000,Jz,50,levels=levels,cmap='RdYlBu_r')
        colorbarp = plot_Jz.get_position()
        colorbar_position = fig3.add_axes([colorbarp.x0+colorbarp.x1-0.135,
            colorbarp.y0, 0.03, colorbarp.height])
        colorBar_ticks = np.arange(round(Jz.min()), round(Jz.max()))
        colorBar = plt.colorbar(plot_Jz_contour,cax=colorbar_position,
            ticks=colorBar_ticks,spacing='uniform',fraction=0.043, pad=0.03)
        tick_locator = ticker.MaxNLocator(nbins=8)
        colorBar.locator = tick_locator
        colorBar.update_ticks()
        colorBar.ax.tick_params(labelsize=6)
        plot_Jz.text(1.01,-0.075, '$j_z~(A/m{^2})$', fontsize=8, transform=plot_Jz.transAxes)
        
        # Plot A contours
        if (num_Ap_lt0 > num_Ap_gt0):
            plot_Jz.contour(xe*1000,ye*1000,Ap,50,cmap='Greys_r',linewidths=0.8, zorder=1)
        else:
            plot_Jz.contour(xe*1000,ye*1000,Ap,50,cmap='Greys',linewidths=0.8, zorder=1)
        # Plot Ab contour
        plot_Jz.contour(xe*1000,ye*1000,Ap,[Ab],cmap='Greys',linewidths=2.0, zorder=1)
        # Plot Bz_max
        plot_Jz.plot(X0*1000, Y0*1000, marker = '.',color='w');


        # Plot remaining flow velocity & Bt vectors
        plot_Jz.quiver(np.append(1000*xe,xe[-4]*1000), 
                   np.append(np.zeros(len(xe)),ye[9]*1000),
                   np.append(Bx,1e-8/B_magni_max),
                   np.append(By,0),
                   color='w',minshaft=1.5,headwidth=3,scale=10)
        plot_Jz.quiver(np.append(1000*xe,xe[1]*1000), 
                   np.append(np.zeros(len(xe)),ye[115]*1000), 
                   np.append(Vx_rmn/1000,VA_mean), 
                   np.append(Vy_rmn/1000,0),
                   color='yellowgreen',minshaft=1.5,headwidth=3,
                   scale=arrow_scale,scale_units='inches')
        plot_Jz.quiver(xe[1]*1000,ye[115]*1000, 
               VA_mean,0,color='tab:green',minshaft=1.5,
               width=0.01,headwidth=3,zorder=3,
               scale=arrow_scale,scale_units='inches')

        str_VA = '$V_A$ = ' + str(round(VA_mean)) + ' km/s'
        plot_Jz.text((xe[-4])*1000, ye[13]*1000, '10 nT',fontsize=12, color='white')
        plot_Jz.text((xe[1])*1000, ye[119]*1000, str_VA,fontsize=10, color='tab:green') 
        plot_Jz.set_aspect('equal')
        # Save figure
        if saveFig:
            fig3.savefig(rootDir + spacecraftID + startTime.strftime('%Y%m%d%H%M%S') 
                + '_' + endTime.strftime('%Y%m%d%H%M%S') + '_Jz.png', 
                    format='png', dpi=450)
    return

def selectInterval(startIntv, endIntv, alldata, longer_data_interval):
    """Sometimes, one may would like to adjust boundaries.
    This function calls plot function in GS detection,
    uses ginput to let users select interval, and print intervals.
    In time-series figure, usually present +/- 10*resolution 
    around the specified interval [timeStart, timeEnd] if data available."""

    resolution = (alldata.index[1]-alldata.index[0]).seconds
    if pressureSwitch == 0:
        if longer_data_interval:
            start_new = startIntv-timedelta(seconds=resolution*10)
            end_new =  endIntv+timedelta(seconds=resolution*10)
        else:
            start_new, end_new = startIntv, endIntv
        new_start_temp, new_end_temp = detect.plot_time_series_data(alldata,spacecraftID,
            start=start_new,end=end_new,func='GSR',adjustInterval=True)
        np.savetxt(rootDir + 'saved_boundary.txt', [new_start_temp,new_end_temp], fmt="%.32f")
    elif pressureSwitch == 1:
        if os.path.isfile(rootDir + 'saved_boundary.txt'):
            new_start_temp, new_end_temp = np.loadtxt(rootDir + 'saved_boundary.txt')
        else:
            print('\nError! No saved FR boundary.')
            print('If using the original boundary, set adjustInterval=False.')
            print('Otherwise, please check file path or set pressureSwitch = 0 to select the FR interval. ')
            exit()
    
    # Convert formart from matplotlib to datetime
    new_start_temp = matplotlib.dates.num2date(new_start_temp, tz=None).strftime('%Y-%m-%d %H:%M:%S')
    new_end_temp = matplotlib.dates.num2date(new_end_temp, tz=None).strftime('%Y-%m-%d %H:%M:%S')
    new_start = datetime.strptime(new_start_temp, '%Y-%m-%d %H:%M:%S')
    new_end = datetime.strptime(new_end_temp, '%Y-%m-%d %H:%M:%S')

    return new_start, new_end

def download_process_data(startIntv, endIntv, **kwargs):
    """If the GS detection has been run before the reconstruction,
    there are existed preprocessed files.
    They may equal to or longer than the selected time interval.
    If no such files, download and process a new one.
    Return to processed data file and indicator of whether it 
    is a longer interval.
    """

    # Settings that may be used by independent functions
    func = []
    if 'func' in kwargs:
        func = kwargs['func']
    if 'file_dir' in kwargs:
        rootDir_temp = kwargs['file_dir']
    else:
        rootDir_temp = rootDir
    if 'spacecraftID' in kwargs:
        spacecraftID_temp = kwargs['spacecraftID']
    else:
        spacecraftID_temp = spacecraftID

    # Make directory for downloaded and processed data files
    data_cache_dir = rootDir_temp + spacecraftID_temp + '_' + 'data_reconstruction'
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)
    data_pickle_dir_default = rootDir_temp + spacecraftID_temp + \
    '_' + 'pickle_reconstruction'
    if not os.path.exists(data_pickle_dir_default):
        os.makedirs(data_pickle_dir_default)
    
    # Case 1: data files existed in default data folder
    existedPickles = os.listdir(data_pickle_dir_default)
    existedFile = False
    longer_data_interval = False
    for i in range(len(existedPickles)):
        if str(spacecraftID_temp) in str(existedPickles[i]):
            existStartAll = datetime.strptime(existedPickles[i][len(spacecraftID_temp)+1:len(spacecraftID_temp)+15], '%Y%m%d%H%M%S')
            existEndAll = datetime.strptime(existedPickles[i][len(spacecraftID_temp)+16:len(spacecraftID_temp)+30], '%Y%m%d%H%M%S')
            if (existStartAll<=startIntv) & (existEndAll>=endIntv):
                existedFile = True
                existStart = existStartAll
                existEnd = existEndAll
    
    # Case 2: data files existed from detection results
    data_pickle_dir_detection = rootDir_temp + 'detection_results/'
    if not os.path.exists(data_pickle_dir_detection):
        os.makedirs(data_pickle_dir_detection)
    existedPickles2 = os.listdir(data_pickle_dir_detection)
    existedFile2 = False
    for i in range(len(existedPickles2)):
        if str(spacecraftID_temp) in str(existedPickles2[i]):
            existStartAll2 = datetime.strptime(existedPickles2[i][len(spacecraftID_temp)+1:len(spacecraftID_temp)+15], '%Y%m%d%H%M%S')
            existEndAll2 = datetime.strptime(existedPickles2[i][len(spacecraftID_temp)+16:len(spacecraftID_temp)+30], '%Y%m%d%H%M%S')
            if (existStartAll2<=startIntv) & (existEndAll2>=endIntv):
                existedFile2 = True
                existStart2 = existStartAll2
                existEnd2 = existEndAll2

    # Case 3: Exactly the same pickle file exists that may processed from the last run
    pickle_path_name = data_pickle_dir_default + '/' + spacecraftID_temp +'_' + \
        (startIntv-timedelta(minutes=10)).strftime('%Y%m%d%H%M%S') + '_'+ \
        (endIntv+timedelta(minutes=10)).strftime('%Y%m%d%H%M%S') + '_preprocessed.p'


    # Check Case 3:
    if (os.path.isfile(pickle_path_name)):
        print('\nPreprocessed file exists, loading... ')
        alldata = pd.read_pickle(pickle_path_name)
    # Check Case 2:
    elif existedFile2:
        # If an existed file has a time interval >= selected one, loading data...
        print('\nPreprocessed file exists from detection, loading... ')
        pickle_path_name2 = data_pickle_dir_detection + spacecraftID_temp \
        +'_' + existStart2.strftime('%Y%m%d%H%M%S') + '_'+ \
        existEnd2.strftime('%Y%m%d%H%M%S') +'/data_pickle/'\
        + spacecraftID_temp +'_' + existStart2.strftime('%Y%m%d%H%M%S') + '_'+ \
        existEnd2.strftime('%Y%m%d%H%M%S') + '_preprocessed.p'
        if ((spacecraftID == 'SOLARORBITER') or (spacecraftID == 'PSP')):
            print("Notice that the detection has sometimes been done for two resolutions.")
            print("Please take a look whether the correct processed data is loaded.")
            print("The current load file is {}.".format(pickle_path_name2))
            print("\nIf not the correct one, please try to switch names temporarily.")
        alldata_detection = pd.read_pickle(pickle_path_name2)
        if func == 'simple_VHT':
            selectedRange_mask = (alldata_detection.index >= startIntv) & (alldata_detection.index <= endIntv)
            alldata = alldata_detection.iloc[selectedRange_mask]
        else:
            alldata = alldata_detection
        longer_data_interval = True
    # Check Case 1:
    elif existedFile:
        # If an existed file has a time interval >= selected one, loading data...
        print('\nPreprocessed file exists (longer interval), loading... ')
        pickle_path_name1 = data_pickle_dir_default + '/' + spacecraftID_temp +'_' + \
        existStart.strftime('%Y%m%d%H%M%S') + '_'+ \
        existEnd.strftime('%Y%m%d%H%M%S') + '_preprocessed.p'
        alldata_long = pd.read_pickle(pickle_path_name1)
        # Trim data
        if func == 'simple_VHT':
            selectedRange_mask = (alldata_long.index >= startIntv) & (alldata_long.index <= endIntv)
            alldata = alldata_long.iloc[selectedRange_mask]
        else:
            alldata = alldata_long
        longer_data_interval = True
    else:
        # No existed data file, download & process a new one.
        data_dict = detect.download_data(spacecraftID_temp, data_cache_dir, 
            startIntv-timedelta(minutes=10), endIntv+timedelta(minutes=10),isVerbose=True)
        print("\n---------------------------------------------")
        alldata = detect.preprocess_data(data_dict, data_pickle_dir_default, 
            isVerboseR=True, isPlotFilterProcess=False)

    return alldata, longer_data_interval

def reconstruction(yourPath, **kwargs):
    """Here is the main function that controls 
    all processes during the GS reconstruction."""

    ######################################################################
    ################################ Default #############################
    ######################################################################
    # Physical constants
    global AU, k_Boltzmann ,miu
    AU, k_Boltzmann, miu = 1.49e11, 1.3806488e-23, 4.0*np.pi*1e-7 
    # Default parameters for the GS reconstruction
    global uni_grid, polyOrder, nx, ny, dmid, get_Ab, dAl0, dAr0, pressureSwitch
    uni_grid, polyOrder, nx, ny, dmid, get_Ab, dAl0, dAr0, pressureSwitch = 1,3,15,131,0,0,0,0,0
    # Personalized settings
    if 'polyOrder' in kwargs: polyOrder = kwargs['polyOrder']
    if 'nx' in kwargs: nx = kwargs['nx']
    if 'ny' in kwargs: ny = kwargs['ny']
    if 'dmid' in kwargs: dmid = kwargs['dmid']
    if 'get_Ab' in kwargs: 
        get_Ab = kwargs['get_Ab']
    else:
        print("Error! Please include the controller 'get_Ab' to select map boundary.")
        exit()
    if 'dAl0' in kwargs: dAl0 = kwargs['dAl0']
    if 'dAr0' in kwargs: dAr0 = kwargs['dAr0']
    if 'pressureSwitch' in kwargs: 
        pressureSwitch = kwargs['pressureSwitch']
    else:
        print("Error! Please include the controller 'pressureSwitch' to implement reconstruction.")
        exit()

    ######################################################################
    ########################### Add-on settings ##########################
    ######################################################################
    # Read add-on settings
    global checkPtAFitting, plotJz, saveFig, plotWalenRelation 
    global plotTimeSeries, plotHodogram, checkHT
    global includeTe, includeNe
    
    includeTe = False
    if 'includeTe' in kwargs: includeTe = kwargs['includeTe']
    includeNe = False
    if 'includeNe' in kwargs: includeNe = kwargs['includeNe']
    checkPtAFitting = False
    if 'checkPtAFitting' in kwargs:checkPtAFitting = kwargs['checkPtAFitting']
    plotJz = False
    if 'plotJz' in kwargs:plotJz =kwargs['plotJz']
    plotWalenRelation = False
    if 'plotWalenRelation' in kwargs: plotWalenRelation = kwargs['plotWalenRelation']
    plotTimeSeries = False
    if 'plotSpacecraftTimeSeries' in kwargs: plotTimeSeries = kwargs['plotSpacecraftTimeSeries']
    plotHodogram = False
    if 'plotHodogram' in kwargs: plotHodogram = kwargs['plotHodogram']
    saveFig = False
    if 'saveFig' in kwargs:saveFig = kwargs['saveFig']
    checkHT = False
    if 'checkHT' in kwargs:checkHT = kwargs['checkHT']
    detection2Reconstruction = False
    if 'detection2Reconstruction' in kwargs: 
        detection2Reconstruction = kwargs['detection2Reconstruction']
    helicityTwist = False
    if 'helicityTwist' in kwargs: helicityTwist = kwargs['helicityTwist']
    ######################################################################
    #####################  Download & process data #######################
    ######################################################################
    global alldata, spacecraftID, rootDir
    spacecraftID = kwargs['spacecraftID']
    rootDir = yourPath

    if ((spacecraftID == 'ACE') or (spacecraftID == 'ULYSSES') or 
        (spacecraftID == 'SOLARORBITER')) & (includeNe or includeTe):
        print("Error! ACE, ULYSSES, OR SOLARORBITER does not have electron data.")
        print("Please set includeTe = False and includeNe = False.")
        exit()
    # Read start and end times of interval
    # If from detection result...
    if ('FR_list' in kwargs) & ('eventNo' in kwargs):
        SFR_detection_list = kwargs['FR_list']
        eventNo = kwargs['eventNo']
        startIntv, endIntv = SFR_detection_list['startTime'][eventNo], SFR_detection_list['endTime'][eventNo]
    # If specified in the main command line...
    elif ('timeStart' in kwargs) & ('timeEnd' in kwargs):
        startIntv, endIntv = kwargs['timeStart'], kwargs['timeEnd']
    # No specified times, print error info.
    else:
        print('Error! Please indicate starting and ending times.')
        print('- If reconstruct an event from DETECTION result, \
            \n  please indicate FR_list and eventNo and timestamps \
            \n  will be read from detection result.')
        print('- Otherwise please specify timeStart and timeEnd.')
        exit()
    # Till this step, there must be a pair of starting & ending times in
    # reconstruction function, either from FR_list or specified times.
    # Otherwise, this function will exit.

    # Download & process data
    alldata, longer_data_interval = download_process_data(startIntv,endIntv)
    
    ######################################################################
    #########################  Interval setting ##########################
    ######################################################################
    # If would like to adjust the current interval...
    global startTime, endTime
    adjustInterval=False
    if 'adjustInterval' in kwargs: adjustInterval = kwargs['adjustInterval']
    if adjustInterval:
        startTime, endTime = selectInterval(startIntv,endIntv,alldata,longer_data_interval)
    else:
        startTime, endTime = startIntv, endIntv

    ######################################################################
    #######################  Prepare reconstruction ######################
    ######################################################################    
    # Obtain flux rope axis first.
    global AlphaMach, Xi1, Ab, xe, ye, X0, Y0   
    
    # Error information
    # Since the times of interval have been given in previous line,
    # now one just needs to determine from which method the flux rope axis is obtained.
    # As long as adjustAxis == True, the axis will be force to obtain from reconstruction.
    # If would like to get it from detection results, must have the following
    # ('FR_list' in kwargs) & ('eventNo' in kwargs) & (adjustAxis == False)

    adjustAxis = False
    if 'adjustAxis' in kwargs: 
        adjustAxis = kwargs['adjustAxis']
    # If no source of axis is specified, print error info
    if (('FR_list' not in kwargs) or ('eventNo' not in kwargs)) & (adjustAxis == False):
        print("\nError! Please indicate a source of flux rope z-axis:")
        print("- If from DETECTION: please indicate the source of \
            \n   FR_list & specify eventNo, and set adjustAxis = False.")
        print("- If from RECONSRUCTION: please set adjustAxis = True.")
        exit()
    # If get from detection result...
    elif ('FR_list' in kwargs) & ('eventNo' in kwargs) & (adjustAxis == False): 
        print("\n##############################################")
        print('\nStart the reconstruction...')
        SFR_detection_list = kwargs['FR_list']
        eventNo = kwargs['eventNo']
        # Calculate nesscessary parameters from clouddata
        Pt, VA_mean, V_rmn_FR, \
        B_inFR, A1, Xa, AlphaMach, Ab = clouddata(startTime, endTime, alldata,
            func='hu34n', FR_list=SFR_detection_list, eventNum=eventNo)
    # If get from GS reconstruction
    else:
        # Find the axis from optcloud or obtainAxis
        print("\n---------------------------------------------")
        print("\nGetting flux rope axis...")
        if ((pressureSwitch == 0) & (get_Ab == 0)):
            z_optcloud = obtainAxis.findZaxis(rootDir,startTime,endTime,
                alldata,polyOrder,adjustInterval=adjustInterval,
                det2rec=detection2Reconstruction)
        elif ((pressureSwitch == 0) & (get_Ab != 0)) or ((pressureSwitch == 1) & (get_Ab != 0)):
            # Only find the axis when pressureSwitch = 0.
            # When it is 1, load saved zs file.
            if os.path.isfile(rootDir + 'zs_select.txt'):
                print("\nLoading axis file zs_select.txt from",rootDir)
                z_optcloud = np.loadtxt(rootDir + 'zs_select.txt')
                print("\n##############################################")
                print('\nStart the reconstruction...')
            else:
                print("\nLoading axis file zs_select.txt from",rootDir)
                print("Error! No file was found.")
                print("Please set pressureSwitch = 0 & get_Ab = 0 to obtain the axis.")
                exit()
        # Calculate nesscessary parameters from clouddata
        Pt, VA_mean, V_rmn_FR, \
        B_inFR, A1, Xa, AlphaMach, Ab = clouddata(startTime,endTime,
            alldata,func='hu34n',fluxropeAxis=z_optcloud)
        eventNo = [] # To indicate where the axis is obtained in a later step

    ######################################################################
    #########################  Start reconstruction ######################
    ###################################################################### 
    # Resample data to reconstruct & default value is 131 x 15
    # Subscript nr means normalized & resampled
    Bx_inFR_nr,By_inFR_nr,Pt_p_nr,Ap_nr,Vx_rmn_FR,Vy_rmn_FR = var_resample(A1,B_inFR,Pt,V_rmn_FR)
    Xi1 = np.linspace(Xa[0]/L0, Xa[-1]/L0, num=nx)

    # Attention on prime symbol, parameters are still derived from A' or with (1-AlphaMach)
    # Subscript with "physical" means mostly (1-AlphaMach) has been removed (no guaranteed for all).
    # Get A' contour, dPt, Bx_from_A_contour, By_from_A_contour
    A_pri_contour, dPt, \
    Bx_from_A_contour_physical, By_from_A_contour_physical = Ap_calculation(Ap_nr,Bx_inFR_nr,By_inFR_nr,Pt_p_nr)


    # Return to values with unit before normalization
    A_contour = A_pri_contour*A0
    A_contour_physical = A_contour/(1-AlphaMach)
    # np.set_printoptions(threshold=np.inf)
    # print(A_contour)

    
    Bz_from_fit_physical = Bz_calculation(A_pri_contour,B_inFR,Ab)
    Jz_from_fit_physical = Jz_calculation(dPt)
    xe, ye = x*L0/AU, y*L0/AU
    X0, Y0 = xe[I0], ye[J0]

    # Print parameters/results derived from the GS reconstruction.
    print("\n---------------------------------------------")
    print("\nParameters for the Grad-Shafranov reconstruction:")
    print("- The order of polynomials                     = {}".format(polyOrder))
    print("- The grid points                              = {}, {}".format(nx, ny))
    print("- Is the boundary of A contour selected?       = {}".format((get_Ab == 1) or (get_Ab == -1)))
    print("- The left and right extrapolation percentages = {}, {}".format(dAl0, dAr0))
    print("- Is the thermal pressure included?            = {}".format(pressureSwitch==1))
    if np.size(eventNo)>0:
        print("- Where is the flux rope axis obtained from?   = {}".format('Detection'))
    elif adjustAxis==True:
        print("- Where is the flux rope axis obtained from?   = {}".format('Reconstruction'))
        print("- The selected axis is {}.".format(z_optcloud))

    # The following parameters are calculated only when the thermal pressure is added.
    if pressureSwitch == 1:
        Bz_max = Bz_from_fit_physical[J0][I0]
        Jz_max = abs(Jz_from_fit_physical).max()
        print("\n---------------------------------------------")
        print("\nResults derived from the Grad-Shafranov reconstruction:")
        print("- The maximum of the axial magnetic field Bz   = {} nT".format(Bz_max))
        print("- The maximum of the axial current density jz  = {} A/m^2".format(Jz_max))
        
        # Estimate the toroidal & poloidal magnetic flux
        if (get_Ab == 1) or (get_Ab == -1):
            toroidalFlux, poloidalFlux, current = flux_calculation(Xi1, A_pri_contour, 
                Bz_from_fit_physical,Jz_from_fit_physical)
            print("- Estimated axial current                      = {} A".format(current))
            print("- Estimated toroidal magnetic flux             = {} Wb".format(toroidalFlux))
            print("- Estimated poloidal magnetic flux at 1 AU     = {} Wb".format(poloidalFlux))
            if helicityTwist:
                relative_helicity = helicity_calculation(A_contour_physical,
                    Bx_from_A_contour_physical,By_from_A_contour_physical,
                    Bz_from_fit_physical)
                # Unit of relative_helicity: nT^2AU^3 = (1e-9)^2 T^2AU^3
                # Unit of toroidalFlux: T m^2 = T (m/AU_unit)^2 = T AU (1/1.49e11)^2
                # Unit of toroidalFlux^2: T^2 AU^4 (1/1.49e11)^4
                # Unit of twist: (1e-9)^2 T^2AU^3 / T^2 AU^4 (1/1.49e11)^2 = (1e-9)^2 / (1/1.49e11)^4 per AU
                avg_twist = relative_helicity*(1e-9**2)/((toroidalFlux/AU**2)**2)
                print("- Estimated relative helicity per unit length  = {} nT^2AU^3".format(relative_helicity))
                print("- Estimated average twist per AU               = {}".format(avg_twist))
    
    ######################################################################
    #########################  Print on terminal #########################
    ######################################################################
    # Print information of figures.
    print("\n---------------------------------------------")
    print("\nFigures (see labels for more info):")
    if detection2Reconstruction & (adjustInterval == False):
        print("- Filter process of the original data in detection.")
        print("- Time-series plot of detection interval.")
    if adjustInterval & (detection2Reconstruction == False): 
        print("- Time-series plot for adjusted interval.")
    if plotHodogram: 
        print("- Hodograms: subscripts 1, 2, and 3 correspond to the maximum, \
            \n   intermediate, and minimum variance in the magnetic field.")
    if checkHT: print("- Comparison between -VHT x B and -VHT(t) x B.")
    if plotTimeSeries: print("- Time-series from the original spacecraft dataset.")
    if plotWalenRelation: print("- The Walen relation between the remaining \
        \n   flow and Alfven velocities. Red, green, and blue circles \
        \n   represent components (R, T, N) or (X, Y, Z).")
    print("- Four pressures versus A'.")
    if checkPtAFitting: print("- Fitting curves of Pt'(A').")
    print("- The reconstructed cross-sectional map.")
    print("- Pt' versus A'.")
    if plotJz: print("- The map of the axial current density Jz.")
    
    # Plot the reconstruction results
    plotReconstruction(A_contour,Bz_from_fit_physical,VA_mean,
                       Bx_inFR_nr,By_inFR_nr,Ap_nr,Pt_p_nr,
                       Vx_rmn_FR,Vy_rmn_FR,Jz=Jz_from_fit_physical)
    
    print("\n---------------------------------------------")
    if pressureSwitch == 0:
        print("\nIf satisfying with the current Pt'(A'),")
        print("please set pressureSwitch = 1 & run again to \
            \n   have the final reconstruction results.")
    elif pressureSwitch == 1:
        if saveFig:
            print("\nFigures saved to {}".format(rootDir))
        print("\nDone.")
    plt.show()
    
    return
