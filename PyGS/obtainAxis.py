"""
This file is used to obtain the flux rope axis.
Translated & upgraded from the original GSR code "optcloud.m".

Authors: Yu Chen & Qiang Hu
Affiliation: CSPAR & Department of Space Science @ The University of Alabama in Huntsville

References: 
Hu, Q., & Sonnerup, B. U. Ö. 2001, GeoRL, 28, 467
Hu, Q., & Sonnerup, B. U. Ö. 2002, JGR, 107, 1142
Hu, Q., Smith, C. W., Ness, N. F., & Skoug, R. M. 2004, JGRA, 109, A03102
Teh, W.-L. 2018, EP&S, 70, 1
Chen, Y., Hu, Q., Zhao, L., Kasper, J. C., & Huang, J. 2021, ApJ, 914, 108

Version 0.0.1 @ June 2023

"""

from __future__ import division
import os
import pickle
import math
import numpy as np
import pandas as pd
from numpy import linalg as la
import pandas as pd
import scipy
from scipy import signal
from scipy import interpolate
from datetime import datetime
from datetime import timedelta
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from PyGS import FluxRopeDetection as detect
from PyGS import ReconstructionMisc

def var_downsample(x1, x2):
    # Downsample two variables

    x1_downsample = signal.resample_poly(x1,nx,len(x1))
    x2_downsample = signal.resample_poly(x2,nx,len(x2))
    
    return x1_downsample, x2_downsample

def MVABperp(B, flag, ev):
    # Calculate the MVAB frame

    B_UnitVector = np.zeros((len(B),3))
    if flag == 0:
        B_Magnitude = np.array(np.sqrt(np.square(B).sum(axis=1)))
        B_UnitVector[:,0] = B.iloc[:,0]/B_Magnitude
        B_UnitVector[:,1] = B.iloc[:,1]/B_Magnitude
        B_UnitVector[:,2] = B.iloc[:,2]/B_Magnitude
    else:
        B_UnitVector = np.array([B.iloc[:,0], B.iloc[:,1], B.iloc[:,2]]).T

    delta = np.eye(len(ev))
    M = np.zeros((3,3))
    P = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            M[i,j] = np.mean((B_UnitVector[:,i]*B_UnitVector[:,j]))\
            -np.mean(B_UnitVector[:,i])*np.mean(B_UnitVector[:,j])
            P[i,j] = delta[i,j]-ev[i]*ev[j]

    # D, X = scipy.linalg.eigh(P@M@P)
    D, X = la.eigh(P@M@P)
    limda, In = sorted(D), np.argsort(D)

    X = np.array([X[:,In[2]], X[:,In[1]], X[:,In[0]], np.flipud(limda)]).T
    aBxi = np.mean(np.matmul(B_UnitVector,X[:,0:3]), axis=0)

    return X

def MVAB2(B, flag):
    # Calculate the MVAB frame

    B_UnitVector = np.zeros((len(B),3))
    if flag == 0:
        B_Magnitude = np.array(np.sqrt(np.square(B).sum(axis=1)))
        B_UnitVector[:,0] = B[0]/B_Magnitude
        B_UnitVector[:,1] = B[1]/B_Magnitude
        B_UnitVector[:,2] = B[2]/B_Magnitude
    elif flag == 3:
        B_Magnitude = np.array(np.sqrt(B[0]**2 + B[2]**2))
        B_UnitVector[:,0] = B[0]/B_Magnitude
        B_UnitVector[:,1] = B[1]
        B_UnitVector[:,2] = B[2]/B_Magnitude
    elif flag ==2:
        B_Magnitude = np.array(np.sqrt(B[1]**2))
        B_UnitVector[:,0] = B[0]/B_Magnitude
        B_UnitVector[:,1] = B[1]
        B_UnitVector[:,2] = B[2]/B_Magnitude
    else:
        B_UnitVector = np.array([B[0], B[1], B[2]]).T

    M = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            M[i,j] = np.mean((B_UnitVector[:,i]*B_UnitVector[:,j]))\
            -np.mean(B_UnitVector[:,i])*np.mean(B_UnitVector[:,j])
    
    D, X = la.eig(M)
    limda, In = sorted(D), np.argsort(D)
    X = np.array([X[:,In[2]], X[:,In[1]], X[:,In[0]]]).T
    limda_flip = np.flipud(limda)
    r = len(B)-1
    D21 = np.sqrt(limda_flip[2]*(limda_flip[1]+limda_flip[0]
                              -limda_flip[2])/(r*(limda_flip[0]-limda_flip[1])**2))
    D23 = np.sqrt(limda_flip[2]*(limda_flip[1]+limda_flip[2]
                              -limda_flip[2])/(r*(limda_flip[1]-limda_flip[2])**2))
    D13 = np.sqrt(limda_flip[2]*(limda_flip[0]+limda_flip[2]
                              -limda_flip[2])/(r*(limda_flip[0]-limda_flip[2])**2))
    D = np.array([D21, D23, D13])
    M13 = np.array([limda_flip, D]).T

    return X, M13

def MVAdirection(VHT, B):
    """Find the axis via the MVAB.
    The initial guess is the intermediate variance direction.
    Return transformed B, VHT, matrix and MVA directions."""

    VHT_UnitVector = -VHT/np.array(np.sqrt(np.square(VHT).sum()))
    X_perp = MVABperp(B, 0, VHT_UnitVector)
    Y, X = X_perp[:,0], VHT_UnitVector
    Z = np.cross(X, Y)
    transMatrix = np.array([X, Y, Z]).T
    B_initialGuess = np.matmul(B, transMatrix)
    VHT_initialGuess = np.matmul(-VHT, transMatrix)
    XX, Lamerr = MVAB2(B_initialGuess, 0)
    leppz = XX[:,1]
    if np.dot(leppz, [0,0,1]) < 0: leppz = -leppz
    lep_polar = np.arccos(np.dot(leppz, [0,0,1]))
    lep_long = np.arccos(np.dot(leppz, [1,0,0])/np.sin(lep_polar))
    if np.dot(leppz, [0,1,0]) < 0: lep_long = 2*np.pi-lep_long

    return B_initialGuess, VHT_initialGuess, transMatrix, lep_polar, lep_long

def extreme_n_turning(longitude, latitude, VHT_initialGuess, B_initialGuess, **kwargs):
    """Find the extreme value of A' for a pair of angles (long,lat) or (phi,theta).
    Return A values, extreme A value, index, and B_inFR."""
    
    i1, i2 = None, nx
    # Calculate the matrix to FR frame
    matrix_transToFR = detect.angle2matrix(latitude, longitude-90, -VHT_initialGuess)
    # Transform B into FR frame
    B_inFR = B_initialGuess.dot(matrix_transToFR)
    Bx_inFR, By_inFR, Bz_inFR = B_inFR.iloc[:,0], B_inFR.iloc[:,1], B_inFR.iloc[:,2]
    # Projection of VHT
    XSV = VHT_initialGuess-np.dot(VHT_initialGuess, matrix_transToFR.T[2])*matrix_transToFR.T[2]
    VHT_XS = la.norm(XSV)
    ds = dt*VHT_XS
    # Calculate A'
    dA = np.zeros(nx)
    for i in range(1, nx):
        dA[i] = -(1-AlphaMach)*(By_inFR[i] + By_inFR[i-1])*ds[i]*0.5
    A_prime = np.cumsum(dA)
    
    # Call separately for residue map and two angles comparison
    if 'longitude_var1' in kwargs:longitude_var = kwargs['longitude_var1']
    else: longitude_var = longitude
    if 'latitude_var1' in kwargs: latitude_var = kwargs['latitude_var1']
    else: latitude_var = latitude
    
    # For each pair of (phi,theta), find the extreme value of A & turn point
    if ((longitude_var == 0) & (latitude_var == 0)) | (i1 is None):# interval with equal A value at ends
        if (By_inFR[0]  <=  0) | (By_inFR[1] < 0):
            # if A is positive
            A_turn_index0 = np.argwhere(A_prime == A_prime.max())
            A_turn_index = A_turn_index0[0][0]
            if A_prime[0]<A_prime[nx-1]:
                i1 = np.argwhere(abs(A_prime[0:A_turn_index+1]-A_prime[nx-1]) 
                    == min(abs(A_prime[0:A_turn_index+1]-A_prime[nx-1])))[0][0]
                i2 = nx-1
            else:
                i1 = 0
                i2 = np.argwhere(abs(A_prime[A_turn_index:nx]-A_prime[0]) 
                    == min(abs(A_prime[A_turn_index:nx]-A_prime[0])))[0][0]+A_turn_index
            A_extreme = max(A_prime[i1], A_prime[i2])
        else: # if A is negative
            A_turn_index0 = np.argwhere(A_prime == A_prime.min())
            A_turn_index = A_turn_index0[0][0]
            if A_prime[0]>A_prime[nx-1]:
                i1 = np.argwhere(abs(A_prime[0:A_turn_index+1]-A_prime[nx-1]) 
                    == min(abs(A_prime[0:A_turn_index+1]-A_prime[nx-1])))[0][0]
                i2 = nx-1
            else:
                i1 = 0
                i2 = np.argwhere(abs(A_prime[A_turn_index:nx]-A_prime[0]) 
                    == min(abs(A_prime[A_turn_index:nx]-A_prime[0])))[0][0]+A_turn_index
            A_extreme = min(A_prime[i1], A_prime[i2])

    return A_prime, A_extreme, A_turn_index, i1, i2, B_inFR

def spilt_PtA(A_prime, A_extreme, A_turn_index, i1, i2, Bz_inFR):
    """According to turning point, spilt two Pt'(A') curves
    Return four arrays."""
    
    # Pt has the full expression to find the optimal axis with Pt versus A
    Pt = ((1-AlphaMach)**2)*np.array(Bz_inFR)**2/(2*miu) \
    + pressureSwitch*np.array(Ppe) + pressureSwitch*AlphaMach*(1-AlphaMach)*PBB
    # Find A' with a finer grid
    A_finer = np.linspace(A_prime[A_turn_index], A_extreme, math.ceil((i2-i1+1)/2))
    # Sort A' and get indices
    A_primehalf, indexA1h = sorted(A_prime[i1:A_turn_index+1]), np.argsort(A_prime[i1:A_turn_index+1])
    # Interpolate Pt'(A') for first half
    interp_func_Pt1 = interpolate.interp1d(A_primehalf, Pt[indexA1h+i1])
    Pt1half = interp_func_Pt1(A_finer)
    # Sort A' and get indices
    A2half, indexA2h = sorted(A_prime[A_turn_index:i2+1]), np.argsort(A_prime[A_turn_index:i2+1])
    # Interpolate Pt'(A') for first half
    interp_func_Pt2 = interpolate.interp1d(A2half, Pt[indexA2h+A_turn_index])
    Pt2half = interp_func_Pt2(A_finer)

    return Pt, A_finer, Pt1half, Pt2half

def findResidue(VHT_initialGuess, B_initialGuess, longitude_var, latitude_var):
    """Calculate residue for each pair of (phi,theta)"""
    
    Pt = np.zeros(nx)
    A_prime, A_extreme, A_turn_index, i1, i2 = [], [], [], [], []

    # Find the extreme A' values & turning point index,
    # split Pt' versus A' branches, and calculate residue.
    A_prime, A_extreme, A_turn_index, \
    i1, i2, B_inFR = extreme_n_turning(longitude_var+90, 
        latitude_var, VHT_initialGuess,B_initialGuess)
    # If i1 & i2 are between two ends and turn point, set residue to be 20
    if (i1 is None) | (i2 is None) | (i1 == nx-1) | (i2 == 0) \
    | (abs(i1-i2) <= nx/2) | (i1 == A_turn_index) | (i2 == A_turn_index):
        residue = 20
    else: 
        # Get Pt values for two branches
        Pt, A_finer, Pt1half, Pt2half = spilt_PtA(A_prime, 
            A_extreme, A_turn_index, i1, i2, B_inFR.iloc[:,2])
        # Calculate residue
        pwsub = abs(Pt2half-Pt1half)
        residue = la.norm(pwsub)/(max(Pt[i1:i2+1])-min(Pt[i1:i2+1])) # sqrt(length(pwsub)); 

    return residue

def findMinResidue(VHT_initialGuess, B_initialGuess, min_residue_index, angle1, angle2, func, **kwargs):
    """Find the residue for three pairs of directions.
    # First time:
    Find residue for two pairs of directions only, (min, mva) & (min, select)
    # Second time:
    Find it for [select, min, mva]
    """
    
    min_residue, uncertainty = np.zeros(2),np.zeros(2)
    Pt = np.zeros(nx)
    A_prime, A_extreme, A_turn_index, i1, i2 = [], [], [], None, None
    
    # Extract (long,lat) for the minimum residue
    min_residue_row, min_residue_column = min_residue_index[0][0], min_residue_index[0][1]
    min_residue_long, min_residue_lat = round(longitude_all[min_residue_row]*180/np.pi), round(latitude_all[min_residue_column]*latmax)
    min_residue_long_rad, min_residue_lat_rad = min_residue_long*np.pi/180, min_residue_lat*np.pi/180
    
    if (func == 'select_z'):
        # Now min_residue_long_rad is actually select_z_long
        angle3, angle4 = kwargs['angle3'], kwargs['angle4']
        long_min, lat_min = [min_residue_long_rad,angle1,angle3], [min_residue_lat_rad,angle2,angle4]
        fig_multi_directions, ax1 = plt.subplots(1, 1, figsize=(4.5, 4))
        len_direction = 3
    else:
        long_min, lat_min = [min_residue_long_rad,angle1], [min_residue_lat_rad,angle2]
        len_direction = 2
    
    # Based on these two pairs of angles, find the extreme A, turning point,
    # split Pt' versus A' branches, and calculate residue.
    for kk in range(len_direction):
        A_prime, A_extreme, A_turn_index, \
        i1, i2, B_inFR = extreme_n_turning(long_min[kk]*180/np.pi+90, 
            lat_min[kk]*180/np.pi, VHT_initialGuess,B_initialGuess, 
            longitude_var1 = min_residue_long, latitude_var1 = min_residue_lat)
        # Calculate the residue
        if (i1 is None) | (i2 is None) | (i1 == nx-1) | (i2 == 0) \
        | (abs(i1-i2) <= nx/2) | ((min_residue_long == 0) | (min_residue_long == 180) | (min_residue_long == 360) & (min_residue_lat == 90)):
            residue = 20
        else: 
            Pt, A_finer, Pt1half, Pt2half = spilt_PtA(A_prime, 
                A_extreme, A_turn_index, i1, i2, B_inFR.iloc[:,2])
            # Calculate uncertainty by error propagation
            pwsub = abs(Pt2half-Pt1half)
            sig2_1st = np.var(signal.detrend(Pt1half))
            sig2_2nd = np.var(signal.detrend(Pt2half))

            # Does not plot for [mva, min] directions
            # Plot [select, min, mva] together
            if (func == 'select_z'):
                plotMutliDirections(A_finer, Pt1half, Pt2half, kk)
        
        # The difference residue
        if kk < 2:
            if (max(Pt[i1:i2+1]) == min(Pt[i1:i2+1])):
                min_residue[kk] = np.nan
                uncertainty[kk] = np.nan
            else:
                min_residue[kk] = la.norm(pwsub)/(max(Pt[i1:i2+1])-min(Pt[i1:i2+1])) #/sqrt(length(pwsub)); 
                uncertainty[kk] = np.sqrt(sig2_1st+sig2_2nd)/(max(Pt[i1:i2+1])-min(Pt[i1:i2+1])) 

    return min_residue, min_residue_long_rad, min_residue_lat_rad, uncertainty

def plotMutliDirections(A1, Pt1, Pt2, kk):
    """Plot Pt'(A') in three directions.
    Will show up after selecting a point."""

    if len(A1) == 0:
        print("\nError! Please try a different position.")
        fig_residue_map.savefig(fig_sav_path, format='png', dpi=450)
        print("Temporary residue map saved to {}".format(rootDir))
        exit()

    line_color = ['tab:red','tab:blue','tab:green']
    plt.xlabel("$A'$")
    plt.ylabel("$P_t'$")
    pta1, = plt.plot(A1, Pt1, marker = 'o', markerfacecolor='none', 
        linestyle='-', linewidth=1, color = line_color[kk])
    pta12, = plt.plot(A1, Pt2, marker = '*',linestyle='-', linewidth=1, color = line_color[kk])
    plt.title("$P_t'$ versus $A'$ in different directions", fontsize = 10)
    if kk == 0:
        plt.legend(labels=["Selected-z $Pt'(A')_{1st}$", 
            "Selected-z $Pt'(A')_{2nd}$"], loc='best',prop={'size':6})
    if kk == 1:
        plt.legend(labels=["Selected-z $Pt'(A')_{1st}$", 
            "Selected-z $Pt'(A')_{2nd}$", "Min. Res. $Pt'(A')_{1st}$", 
            "Min. Res. $Pt'(A')_{2nd}$"], loc='best',prop={'size':6})
    if kk == 2:
        plt.legend(labels=["Selected-z $Pt'(A')_{1st}$", 
            "Selected-z $Pt'(A')_{2nd}$", "Min. Res. $Pt'(A')_{1st}$", 
            "Min. Res. $Pt'(A')_{2nd}$","MVA $Pt'(A')_{1st}$", 
            "MVA $Pt'(A')_{2nd}$"], loc='best',prop={'size':6})

    return

def minDirection(VHT_initialGuess, B_initialGuess, transMatrix, lep_long, lep_polar):
    """Find the minimum residue direction.
    Return the full residue map for plot, and minimum direction results."""

    global latmax, longitude_all, latitude_all
    opt_residue_all = np.zeros((37,19)) # residue in 2D arrays
    i1, i2 = None, nx
    latmax, latstep, long_intial, longstep, long_last = 90, 5, 0, 10, 360
    
    # Loop (longitude,latitude) & calculate residue for each pair of angles
    for longitude_var in range(long_intial,long_last+1, longstep):
        for latitude_var in range(0, latmax+1, latstep):
            residue = findResidue(VHT_initialGuess, B_initialGuess, 
                longitude_var, latitude_var)
            # Save residue for the corresponding (long, lat)
            opt_residue_all[int((longitude_var-long_intial)/longstep+1)-1, 
            int(latitude_var/latstep+1)-1] = residue

    # Also output angles from the previous step
    longitude_all = np.arange(long_intial,long_last+1,longstep)*np.pi/180
    latitude_all = np.arange(0,latmax+1,latstep)/latmax
    longitude_all_reshape = longitude_all.reshape((len(longitude_all),1))
    latitude_all_reshape = latitude_all.reshape((1,len(latitude_all)))
    X = np.cos(longitude_all_reshape) * latitude_all_reshape
    Y = np.sin(longitude_all_reshape) * latitude_all_reshape

    # Find the minimum residue & indices
    opt_residue_lesslayer = opt_residue_all[:, 0:np.size(X,1)-1]
    min_residue = opt_residue_lesslayer.min()
    min_residue_index = np.argwhere(opt_residue_all == min_residue) 
    # Compare with MVA direction
    min_residue_2dir, min_residue_long_rad, min_residue_lat_rad, \
    uncertainty_mva = findMinResidue(VHT_initialGuess, B_initialGuess, 
        min_residue_index, lep_long, lep_polar, func='min_mva')
    # Get three components of z-axis from the minimum residue direction angles
    Zmin = [np.sin(min_residue_lat_rad)*np.cos(min_residue_long_rad),
    np.sin(min_residue_lat_rad)*np.sin(min_residue_long_rad),
    np.cos(min_residue_lat_rad)]
    
    return opt_residue_all, X, Y, min_residue_long_rad, min_residue_lat_rad, min_residue_2dir, Zmin

def plotResidue(x, y, min_residue_long_rad, min_residue_lat_rad, opt_residue_all,min_residue):
    # Plot the residue map.

    # The second part to plot Residue
    chi2cl = [1, 2.3] # uncertainty
    reslevel = [chi2cl[0],chi2cl[0]]+min_residue  
    nlong, nlat = np.size(x,0), np.size(x,1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal', adjustable='box')
    plt.plot(x,y,marker = '.', markersize = 1.5, linestyle='', color='gray')
    plt.plot((min_residue_lat_rad/(np.pi/2))*np.cos(min_residue_long_rad), 
        (min_residue_lat_rad/(np.pi/2))*np.sin(min_residue_long_rad), 
        marker = '.', markersize = 10.0, linestyle='', color='k')
    plt.axis('off')
    
    plt.text(1.1, 0, '0$^\circ$', ha='center', va='center', fontsize=9, color='gray')
    plt.text(0.1, 1.05, '90$^\circ$', ha='center', va='center', fontsize=9, color='gray')
    plt.text(-1.1, 0, '180$^\circ$', ha='center', va='center', fontsize=9, color='gray')
    plt.text(0, -1.1, '270$^\circ$', ha='center', va='center', fontsize=9, color='gray')
    plt.text(0, y[9,4], '20$^\circ$', ha='center', va='center', fontsize=9, color='gray')
    plt.text(0.025, y[9,8], '40$^\circ$', ha='center', va='center', fontsize=9, color='gray')
    plt.text(0.05, y[9,12], '60$^\circ$', ha='center', va='center', fontsize=9, color='gray')
    plt.text(0.075, y[9,16], '80$^\circ$', ha='center', va='center', fontsize=9, color='gray')
    plt.contour(x[:,0:nlat-1], y[:,0:nlat-1], opt_residue_all[0:nlong,0:nlat-1], [reslevel[0]], cmap='binary_r')
    return

def checkNumPoints(B_initialGuess_temp,Ppe_temp, PBB_temp, dt_temp):
    """Check how many data points in this interval
    If interval is too long, downsample to 101 points."""
    
    global nx
    nx = min(101,len(B_initialGuess_temp))
    if nx != len(B_initialGuess_temp):
        B_initialGuess_array, Ppe = var_downsample(B_initialGuess_temp, Ppe_temp)
        B_initialGuess = pd.DataFrame(B_initialGuess_array, columns=['0','1','2'])
        B_Mag = np.array(np.sqrt(np.square(B_initialGuess).sum(axis=1)))
        PBB = B_Mag**2/(2*miu)
        dt_upper = np.ones(nx-1)*(sum(dt_temp)/(nx-1))
        dt = np.append(dt_temp[0],dt_upper)
    else:
        B_initialGuess, Ppe, PBB = B_initialGuess_temp, Ppe_temp, PBB_temp
        dt = dt_temp

    return B_initialGuess, Ppe, PBB, dt

def findZaxis(rootDir, timeStart, timeEnd, alldata, polyOrder, **kwargs):
    """This function finds the flux rope axis via both MVA & minimum difference residue
    For both methods, the main steps include: 
        use the intermediate direction via MVAB as the initial guess
        find the turning point & extreme values of A'
        interpolate Pt' based on A' values
        once Pt'(A') for two branches are obtained, calculate the difference residue
    Compare MVA direction, minimum residue direction, and user-selected directions.
    Also outputs information of the last two directions on the terminal.
    
    *Reference: 
    Hu, Q., & Sonnerup, B. U. Ö. 2002, JGR, 107, 1142
    Hu, Q., & Sonnerup, B. U. Ö. 2001, GeoRL, 28, 467
    *Corresponds to optcloud.m from the original GS reconstruction
    """
    
    # Constants & default settings
    global pressureSwitch, k_Boltzmann, miu, nx, fig_sav_path
    pressureSwitch = 1
    k_Boltzmann, miu = 1.3806488e-23, 4.0*np.pi*1e-7 
    fig_sav_path = rootDir + timeStart.strftime('%Y%m%d%H%M%S') \
    + '_' + timeEnd.strftime('%Y%m%d%H%M%S') + '_residue_temporary.png'
    
    adjustInterval = False
    if 'adjustInterval' in kwargs:
        adjustInterval = kwargs['adjustInterval']
    det2rec = False
    if 'det2rec' in kwargs:
        det2rec = kwargs['det2rec']
    ###################################################
    ############## (1) Get data segament ##############
    ############## (2) Determine VHT ##################
    ###################################################
    # Ppe = Pp + Pp; PBB = the magnetic pressure
    global dt, AlphaMach, Ppe, PBB
    B, Ppe_temp, PBB_temp, dt_temp, VHT_kms, \
    AlphaMach = ReconstructionMisc.clouddata(timeStart, timeEnd, 
        alldata, pressureSwitch=pressureSwitch, 
        polyOrder=polyOrder, func='optcloud_simple')
    ###################################################
    ############ (3) Initial guess via MVA ############
    ###################################################
    # Initial guess as the intermediate variance direction via MVAB
    VHT = VHT_kms*1e3
    B_initialGuess_temp, VHT_initialGuess, transMatrix, \
    lep_polar, lep_long = MVAdirection(VHT,B)

    # If interval is too long, downsample to 101 points
    B_initialGuess, Ppe, PBB, dt = checkNumPoints(B_initialGuess_temp, 
        Ppe_temp, PBB_temp, dt_temp)

    ###################################################
    ####### (4) The minimum difference residue ########
    ###################################################
    opt_residue_all, X, Y, min_residue_long_rad, min_residue_lat_rad, \
    min_residue, Zmin = minDirection(VHT_initialGuess, B_initialGuess, 
        transMatrix, lep_long, lep_polar)

    coord_transback = np.array(transMatrix.T)
    z_minres_direction = np.matmul(Zmin,np.linalg.inv(coord_transback.T))
    
    # The first part to plot Residue
    global fig_residue_map
    fig_residue_map, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    ax2.set_aspect('equal', adjustable='box')
    # The second part to plot Residue
    plotResidue(X, Y, min_residue_long_rad, min_residue_lat_rad, opt_residue_all,min_residue)

    II = 'n'
    click = 1
    while (II == 'n') & (click <= 5):
        # plotResidue(X, Y, min_residue_long_rad, min_residue_lat_rad, opt_residue_all,min_residue)

        ###################################################
        ######### (5) Calculate for selected axis #########
        ###################################################
        print("\n================ Trial No.",click,"================")
        print("\nPlease follow the instruction to select a flux rope axis.")
        print("\n- Click to select and check the other figures to see whether satisfied.")
        print("- After selection, answer the question and input either y or n.")
        print("     If negative, select an alternative point on the residue map.")
        print("- Each running can be selected five times.")
        
        # For the correct choice of z axis all the circles and stars 
        # should lie on one single-valued curve representing Pt(A).
        aa = plt.ginput(1)
        xg = aa[0][0]
        yg = aa[0][1]
        # Plot selected points and clicks
        line_color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
        plt.plot(xg, yg, marker = 'x', markersize = 6, linestyle='', color=line_color[click-1])
        plt.text(xg, yg+0.05, click, ha='center', va='center', color=line_color[click-1], fontsize=9)
        
        disg = np.sqrt((X-xg)**2+(Y-yg)**2)
        disg_min_index = np.argwhere(disg == disg.min())
        min_residue_selectz, select_long_rad, \
        select_lat_rad, uncertainty = findMinResidue(VHT_initialGuess, 
            B_initialGuess, disg_min_index, min_residue_long_rad, 
            min_residue_lat_rad, func='select_z', angle3=lep_long, angle4=lep_polar)
        z_select_direction = [np.sin(select_lat_rad)*np.cos(select_long_rad), 
        np.sin(select_lat_rad)*np.sin(select_long_rad), np.cos(select_lat_rad)]
        Zzpg = np.matmul(z_select_direction, np.linalg.inv(coord_transback.T))
        
        print("\n---------------------------------------------")
        print("\nFlux rope axis with the minimum residue direction is {}.".format(z_minres_direction))
        # print("The corresponding angle theta is {} and phi is {}.".format(round(z_minres_theta,4), round(z_minres_phi,4)))
        
        # Determine whether z-axis needs to be flipped
        Bz_inFR_temp = B@Zzpg
        num_Bz_lt0 = (Bz_inFR_temp[2]<0).sum()
        num_Bz_gt0 = (Bz_inFR_temp[2]>0).sum()
        if (num_Bz_lt0 > num_Bz_gt0):
            Zzpg = -Zzpg
            print("User-selected flux rope axis is {}.".format(Zzpg))
            print("The uncertainty of the two axes are {} and {}.".format(uncertainty[-1],uncertainty[0]))
            print('\nAttention: the direction of selected z-axis is flipped due to negative Bz.')
        else:
            print("User-selected flux rope axis is {}.".format(Zzpg))
            print("The uncertainty of the two axes are {} and {}.".format(uncertainty[-1],uncertainty[0]))
        # Plot four pressures versus A' to determine the result
        ReconstructionMisc.clouddata(timeStart,timeEnd,alldata,
            pressureSwitch=pressureSwitch,polyOrder=polyOrder,
            fluxropeAxis=Zzpg,func='optcloud_full',plotFigure=True)
        

        ###################################################
        ########## Print information on terminal ##########
        ###################################################
        print("\n---------------------------------------------")
        print("\nFigures show parameters derived with the selected z-axis:")
        print("- The residue map.")
        print("- Pt'(A') in different directions: the MVA, the minimum residue, and selected z-axis.")
        print("- Parameters along the spacecraft path: A', \
            \n   B in flux rope frame, pressures, and plasma beta.")
        print("- Four pressures versus A'.")
        
        plt.show(block=False)
        plt.pause(1)

        print("\n---------------------------------------------")
        print("\nIs the optimal invariance direction found? [y/n]:", end = ' ')
        II = input()
        if II == 'n':
            # print("\n---------------------------------------------")
            if adjustInterval & det2rec:
                plt.close(1), plt.close(2), plt.close(4), plt.close(5), plt.close(6)
            elif adjustInterval or det2rec: 
                plt.close(1), plt.close(3), plt.close(4), plt.close(5)
            else: 
                plt.close(2), plt.close(3), plt.close(4)
            click = click+1
            if click > 5:
                print("Time out. Please run this function again.")
                fig_residue_map.savefig(fig_sav_path, format='png', dpi=450)
                print("Temporary residue map saved as {}.".format(fig_sav_path))
                quit()
        elif II == 'y': 
            plt.close(1),plt.close(2), plt.close(3), plt.close(4)
            plt.close(5), plt.close(6)
            
            np.savetxt(rootDir + 'zs_select.txt', [Zzpg], fmt="%.8f")
            print("\n##############################################")
            print('\nStart the reconstruction with the adjusted axis...')
            break
        else:
            print("Error! Please input either y or n.")
            fig_residue_map.savefig(fig_sav_path, format='png', dpi=450)
            print("Temporary residue map saved as {}.".format(fig_sav_path))
            quit()
        
    return Zzpg
            
