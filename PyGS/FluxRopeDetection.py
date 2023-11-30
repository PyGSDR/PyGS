"""
This file includes several functions needed for GS detection.
Mainly upgraded & modified from the original GS detection code by Dr. Jinlei Zheng:
- Migrate from Python 2 to Python 3
- Add more spacecraft datasets, such as PSP, Solar Orbiter, and Ulysses;
- Streamline the preprocessing step;
- Accommodate functions with multiple data resolutions;
- Add thermal pressure and extend the original GS equation;
- Add electron data as an optional input;
- Improve the VHT calculation in "findVHT";
- Adjust and streamline the clean-up process;
- Calculate more parameters for each flux rope record;
- Add an optional function in time-series plots, such as time-series plot and adjust interval;
- Add a main detection function that can call all others;

TBC...

Authors: Yu Chen, Jinlei Zheng, and Qiang Hu
Affiliation:  CSPAR & Department of Space Science @ The University of Alabama in Huntsville

References: 
- Hu, Q., & Sonnerup, B. U. Ö. 2002, JGR, 107, 1142
- Zheng, J., & Hu, Q. 2018, ApJL, 852, L23
- Hu, Q., Zheng, J., Chen, Y., le Roux, J., & Zhao, L. 2018, ApJS, 239, 12
- Teh, W.-L. 2018, EP&S, 70, 1
- Chen, Y., Hu, Q., Zhao, L., Kasper, J. C., & Huang, J. 2021, ApJ, 914, 108
- Chen, Y., & Hu, Q. 2022, ApJ, 924, 43

Version 0.0.1 @ June 2023

"""

from __future__ import division # Treat integer as float.
import os
import sys
import numpy as np # Scientific calculation package.
from numpy import linalg as la
import math
from ai import cdas # Import CDAweb API package.
from spacepy import pycdf
from scipy import integrate
from scipy import stats
from scipy.signal import savgol_filter # Savitzky-Golay filter
import scipy as sp
import time
from datetime import datetime # Import datetime class from datetime package.
from datetime import timedelta
import pandas as pd
import pickle
import multiprocessing
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt # Plot package, make matlab style plots.
from matplotlib.ticker import AutoMinorLocator
from matplotlib import dates
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from PIL import Image

import warnings
warnings.filterwarnings(action="ignore")
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
############################################## User defined functions ##############################################

# Global variables.
# Physics constants.
mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability
m_proton = 1.6726219e-27 # Proton mass. In kg.
m_alpha = 6.64424e-27 # Alpha mass
factor_deg2rad = np.pi/180.0 # Convert degree to rad.
k_Boltzmann = 1.3806488e-23 # Boltzmann constant

#############################################################################################################
def download_data(source, data_cache_dir, datetimeStart, datetimeEnd, **kwargs):
    """This functions downloads all specifed data from 
    different spacecraft, i.e., ACE, WIND, ULYSSES, PSP, and SOLARORBITER.
    All downloaded files will be put into a dictionary, which will be 
    called directly in the preprocess function."""

    # Need packages:
    # from ai import cdas # Import CDAweb API package.
    # from datetime import datetime # Import datetime class from datetime package.

    # If turn cache on, do not download from one dataset more than one time. 
    # There is a bug to casue error.
    # Make sure download every variables you need from one dataset at once.
    cdas.set_cache(True, data_cache_dir)
    
    isVerbose = False
    if 'isVerbose' in kwargs: isVerbose = kwargs['isVerbose']

    # Check if there are multiple resolutions in the current interval.
    MutliResolution = False
    
    if source == 'ACE':
        print('\nDownloading data from the ACE spacecraft.')

        # Download magnetic field data from ACE.
        if isVerbose:
            print('\nThe magnetic field data from AC_H0_MFI...')
        try:
            AC_H0_MFI = cdas.get_data('istp_public', 'AC_H0_MFI', datetimeStart, datetimeEnd, ['BGSEc'], cdf=True)
        except cdas.NoDataError:
            print('No magnetic field data!')
            AC_H0_MFI = None
            exit()
        print('Done.')

        # Download solar wind data Np, V_GSE, and Thermal speed.
        if isVerbose:
            print('The plasma bulk properties data from AC_H0_SWE...')
        # Np: Solar Wind Proton Number Density, scalar.
        # V_GSE: Solar Wind Velocity in GSE coord., 3 components.
        # Tpr: radial component of the proton temperature. 
        # Alpha to proton density ratio.
        try:
            AC_H0_SWE = cdas.get_data('istp_public', 'AC_H0_SWE', datetimeStart, datetimeEnd, ['Np','V_GSE','Tpr','alpha_ratio'], cdf=True)
        except cdas.NoDataError:
            print('No plasma data!')
            exit()
            AC_H0_SWE = None
        if isVerbose: print('Done.')

        # Download the spacecraft position data.
        if isVerbose: print('The spacecraft position data from AC_OR_SSC...')
        try:
            AC_OR_SSC = cdas.get_data('istp_public', 'AC_OR_SSC', datetimeStart, datetimeEnd, ['RADIUS'], cdf=True)
        except cdas.NoDataError:
            print('No position data!')
            AC_OR_SSC = None
        if isVerbose: print('Done.')

        data_dict = {'ID':'ACE', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 
        'AC_H0_MFI':AC_H0_MFI, 'AC_H0_SWE':AC_H0_SWE, 'AC_OR_SSC':AC_OR_SSC, 'MutliResolution':MutliResolution}
    
    elif source == 'WIND':
        print('\nDownloading data from the WIND spacecraft...')

        # Download magnetic field data.
        if isVerbose: print('\nThe magnetic field data from WI_H0_MFI...')
        try:
            WI_H0_MFI = cdas.get_data('istp_public', 'WI_H0_MFI', datetimeStart, datetimeEnd, ['BGSE'], cdf=True)
        except cdas.NoDataError:
            print('No magnetic field data!')
            WI_H0_MFI = None
            exit()
        if isVerbose: print('Done.')

        # Download solar wind data.
        if isVerbose: print('The plasma bulk properties data from WI_K0_SWE...')
        try:
            WI_K0_SWE = cdas.get_data('istp_public', 'WI_K0_SWE', datetimeStart, datetimeEnd, ['Np','V_GSE','THERMAL_SPD','SC_pos_R'], cdf=True)
        except cdas.NoDataError:
            print('No plasma data!')
            exit()
            WI_K0_SWE = None
        if isVerbose: print('Done.')

        # Download electron data. Unit is Kelvin.
        if (datetimeStart>=datetime(1994,12,29,0,0,2) and datetimeEnd<=datetime(2001,5,31,23,59,57)):
            if isVerbose: print('The electron data from WI_H0_SWE...')
            try:
                WI_H0_SWE = cdas.get_data('istp_public', 'WI_H0_SWE', datetimeStart, datetimeEnd, ['Te','el_density'], cdf=True)
            except cdas.NoDataError:
                print('No electron data available!')
        elif (datetimeStart>=datetime(2002,8,16,0,0,5)):
            if isVerbose: print('The electron data from WI_H5_SWE...')
            try:
                WI_H5_SWE = cdas.get_data('istp_public', 'WI_H5_SWE', datetimeStart, datetimeEnd, ['T_elec','N_elec'], cdf=True)
            except cdas.NoDataError:
                print('No electron data!')
        else:
            print('No electron data!')
        if isVerbose: print('Done.')

        # Download Alpha number density (Na (n/cc) from non-linear analysis).
        if isVerbose: print('The alpha data from WI_H1_SWE...')
        try:
            WI_H1_SWE = cdas.get_data('istp_public', 'WI_H1_SWE', datetimeStart, datetimeEnd, ['Alpha_Na_nonlin','Alpha_W_nonlin'], cdf=True)
        except cdas.NoDataError:
            print('No alpha particle data!')
            WI_H1_SWE = None
        if isVerbose: print('Done.')

        if 'WI_H0_SWE' in locals():
            data_dict = {'ID':'WIND', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 
            'WI_H0_MFI':WI_H0_MFI, 'WI_K0_SWE':WI_K0_SWE, 'WI_H1_SWE':WI_H1_SWE, 'WI_H0_SWE':WI_H0_SWE, 'MutliResolution':MutliResolution}
        elif 'WI_H5_SWE' in locals():
            data_dict = {'ID':'WIND', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 
            'WI_H0_MFI':WI_H0_MFI, 'WI_K0_SWE':WI_K0_SWE, 'WI_H1_SWE':WI_H1_SWE, 'WI_H5_SWE':WI_H5_SWE, 'MutliResolution':MutliResolution}
        else:
            data_dict = {'ID':'WIND', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 
            'WI_H0_MFI':WI_H0_MFI, 'WI_K0_SWE':WI_K0_SWE, 'WI_H1_SWE':WI_H1_SWE, 'MutliResolution':MutliResolution}

    elif source == 'ULYSSES':
        print('\nDownloading data from the Ulysses spacecraft.')

        # Download magnetic field data.
        if isVerbose: print('\nThe magnetic field data from UY_1MIN_VHM...')
        try:
            UY_1MIN_VHM = cdas.get_data('istp_public', 'UY_1MIN_VHM', datetimeStart, datetimeEnd, ['B_RTN'], cdf=True)
        except cdas.NoDataError:
            print('No magnetic field data!')
            UY_1MIN_VHM = None
            exit()
        if isVerbose: print('Done.')

        # Download solar wind data.
        if isVerbose: print('The plasma bulk properties data from UY_M0_BAI...')
        # Density: proton & alpha
        # Temperature: T-large & T-small
        # Velocity in RTN
        try:
            UY_M0_BAI = cdas.get_data('istp_public', 'UY_M0_BAI', datetimeStart, datetimeEnd, ['Density','Temperature','Velocity'], cdf=True)
        except cdas.NoDataError:
            print('No plasma data!')
            exit()
            UY_M0_BAI = None
        if isVerbose: print('Done.')

        # Download the spacecraft position data.
        if isVerbose: print('The spacecraft position data from UY_COHO1HR_MERGED_MAG_PLASMA...')
        try:
            UY_SC_POS = cdas.get_data('istp_public', 'UY_COHO1HR_MERGED_MAG_PLASMA', datetimeStart, datetimeEnd, ['heliocentricDistance','heliographicLatitude'], cdf=True)
        except cdas.NoDataError:
            print('No position data!')
            UY_SC_POS = None
        if isVerbose: print('Done.')

        data_dict = {'ID':'ULYSSES', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 
        'UY_1MIN_VHM':UY_1MIN_VHM, 'UY_M0_BAI':UY_M0_BAI, 'UY_SC_POS':UY_SC_POS, 'MutliResolution':MutliResolution}

    elif source == 'PSP':
        print('\nDownloading data from the PSP spacecraft.')

        # Download magnetic field data.
        if isVerbose: print('\nThe magnetic field data from PSP_FLD_L2_MAG_RTN...')
        try:
            PSP_FLD_L2_MAG_RTN = cdas.get_data('istp_public', 'PSP_FLD_L2_MAG_RTN', datetimeStart, datetimeEnd, ['psp_fld_l2_mag_RTN'], cdf=True)
        except cdas.NoDataError:
            print('No magnetic field data!')
            PSP_FLD_L2_MAG_RTN = None
            exit()
        if isVerbose: print('Done.')

        # Download solar wind data.
        if isVerbose: print('The plasma bulk properties & spacecraft position data from PSP_SWP_SPC_L3I...')
        # Subscript gd means "Only Good Quality"
        # wp_moment_gd: radial [most probable] thermal speed (km/s)
        try:
            PSP_SWP_SPC_L3I = cdas.get_data('istp_public', 'PSP_SWP_SPC_L3I', datetimeStart, datetimeEnd, 
                ['vp_moment_RTN_gd', 'np_moment_gd', 'wp_moment_gd','sc_pos_HCI', 'sc_vel_HCI'], cdf=True)
        except cdas.NoDataError:
            print('No plasma data!')
            PSP_SWP_SPC_L3I = None
            exit()
        if isVerbose: print('Done.')

        # Check if there are mutliple resolutions in PSP data.
        # Encounter mode: 0.874s & 1.7476s
        #   Data will be processed to 1s & 28s, thus MutliResolution = True
        # Cruise mode: 27.962s
        #   Data will be processed to 28s only.
        # It is better not to mix data with different resolutions.

        V_index_1 = PSP_SWP_SPC_L3I['Epoch'][1:-1]
        V_index_2 = PSP_SWP_SPC_L3I['Epoch'][0:-2]
        V_index_diff = V_index_1 - V_index_2
        
        if (len(V_index_diff) == (V_index_diff < timedelta(seconds=2)).sum()):
            MutliResolution = True
        elif (len(V_index_diff) == (V_index_diff > timedelta(seconds=27)).sum()):
            MutliResolution = False
        else:
            print('\nError: multiple resolutions detected!')
            print('Please adjust the searching interval.')
            exit()
        
        # Download alpha particle data
        if isVerbose: print('The alpha particle data from PSP_SWP_SPI_SF0A_L3_MOM...')
        # Be careful since the SPAN-Ion does not have full FOV.
        # Partial moment, temperature is in eV.
        try:
            PSP_SWP_SPI_SF0A_L3_MOM = cdas.get_data('istp_public', 'PSP_SWP_SPI_SF0A_L3_MOM', datetimeStart, datetimeEnd, 
                ['DENS','TEMP'], cdf=True)
        except cdas.NoDataError:
            print('No alpha particle data!')
            PSP_SWP_SPI_SF0A_L3_MOM = None
        if isVerbose: print('Done.')

        # Download electron data.
        if isVerbose: print('The electron data from PSP_FLD_L3_SQTN_RFS_V1V2...')
        # electron number density & core temperature (eV) from Simplified Quasi-Thermal Noise (SQTN) 
        try:
            PSP_FLD_L3_SQTN_RFS_V1V2 = cdas.get_data('istp_public', 'PSP_FLD_L3_SQTN_RFS_V1V2', datetimeStart, datetimeEnd, 
                ['electron_density','electron_core_temperature'], cdf=True)
        except cdas.NoDataError:
            print('No electron data!')
            PSP_FLD_L3_SQTN_RFS_V1V2 = None
        if isVerbose: print('Done.')

        data_dict = {'ID':'PSP', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 
        'PSP_FLD_L2_MAG_RTN':PSP_FLD_L2_MAG_RTN, 'PSP_SWP_SPC_L3I':PSP_SWP_SPC_L3I,
        'PSP_SWP_SPI_SF0A_L3_MOM':PSP_SWP_SPI_SF0A_L3_MOM, 'PSP_FLD_L3_SQTN_RFS_V1V2':PSP_FLD_L3_SQTN_RFS_V1V2, 
        'MutliResolution':MutliResolution}

    elif source == 'SOLARORBITER':
        print('\nDownloading data from the Solar Orbiter spacecraft.')

        # Download magnetic field data.
        if isVerbose: print('\nThe magnetic field data from SOLO_L2_MAG-RTN-NORMAL...')
        try:
            SOLO_L2_MAG_RTN_NORMAL = cdas.get_data('istp_public', 'SOLO_L2_MAG-RTN-NORMAL', datetimeStart, datetimeEnd, ['B_RTN'], cdf=True)
        except cdas.NoDataError:
            print('No magnetic field data!')
            SOLO_L2_MAG_RTN_NORMAL = None
            exit()
        if isVerbose: print('Done.')

        # Download solar wind data.
        if isVerbose: print('The plasma bulk properties data from SOLO_L2_SWA-PAS-GRND-MOM...')
        # Temperature is in eV
        try:
            SOLO_L2_SWA_PAS_GRND_MOM = cdas.get_data('istp_public', 'SOLO_L2_SWA-PAS-GRND-MOM', datetimeStart, datetimeEnd, ['N','V_RTN','T','V_SOLO_RTN'], cdf=True)
        except cdas.NoDataError:
            print('No plasma data!')
            SOLO_L2_SWA_PAS_GRND_MOM = None
            exit()
        if isVerbose: print('Done.')

        V_index_1 = SOLO_L2_SWA_PAS_GRND_MOM['Epoch'][1:-1]
        V_index_2 = SOLO_L2_SWA_PAS_GRND_MOM['Epoch'][0:-2]
        V_index_diff = V_index_1 - V_index_2
        
        if (len(V_index_diff) == (V_index_diff < timedelta(seconds=5)).sum()):
            MutliResolution = True
        elif (len(V_index_diff) == (V_index_diff > timedelta(seconds=27)).sum()):
            MutliResolution = False
        else:
            print('\nError: multiple resolutions detected!')
            print('Please adjust the searching interval.')
            exit()

        # Download the spacecraft position data.
        if isVerbose: print('The spacecraft position data from SOLO_COHO1HR_MERGED_MAG_PLASMA...')
        try:
            SOLO_SC_POS = cdas.get_data('istp_public', 'SOLO_COHO1HR_MERGED_MAG_PLASMA', datetimeStart, datetimeEnd, ['radialDistance'], cdf=True)
        except cdas.NoDataError:
            print('No position data!')
            SOLO_SC_POS = None
        if isVerbose: print('Done.')

        data_dict = {'ID':'SOLARORBITER', 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}, 
        'SOLO_L2_MAG_RTN_NORMAL':SOLO_L2_MAG_RTN_NORMAL, 'SOLO_L2_SWA_PAS_GRND_MOM':SOLO_L2_SWA_PAS_GRND_MOM, 
        'SOLO_SC_POS':SOLO_SC_POS, 'MutliResolution':MutliResolution}

    else:
        print('Please specify the correct spacecraft ID, \'WIND\',\'ACE\',\'ULYSSES\',\'PSP\',\'SOLARORBITER\'')
        data_dict = None
    return data_dict

#############################################################################################################
def trim_drop_duplicates(target_dataframe, timeStart, timeEnd, drop):
    """Some times cdas API will download wrong time range.
    Trim data and drop duplicates."""

    target_dataframe = target_dataframe[(target_dataframe.index>=timeStart)
    &(target_dataframe.index<=timeEnd)]
    if drop: 
        target_dataframe.drop_duplicates(keep='first', inplace=True)
    target_dataframe.sort_index(axis=0, ascending=True, inplace=True, kind='quicksort')
    
    return target_dataframe

#############################################################################################################
def butterworth_filter_3(X_dataframe, isVerbose, tag):
    """Remove all data which fall outside three standard deviations.
    Apply Butterworth filter.
    This function treats dataframe with three components."""

    if (tag == "V") or (tag == "V1"):
        column_label = ['V0','V1','V2']
        if tag == "V":
            wn_range = [0.005, 0.05]
        elif tag == "V1":
            wn_range = [0.005] # random value since not use in the first run
    elif tag == 'B':
        column_label = ['B0','B1','B2']
        wn_range = [0.45]
    n_removed_X0_total, n_removed_X1_total, n_removed_X2_total = 0, 0, 0

    # Apply Butterworth filter.
    for Wn in wn_range:
        if tag == 'V1':
            X0_dif_std, X1_dif_std, X2_dif_std = X_dataframe.std(skipna=True, numeric_only=True)
            X0_dif_mean, X1_dif_mean, X2_dif_mean = X_dataframe.mean(skipna=True, numeric_only=True)
            X_dataframe_dif = X_dataframe.copy()
        else:
            # print('Applying Butterworth filter with cutoff frequency = {}, remove spikes...'.format(Wn))
            # Note that, sp.signal.butter cannot handle np.nan. Fill the nan before using it.
            X_dataframe.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
            X_dataframe.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
            # Create an empty DataFrame to store the filtered data.
            X_dataframe_LowPass = pd.DataFrame(index = X_dataframe.index, columns = [column_label[0], column_label[1], column_label[2]])
            # Design the Buterworth filter.
            N  = 2    # Filter order
            B, A = sp.signal.butter(N, Wn, output='ba')
            # Apply the filter.
            try:
                X_dataframe_LowPass[column_label[0]] = sp.signal.filtfilt(B, A, X_dataframe[column_label[0]])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                X_dataframe_LowPass[column_label[0]] = X_dataframe[column_label[0]].copy()
            
            try:
                X_dataframe_LowPass[column_label[1]] = sp.signal.filtfilt(B, A, X_dataframe[column_label[1]])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                X_dataframe_LowPass[column_label[1]] = X_dataframe[column_label[1]].copy()
            
            try:
                X_dataframe_LowPass[column_label[2]] = sp.signal.filtfilt(B, A, X_dataframe[column_label[2]])
            except:
                print('Encounter exception, skip sp.signal.filtfilt operation!')
                X_dataframe_LowPass[column_label[2]] = X_dataframe[column_label[2]].copy()

            # Calculate the difference between X_dataframe_LowPass and X_dataframe.
            X_dataframe_dif = pd.DataFrame(index = X_dataframe.index, columns = [column_label[0], column_label[1], column_label[2]]) # Generate empty DataFrame.
            X_dataframe_dif[column_label[0]] = X_dataframe[column_label[0]] - X_dataframe_LowPass[column_label[0]]
            X_dataframe_dif[column_label[1]] = X_dataframe[column_label[1]] - X_dataframe_LowPass[column_label[1]]
            X_dataframe_dif[column_label[2]] = X_dataframe[column_label[2]] - X_dataframe_LowPass[column_label[2]]
            # Calculate the mean and standard deviation of X_dataframe_dif.
            X0_dif_std, X1_dif_std, X2_dif_std = X_dataframe_dif.std(skipna=True, numeric_only=True)
            X0_dif_mean, X1_dif_mean, X2_dif_mean = X_dataframe_dif.mean(skipna=True, numeric_only=True)
        # Set the values fall outside n*std to np.nan.
        n_dif_std = 3.89 # 99.99%
        # n_dif_std = 4.417 # ACE
        X0_remove = (X_dataframe_dif[column_label[0]]<(X0_dif_mean-n_dif_std*X0_dif_std))|(X_dataframe_dif[column_label[0]]>(X0_dif_mean+n_dif_std*X0_dif_std))
        X1_remove = (X_dataframe_dif[column_label[1]]<(X1_dif_mean-n_dif_std*X1_dif_std))|(X_dataframe_dif[column_label[1]]>(X1_dif_mean+n_dif_std*X1_dif_std))
        X2_remove = (X_dataframe_dif[column_label[2]]<(X2_dif_mean-n_dif_std*X2_dif_std))|(X_dataframe_dif[column_label[2]]>(X2_dif_mean+n_dif_std*X2_dif_std))
        X_dataframe[column_label[0]][X0_remove] = np.nan
        X_dataframe[column_label[1]][X1_remove] = np.nan
        X_dataframe[column_label[2]][X2_remove] = np.nan
        
        X0_dif_lower_boundary = X0_dif_mean-n_dif_std*X0_dif_std
        X0_dif_upper_boundary = X0_dif_mean+n_dif_std*X0_dif_std
        X1_dif_lower_boundary = X1_dif_mean-n_dif_std*X1_dif_std
        X1_dif_upper_boundary = X1_dif_mean+n_dif_std*X1_dif_std
        X2_dif_lower_boundary = X2_dif_mean-n_dif_std*X2_dif_std
        X2_dif_upper_boundary = X2_dif_mean+n_dif_std*X2_dif_std
        
        n_removed_X0 = sum(X0_remove)
        n_removed_X1 = sum(X1_remove)
        n_removed_X2 = sum(X2_remove)
        n_removed_X0_total += n_removed_X0
        n_removed_X1_total += n_removed_X1
        n_removed_X2_total += n_removed_X2
        
        if isVerbose:
            print('dif_std:', X0_dif_std, X1_dif_std, X2_dif_std)
            print('dif_mean:', X0_dif_mean, X1_dif_mean, X2_dif_mean)
            print('The {}0_dif value range within {} std is [{}, {}]'.format(tag, n_dif_std, X0_dif_lower_boundary, X0_dif_upper_boundary))
            print('The {}1_dif value range within {} std is [{}, {}]'.format(tag, n_dif_std, X1_dif_lower_boundary, X1_dif_upper_boundary))
            print('The {}2_dif value range within {} std is [{}, {}]'.format(tag, n_dif_std, X2_dif_lower_boundary, X2_dif_upper_boundary))
            print('In {}0, this operation removed {} records!'.format(tag, n_removed_X0))
            print('In {}1, this operation removed {} records!'.format(tag, n_removed_X1))
            print('In {}2, this operation removed {} records!!'.format(tag, n_removed_X2))
            print('Till now, in {}0, {} records have been removed!'.format(tag, n_removed_X0_total))
            print('Till now, in {}1, {} records have been removed!'.format(tag, n_removed_X1_total))
            print('Till now, in {}2, {} records have been removed!'.format(tag, n_removed_X2_total))
            print('\n')
    return X_dataframe

#############################################################################################################
def butterworth_filter(X_dataframe, isVerbose, tag):   
    """Remove all data which fall outside three standard deviations.
    Apply Butterworth filter.
    This function treats dataframe with only one component."""

    if (tag == 'Na') or (tag == 'Ne') or (tag == 'Np') or (tag == 'alphaRatio'):
        wn_range = [0.05,0.7]
    elif (tag == 'Ta') or (tag == 'Te') or (tag == 'Tp'):
        wn_range = [0.05, 0.45]

    n_removed_X_total = 0
    # print('Remove all X data which fall outside 3.89 standard deviations...')
    n_std = 3.89 # 99.99%.
    X_std = X_dataframe.std(skipna=True, numeric_only=True)[0]
    X_mean = X_dataframe.mean(skipna=True, numeric_only=True)[0]
    X_remove = (X_dataframe[tag]<(X_mean-n_std*X_std))|(X_dataframe[tag]>(X_mean+n_std*X_std))
    X_dataframe[tag][X_remove] = np.nan

    X_lower_boundary = X_mean-n_std*X_std
    X_upper_boundary = X_mean+n_std*X_std

    n_removed_X = sum(X_remove)
    n_removed_X_total += n_removed_X

    if isVerbose:
        print('{}_std: {}'.format(tag, X_std))
        print('{}_mean: {}'.format(tag, X_mean))
        print('The {} value range within 3.89 std is [{}, {}]'.format(tag, X_lower_boundary, X_upper_boundary))
        print('In {}, {} data has been removed!'.format(tag, n_removed_X))
        print('Till now, in {}, {} records have been removed!'.format(tag, n_removed_X_total))
        print('\n')

    # Apply Butterworth filter to X.
    for Wn in wn_range: # X
        # Note that, sp.signal.butter cannot handle X.nan. Fill the nan before using it.
        X_dataframe.fillna(method='ffill', inplace=True) # Fill missing data, forward copy.
        X_dataframe.fillna(method='bfill', inplace=True) # Fill missing data, backward copy.
        # Create an empty DataFrame to store the filtered data.
        X_LowPass = pd.DataFrame(index = X_dataframe.index, columns = [tag])
        # Design the Buterworth filter.
        N  = 2    # Filter order
        B, A = sp.signal.butter(N, Wn, output='ba')
        # Apply the filter.
        try:
            X_LowPass[tag] = sp.signal.filtfilt(B, A, X_dataframe[tag])
        except:
            print('Encounter exception, skip sp.signal.filtfilt operation!')
            X_LowPass[tag] = X_dataframe[tag].copy()
        # Calculate the difference between X_LowPass and X_dataframe.
        X_dif = pd.DataFrame(index = X_dataframe.index, columns = [tag]) # Generate empty DataFrame.
        X_dif[tag] = X_dataframe[tag] - X_LowPass[tag]
        # Calculate the mean and standard deviation of X_dif. X_dif_std is a Series object, so [0] is added.
        X_dif_std = X_dif.std(skipna=True, numeric_only=True)[0]
        X_dif_mean = X_dif.mean(skipna=True, numeric_only=True)[0]
        # Set the values fall outside n*std to X.nan.
        n_dif_std = 3.89 # 99.99%.
        X_remove = (X_dif[tag]<(X_dif_mean-n_dif_std*X_dif_std))|(X_dif[tag]>(X_dif_mean+n_dif_std*X_dif_std))
        X_dataframe[X_remove] = np.nan
            
        X_dif_lower_boundary = X_dif_mean-n_dif_std*X_dif_std
        X_dif_upper_boundary = X_dif_mean+n_dif_std*X_dif_std
            
        n_removed_X = sum(X_remove)
        n_removed_X_total += n_removed_X
                
        if isVerbose:
            print('{}_dif_std: {}'.format(tag, X_dif_std))
            print('{}_dif_mean: {}'.format(tag,X_dif_mean))
            print('The {}_dif value range within 3 std is [{}, {}]'.format(tag,X_dif_lower_boundary, X_dif_upper_boundary))
            print('In {}, this operation removed {} records!'.format(tag,n_removed_X))
            print('Till now, in {}, {} records have been removed!'.format(tag,n_removed_X_total))

    return X_dataframe

def process_to_resolution(X_dataframe, resampledt, n_interp_limit, timeStart, timeEnd, tag):
    """Interpolate/resample (usually downsample) data to fixed resolution"""

    if tag == 'RD': n_interp_limit = None
    # Linear fit first, to make sure there is no NaN. NaN will propagate by resample.
    # Interpolate according to timestamps. Cannot handle boundary. Do not interpolate NaN longer than 10.
    X_dataframe.interpolate(method='time', inplace=True, limit=n_interp_limit)
    # Drop duplicated records, keep first one.
    X_dataframe.drop_duplicates(keep='first', inplace=True)
    # New added records will be filled with NaN.
    X_dataframe = X_dataframe.resample(resampledt).mean()
    X_dataframe.interpolate(method='time', inplace=True, limit=n_interp_limit)
    X_dataframe = X_dataframe[(X_dataframe.index>=timeStart)\
        &(X_dataframe.index<=timeEnd)]

    return X_dataframe

#############################################################################################################
def preprocess_data(data_dict, data_pickle_dir, **kwargs):
    """This functions will process the downloaded file that was put into a dictionary. 
    Will read different spacecraft datasets first, and process together with 
    their data resolutions."""

    # All temperature are converted to Kelvin.
    # Extract data from dict variable.
    isPlotFilterProcess = True
    isVerbose = False
    isVerboseR = False
    isCheckDataIntegrity = True
    if 'isVerboseR' in kwargs: isVerboseR = kwargs['isVerboseR']
    if 'isPlotFilterProcess' in kwargs: isPlotFilterProcess = kwargs['isPlotFilterProcess']
    MutliResolution = False
    if 'MutliResolution' in kwargs: 
        MutliResolution = kwargs['MutliResolution']
    else: 
        MutliResolution = False
    if isVerbose: print('\nMutliResolution       = {}'.format(MutliResolution))

    timeStart = data_dict['timeRange']['datetimeStart']
    timeEnd = data_dict['timeRange']['datetimeEnd']

    #Truncate datetime, remove miliseconds. Or will be misaligned when resample.
    timeStart = timeStart.replace(microsecond=0)
    timeEnd = timeEnd.replace(microsecond=0)

    timeStart_str = timeStart.strftime('%Y%m%d%H%M%S')
    timeEnd_str = timeEnd.strftime('%Y%m%d%H%M%S')
    if isVerbose:
        print('\nExtracting data from downloaded files...')
    if data_dict['ID']=='WIND':
        if isVerbose: print('\nSpacecraft ID: WIND')
        if data_dict['WI_H0_MFI'] is not None: 
            B_OriFrame_Epoch, B_OriFrame = data_dict['WI_H0_MFI']['Epoch'], data_dict['WI_H0_MFI']['BGSE']
        else: 
            B_OriFrame_Epoch, B_OriFrame = None, None

        # Plasma bulk properties
        if data_dict['WI_K0_SWE'] is not None: 
            # Solar wind velocity in GSE coordinate.
            SW_Epoch, V_OriFrame = data_dict['WI_K0_SWE']['Epoch'], data_dict['WI_K0_SWE']['V_GSE']
            # Proton number density (#/cc) & thermal speed (km/s)
            Np, Vth = data_dict['WI_K0_SWE']['Np'], data_dict['WI_K0_SWE']['THERMAL_SPD']
            # Convert Tp from thermal speed to Kelvin. Thermal speed is in km/s. Vth = sqrt(2KT/M)
            Tp = m_proton * np.square(Vth*1e3) / (2.0*k_Boltzmann)
        else: 
            SW_Epoch, V_OriFrame, Np, TpV, Tp = None, None, None, None, None

        # Electron temperature Te in Kelvin, Ne in #/cc
        if 'WI_H0_SWE' in data_dict:
            Te_Epoch, Te = data_dict['WI_H0_SWE']['Epoch'], data_dict['WI_H0_SWE']['Te']
            Ne_Epoch, Ne = data_dict['WI_H0_SWE']['Epoch'], data_dict['WI_H0_SWE']['el_density']
        elif 'WI_H5_SWE' in data_dict:
            Te_Epoch, Te = data_dict['WI_H5_SWE']['Epoch'], data_dict['WI_H5_SWE']['T_elec']
            Ne_Epoch, Ne = data_dict['WI_H5_SWE']['Epoch'], data_dict['WI_H5_SWE']['N_elec']
        else:
            Te_Epoch, Te, Ne_Epoch, Ne = None, None, None, None
            # print('\nWarning: No electron data!')

        # Alpha data
        if data_dict['WI_H1_SWE'] is not None:
            Na_Epoch, Na = data_dict['WI_H1_SWE']['Epoch'], data_dict['WI_H1_SWE']['Alpha_Na_nonlin']
            Ta_Epoch, Vtha = data_dict['WI_H1_SWE']['Epoch'], data_dict['WI_H1_SWE']['Alpha_W_nonlin']
            Ta = m_alpha * np.square(Vtha*1e3) / (2.0*k_Boltzmann)
        else:
            Na_Epoch, Na, Ta_Epoch, Ta, Vtha = None, None, None, None, None

        # Spacecraft position
        if data_dict['WI_K0_SWE'] is not None: 
            RD_Epoch, RD2Earth = data_dict['WI_K0_SWE']['Epoch'], data_dict['WI_K0_SWE']['SC_pos_R']
            # Radial distance to center of Earth in km
            RD = 1 - RD2Earth/149597870.691
        else: 
            RD_Epoch, RD2Earth, RD = None, None, None

        alphaRatio = None
        # Process missing value. missing value = -9.9999998e+30.
        if B_OriFrame is not None: B_OriFrame[(abs(B_OriFrame) > 10000)] = np.nan
        if Np is not None: Np[(Np < -1e+10) | (Np == math.inf)] = np.nan
        if V_OriFrame is not None: V_OriFrame[(abs(V_OriFrame) > 2500) | (V_OriFrame == math.inf)] = np.nan
        if Tp is not None: Tp[(Tp < -1e+10) | (Tp == math.inf)] = np.nan
        if Na is not None: Na[(Na < 0) | (Na > 1e+4)] = np.nan
        if Te is not None: Te[(Te < 0.0) | (Te == math.inf)] = np.nan
        if Ne is not None: Ne[Ne < 0.0] = np.nan
        if Ta is not None: Ta[(Ta < 0.0) | (Ta > 1e12)] = np.nan

    if data_dict['ID']=='ACE':
        if isVerbose: print('\nSpacecraft ID: ACE')
        if data_dict['AC_H0_MFI'] is not None:
            B_OriFrame_Epoch, B_OriFrame = data_dict['AC_H0_MFI']['Epoch'], data_dict['AC_H0_MFI']['BGSEc']
        else:
            B_OriFrame_Epoch, B_OriFrame = None, None

        if data_dict['AC_H0_SWE'] is not None:
            # Np in #/cc, Tp in Kelvin
            SW_Epoch, V_OriFrame = data_dict['AC_H0_SWE']['Epoch'], data_dict['AC_H0_SWE']['V_GSE']
            Np, Tp, alphaRatio = data_dict['AC_H0_SWE']['Np'], data_dict['AC_H0_SWE']['Tpr'], data_dict['AC_H0_SWE']['alpha_ratio']
        else:
            SW_Epoch, V_OriFrame, Np, Tp, alphaRatio = None, None, None, None, None

        if data_dict['AC_OR_SSC'] is not None:
            RD_Epoch, RD2Earth = data_dict['AC_OR_SSC']['Epoch'], data_dict['AC_OR_SSC']['RADIUS']
            RD = 1 - RD2Earth/149597870.691
        else:
            RD_Epoch, RD2Earth, RD = None, None, None
        
        Ne_Epoch, Ne, Te_Epoch, Te, Na_Epoch, Na, Ta_Epoch, Ta = None, None, None, None, None, None, None, None
        # Process missing value. missing value = -9.9999998e+30.
        # print('Processing missing value...')
        if B_OriFrame is not None: B_OriFrame[(abs(B_OriFrame) > 1000) | (B_OriFrame == -1e31)] = np.nan # B field.
        if Np is not None: Np[Np < -1e+10] = np.nan # Proton number density.
        if V_OriFrame is not None: V_OriFrame[abs(V_OriFrame) > 2500] = np.nan # Solar wind speed.
        if Tp is not None: Tp[Tp < -1e+10] = np.nan # Proton temperature, radial component of T tensor.
        if alphaRatio is not None: alphaRatio[alphaRatio < -1e+10] = np.nan # Na/Np.

    if data_dict['ID']=='ULYSSES':
        if isVerbose: print('\nSpacecraft ID: ULYSSES')
        if data_dict['UY_1MIN_VHM'] is not None:
            B_OriFrame_Epoch, B_OriFrame = data_dict['UY_1MIN_VHM']['Epoch'], data_dict['UY_1MIN_VHM']['B_RTN']
        else:
            B_OriFrame_Epoch, B_OriFrame = None, None
        
        if data_dict['UY_M0_BAI'] is not None:
            SW_Epoch, V_OriFrame = data_dict['UY_M0_BAI']['Epoch'], data_dict['UY_M0_BAI']['Velocity']
            Tlarge, Tp = data_dict['UY_M0_BAI']['Temperature'][:,0], data_dict['UY_M0_BAI']['Temperature'][:,0]
            Np, Na = data_dict['UY_M0_BAI']['Density'][:,0], data_dict['UY_M0_BAI']['Density'][:,1]
            Na_Epoch = data_dict['UY_M0_BAI']['Epoch']
        else:
            SW_Epoch, Na_Epoch, V_OriFrame, Np, Tp, alphaRatio = None, None, None, None, None, None
        if data_dict['UY_SC_POS'] is not None:
            RD_Epoch, RD, Lat = data_dict['UY_SC_POS']['Epoch'], data_dict['UY_SC_POS']['heliocentricDistance'], data_dict['UY_SC_POS']['heliographicLatitude']
        else:
            RD_Epoch, RD = None, None

        Ne_Epoch, Ne, Te_Epoch, Te, Ta_Epoch, Ta, alphaRatio = None, None, None, None, None, None, None
        # Process missing value. missing value = -9.9999998e+30.
        # print('Processing missing value...')
        if B_OriFrame is not None: B_OriFrame[abs(B_OriFrame) > 1000] = np.nan # B field.
        if Np is not None: Np[Np < -1e+10] = np.nan # Proton number density.
        if V_OriFrame is not None: V_OriFrame[abs(V_OriFrame) > 2500] = np.nan # Solar wind speed.
        if Tp is not None: Tp[Tp < -1e+10] = np.nan # Proton temperature, T-small
        if Na is not None: Na[Na < -1e+10] = np.nan

    if data_dict['ID']=='PSP':
        if isVerbose: print('\nSpacecraft ID: PSP')
        if data_dict['PSP_FLD_L2_MAG_RTN'] is not None:
            B_OriFrame_Epoch, B_OriFrame = data_dict['PSP_FLD_L2_MAG_RTN']['epoch_mag_RTN'], data_dict['PSP_FLD_L2_MAG_RTN']['psp_fld_l2_mag_RTN']
        else:
            B_OriFrame_Epoch, B_OriFrame = None, None
        
        if data_dict['PSP_SWP_SPC_L3I'] is not None:
            SW_Epoch, V_OriFrame_noVsc = data_dict['PSP_SWP_SPC_L3I']['Epoch'], data_dict['PSP_SWP_SPC_L3I']['vp_moment_RTN_gd']
            Np, Vth = data_dict['PSP_SWP_SPC_L3I']['np_moment_gd'], data_dict['PSP_SWP_SPC_L3I']['wp_moment_gd']
            Tp = m_proton * np.square(Vth*1e3) / (2.0*k_Boltzmann)
            VHCI_Epoch, VHCI = data_dict['PSP_SWP_SPC_L3I']['Epoch'], data_dict['PSP_SWP_SPC_L3I']['sc_vel_HCI']
        else:
            SW_Epoch, V_OriFrame_noVsc, Np, Vth, Tp, VHCI_Epoch, VHCI = None, None, None, None, None, None, None

        if data_dict['PSP_SWP_SPC_L3I'] is not None:
            RD_Epoch, RD_vector = data_dict['PSP_SWP_SPC_L3I']['Epoch'], data_dict['PSP_SWP_SPC_L3I']['sc_pos_HCI']
            RD = np.sqrt(np.square(RD_vector).sum(axis=1))/149597870.691
        else:
            RD_Epoch, RD_vector, RD = None, None, None

        if data_dict['PSP_SWP_SPI_SF0A_L3_MOM'] is not None:
            # Ta is in eV
            Na_Epoch, Na = data_dict['PSP_SWP_SPI_SF0A_L3_MOM']['Epoch'], data_dict['PSP_SWP_SPI_SF0A_L3_MOM']['DENS']
            Ta_Epoch, Ta_eV = data_dict['PSP_SWP_SPI_SF0A_L3_MOM']['Epoch'], data_dict['PSP_SWP_SPI_SF0A_L3_MOM']['TEMP']
            Ta = Ta_eV * 11604.5250061598
        else:
            Na_Epoch, Na, Ta_Epoch, Ta_eV, Ta = None, None, None, None, None

        if data_dict['PSP_FLD_L3_SQTN_RFS_V1V2'] is not None:
            # Te is in eV
            Ne_Epoch, Ne = data_dict['PSP_FLD_L3_SQTN_RFS_V1V2']['Epoch'], data_dict['PSP_FLD_L3_SQTN_RFS_V1V2']['electron_density']
            Te_Epoch, Te_eV = data_dict['PSP_FLD_L3_SQTN_RFS_V1V2']['Epoch'], data_dict['PSP_FLD_L3_SQTN_RFS_V1V2']['electron_core_temperature']
            Te = Te_eV * 11604.5250061598
        else:
            Ne_Epoch, Ne, Te_Epoch, Te_eV, Te = None, None, None, None, None

        alphaRatio = None
        # Process missing value. missing value = -9.9999998e+30.
        # print('Processing missing value...')
        if B_OriFrame is not None: B_OriFrame[abs(B_OriFrame) > 10000] = np.nan # B field.
        if Np is not None: Np[Np < 0.0] = np.nan # Proton number density.
        if V_OriFrame_noVsc is not None: V_OriFrame_noVsc[abs(V_OriFrame_noVsc) > 10000] = np.nan # Solar wind speed.
        if VHCI is not None: VHCI[VHCI == -1e31] = np.nan
        if Tp is not None: Tp[Tp < -1e+10] = np.nan
        if Te is not None: Te[Te < -1e+10] = np.nan
        if Ne is not None: Ne[Ne < -1e+10] = np.nan
        if Ta is not None: Ta[Ta < 100] = np.nan
        if Na is not None: Na[Na < -1e+10] = np.nan

        # Take spacecraft velocity into account
        V = np.sqrt(np.square(V_OriFrame_noVsc).sum(axis=1))
        V_HCI = np.sqrt(np.square(VHCI).sum(axis=1))
        HCI = np.sqrt(np.square(RD_vector).sum(axis=1))
        VRTN_temp = np.zeros((len(V),3))
        R_unitVector = np.array([RD_vector[:,0]/HCI,RD_vector[:,1]/HCI,RD_vector[:,2]/HCI])
        # np.divide
        Z_unitVector = np.mat([[0.0, 0.0, 1.0]])
        T_unitVector = np.zeros((len(V),3))
        N_unitVector = np.zeros((len(V),3))
        
        i = 0
        while i <= len(V)-1:
            T_unitVector[i] = formRighHandFrame(R_unitVector[:,i], Z_unitVector)
            N_unitVector[i] = formRighHandFrame(T_unitVector[i],R_unitVector[:,i])
            i = i + 1;

        R_unitVector = R_unitVector.transpose()
        matrix_transToRTN = np.array([R_unitVector, T_unitVector, N_unitVector]).T

        i = 0
        while i <= len(V)-1:
            VRTN_temp[i] = VHCI[i,:].dot(matrix_transToRTN[:,i])
            i = i + 1;
        
        V_OriFrame = V_OriFrame_noVsc - VRTN_temp

    if data_dict['ID']=='SOLARORBITER':
        if isVerbose: print('\nSpacecraft ID: SOLAR ORBITER')
        if data_dict['SOLO_L2_MAG_RTN_NORMAL'] is not None:
            B_OriFrame_Epoch, B_OriFrame = data_dict['SOLO_L2_MAG_RTN_NORMAL']['EPOCH'], data_dict['SOLO_L2_MAG_RTN_NORMAL']['B_RTN']
        else:
            B_OriFrame_Epoch, B_OriFrame = None, None
        
        if data_dict['SOLO_L2_SWA_PAS_GRND_MOM'] is not None:
            SW_Epoch, V_OriFrame = data_dict['SOLO_L2_SWA_PAS_GRND_MOM']['Epoch'], data_dict['SOLO_L2_SWA_PAS_GRND_MOM']['V_RTN']
            Np, Tp_eV = data_dict['SOLO_L2_SWA_PAS_GRND_MOM']['N'], data_dict['SOLO_L2_SWA_PAS_GRND_MOM']['T']
            Tp = Tp_eV * 11604.5250061598
        else:
            SW_Epoch, V_OriFrame, Np, Tp_eV, Tp = None, None, None, None, None, None

        if data_dict['SOLO_SC_POS'] is not None:
            RD_Epoch, RD = data_dict['SOLO_SC_POS']['Epoch'], data_dict['SOLO_SC_POS']['radialDistance']
        else:
            RD_Epoch, RD = None, None

        Ne_Epoch, Ne, Te_Epoch, Te, Na_Epoch, Na, Ta_Epoch, Ta, alphaRatio = None, None, None, None, None, None, None, None, None
        # Process missing value. missing value = -9.9999998e+30.
        # print('Processing missing value...')
        if B_OriFrame is not None: B_OriFrame[abs(B_OriFrame) > 10000] = np.nan # B field.
        if Np is not None: Np[Np < 0.0] = np.nan # Proton number density.
        if V_OriFrame is not None: V_OriFrame[abs(V_OriFrame) > 2500] = np.nan # Solar wind speed.
        if Tp is not None: Tp[Tp < 0.0] = np.nan # Proton temperature, radial component of T tensor.

    # print('Putting Data into DataFrame...')
    if B_OriFrame_Epoch is None: return None
    else: B_OriFrame_DataFrame = pd.DataFrame(B_OriFrame, index = B_OriFrame_Epoch, columns = ['B0', 'B1', 'B2'])
      
    if SW_Epoch is not None:
        V_OriFrame_DataFrame = pd.DataFrame(V_OriFrame, index = SW_Epoch, columns = ['V0', 'V1', 'V2'])
        Np_DataFrame = pd.DataFrame(Np, index = SW_Epoch, columns = ['Np'])
        Tp_DataFrame = pd.DataFrame(Tp, index = SW_Epoch, columns = ['Tp']) 
    else:
        V_OriFrame_DataFrame = pd.DataFrame(None, columns = ['Vx', 'Vy', 'Vz'])
        Np_DataFrame = pd.DataFrame(None, columns = ['Np'])
        Tp_DataFrame = pd.DataFrame(None, columns = ['Tp'])
        RD_DataFrame = pd.DataFrame(None, columns = ['RD'])
    
    if V_OriFrame_DataFrame is None:
        print('\nError: the solar wind velocity data are unavailable during the current interval!')
        print('Please adjust the interval.')
        exit()

    if RD_Epoch is not None:
        RD_DataFrame = pd.DataFrame(RD, index = RD_Epoch, columns = ['RD'])
    else:
        RD_DataFrame = pd.DataFrame(None, columns = ['RD'])

    if 'Te' in locals(): Te_DataFrame = pd.DataFrame(Te, index = Te_Epoch, columns = ['Te'])
    if 'Ne' in locals(): Ne_DataFrame = pd.DataFrame(Ne, index = Ne_Epoch, columns = ['Ne'])
    if 'Ta' in locals(): Ta_DataFrame = pd.DataFrame(Ta, index = Ta_Epoch, columns = ['Ta'])
    if 'Na' in locals(): Na_DataFrame = pd.DataFrame(Na, index = Na_Epoch, columns = ['Na'])
    if data_dict['ID'] == 'ACE': 
        alphaRatio_DataFrame = pd.DataFrame(alphaRatio, index = SW_Epoch, columns = ['alphaRatio'])
    else: 
        alphaRatio_DataFrame = pd.DataFrame(None, columns = ['alphaRatio'])
    if data_dict['ID'] == 'PSP': 
        V_OriFrame_noVsc_DataFrame = pd.DataFrame(V_OriFrame_noVsc, index = SW_Epoch, columns = ['VR_noVsc', 'VT_noVsc', 'VN_noVsc'])

    Ne_DataFrame.dropna(inplace = True)
    Te_DataFrame.dropna(inplace = True)
    Ta_DataFrame.dropna(inplace = True)
    Na_DataFrame.dropna(inplace = True)

    # Trim data. Some times cdas API will download wrong time range.
    if B_OriFrame_Epoch is not None:
        B_OriFrame_DataFrame = trim_drop_duplicates(B_OriFrame_DataFrame, timeStart, timeEnd, drop=False)
    if SW_Epoch is not None:
        V_OriFrame_DataFrame = trim_drop_duplicates(V_OriFrame_DataFrame, timeStart, timeEnd, drop=False)
        Np_DataFrame = trim_drop_duplicates(Np_DataFrame, timeStart, timeEnd, drop=False)
        Tp_DataFrame = trim_drop_duplicates(Tp_DataFrame, timeStart, timeEnd, drop=False)
    if RD is not None:
        RD_DataFrame = trim_drop_duplicates(RD_DataFrame, timeStart, timeEnd, drop=False)
    if alphaRatio is not None:
        alphaRatio_DataFrame = trim_drop_duplicates(alphaRatio_DataFrame, timeStart, timeEnd, drop=True)
    if Na_Epoch is not None:
        Na_DataFrame = trim_drop_duplicates(Na_DataFrame, timeStart, timeEnd, drop=True)
    if Ne_Epoch is not None:
        Ne_DataFrame = trim_drop_duplicates(Ne_DataFrame, timeStart, timeEnd, drop=True)
    if Ta_Epoch is not None:
        Ta_DataFrame = trim_drop_duplicates(Ta_DataFrame, timeStart, timeEnd, drop=True)
    if Te_Epoch is not None:
        Te_DataFrame = trim_drop_duplicates(Te_DataFrame, timeStart, timeEnd, drop=True)
    if data_dict['ID'] == 'PSP': 
        V_OriFrame_noVsc_DataFrame = trim_drop_duplicates(V_OriFrame_noVsc_DataFrame, timeStart, timeEnd, drop=False)
    
    
    ##################################################################################################
    
    # Processing data...

    # ====================================== Process B_OriFrame  ======================================
    # print('\nProcessing B...')
    # Keep original data.
    B_OriFrame_DataFrame0 = B_OriFrame_DataFrame.copy(deep=True)
    # #print('B_OriFrame_DataFrame.shape = {}'.format(B_OriFrame_DataFrame.shape))
    # Apply Butterworth filter.
    B_OriFrame_DataFrame = butterworth_filter_3(B_OriFrame_DataFrame, isVerbose, tag="B")

    # ===================================== Process V_OriFrame  ======================================
    if not V_OriFrame_DataFrame.empty:
        # print('\nProcessing V...')
        # Keep original data.
        V_OriFrame_DataFrame0 = V_OriFrame_DataFrame.copy(deep=True)
        # Remove all data which fall outside three standard deviations.
        V_OriFrame_DataFrame = butterworth_filter_3(V_OriFrame_DataFrame, isVerbose, tag="V1")
        # Apply Butterworth filter two times.
        V_OriFrame_DataFrame = butterworth_filter_3(V_OriFrame_DataFrame, isVerbose, tag="V")
  
    # ========================================  Process Np   =========================================
    if not Np_DataFrame.empty:
        # print('\nProcessing Np...')
        # Keep original data.
        Np_DataFrame0 = Np_DataFrame.copy(deep=True)
        # print('Np_DataFrame.shape = {}'.format(Np_DataFrame.shape))
        Np_DataFrame = butterworth_filter(Np_DataFrame, isVerbose, tag='Np')

    # ========================================= Process Tp   =========================================
    if not Tp_DataFrame.empty:
        # print('\nProcessing Tp...')
        # Keep original data.
        Tp_DataFrame0 = Tp_DataFrame.copy(deep=True)
        # print('Tp_DataFrame.shape = {}'.format(Tp_DataFrame.shape))
        Tp_DataFrame = butterworth_filter(Tp_DataFrame, isVerbose, tag='Tp')

    # ========================================= Process Ne  =========================================            
    if not Ne_DataFrame.empty:
        # print('\nProcessing Ne...')
        # Keep originel data.
        Ne_DataFrame0 = Ne_DataFrame.copy(deep=True)
        # print('Ne_DataFrame.shape = {}'.format(Ne_DataFrame.shape))
        Ne_DataFrame = butterworth_filter(Ne_DataFrame, isVerbose, tag='Ne')
        
    # ========================================= Process Te  =========================================
    if not Te_DataFrame.empty:
        # print('\nProcessing Te...')
        # Keep original data.
        Te_DataFrame0 = Te_DataFrame.copy(deep=True)
        # print('Te_DataFrame.shape = {}'.format(Te_DataFrame.shape))
        Te_DataFrame = butterworth_filter(Te_DataFrame, isVerbose, tag='Te')
    
    # ========================================= Process Na   =========================================
    if not Na_DataFrame.empty:
        # print('\nProcessing Na...')
        # Keep original data.
        Na_DataFrame0 = Na_DataFrame.copy(deep=True)
        # print('Na_DataFrame.shape = {}'.format(Na_DataFrame.shape))
        Na_DataFrame = butterworth_filter(Na_DataFrame, isVerbose, tag='Na')
        
    # ========================================= Process Ta  =========================================
    if not Ta_DataFrame.empty:
        # print('\nProcessing Ta...')
        # Keep original data.
        Ta_DataFrame0 = Ta_DataFrame.copy(deep=True)
        # print('Ta_DataFrame.shape = {}'.format(Ta_DataFrame.shape)
        Ta_DataFrame = butterworth_filter(Ta_DataFrame, isVerbose, tag='Ta')
                
    # ===================================== Process alphaRatio  =====================================
    if not alphaRatio_DataFrame.empty:
        # print('\nProcessing alphaRatio...')
        # Keep original data.
        alphaRatio_DataFrame0 = alphaRatio_DataFrame.copy(deep=True)
        # print('alphaRatio_DataFrame.shape = {}'.format(alphaRatio_DataFrame.shape))
        alphaRatio_DataFrame = butterworth_filter(alphaRatio_DataFrame, isVerbose, tag='alphaRatio')
        
    if isPlotFilterProcess:
        fig_line_width = 1
        fig_ylabel_fontsize = 9
        fig_xtick_fontsize = 8
        fig_ytick_fontsize = 8
        fig_legend_size = 5
        fig_formatter = mdates.DateFormatter('%m/%d %H:%M')
        fig,ax = plt.subplots(12,1, sharex=True,figsize=(12, 16))
        
        B0_plot, B1_plot, B2_plot, V0_plot, V1_plot, V2_plot = ax[0], ax[1], ax[2], ax[3], ax[4], ax[5]
        Np_plot, Tp_plot, Ne_plot, Te_plot, Na_plot, Ta_plot = ax[6], ax[7], ax[8], ax[9], ax[10], ax[11]

        # Plotting B0 filter process.
        B0_plot.plot(B_OriFrame_DataFrame0.index, B_OriFrame_DataFrame0['B0'],\
        color = '#252525', linewidth=fig_line_width,label='Original') # Original data.
        B0_plot.plot(B_OriFrame_DataFrame.index, B_OriFrame_DataFrame['B0'],\
        color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
        B0_plot.set_ylabel('B0 (nT)', fontsize=fig_ylabel_fontsize)
        B0_plot.legend(loc='best',prop={'size':fig_legend_size})
        B0_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
        
        # Plotting B1 filter process.
        B1_plot.plot(B_OriFrame_DataFrame0.index, B_OriFrame_DataFrame0['B1'],\
        color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
        B1_plot.plot(B_OriFrame_DataFrame.index, B_OriFrame_DataFrame['B1'],\
        color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
        B1_plot.set_ylabel('B1 (nT)', fontsize=fig_ylabel_fontsize)
        B1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
        B1_plot.legend(loc='best',prop={'size':fig_legend_size})
        
        # Plotting B2 filter process.
        B2_plot.plot(B_OriFrame_DataFrame0.index, B_OriFrame_DataFrame0['B2'],\
        color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
        B2_plot.plot(B_OriFrame_DataFrame.index, B_OriFrame_DataFrame['B2'],\
        color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
        B2_plot.set_ylabel('B2 (nT)', fontsize=fig_ylabel_fontsize)
        B2_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
        B2_plot.legend(loc='best',prop={'size':fig_legend_size})
        
        # Plotting V0 filter process.
        V0_plot.plot(V_OriFrame_DataFrame0.index, V_OriFrame_DataFrame0['V0'],\
        color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
        V0_plot.plot(V_OriFrame_DataFrame.index, V_OriFrame_DataFrame['V0'],\
        color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
        V0_plot.set_ylabel('V0 (km/s)', fontsize=fig_ylabel_fontsize)
        V0_plot.legend(loc='best',prop={'size':fig_legend_size})
        V0_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
        
        # Plotting V1 filter process.
        V1_plot.plot(V_OriFrame_DataFrame0.index, V_OriFrame_DataFrame0['V1'],\
        color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
        V1_plot.plot(V_OriFrame_DataFrame.index, V_OriFrame_DataFrame['V1'],\
        color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
        V1_plot.set_ylabel('V1 (km/s)', fontsize=fig_ylabel_fontsize)
        V1_plot.legend(loc='best',prop={'size':fig_legend_size})
        V1_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
        
        # Plotting V2 filter process.
        V2_plot.plot(V_OriFrame_DataFrame0.index, V_OriFrame_DataFrame0['V2'],\
        color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
        V2_plot.plot(V_OriFrame_DataFrame.index, V_OriFrame_DataFrame['V2'],\
        color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
        V2_plot.set_ylabel('V2 (km/s)', fontsize=fig_ylabel_fontsize)
        V2_plot.legend(loc='best',prop={'size':fig_legend_size})
        V2_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
        V2_plot.xaxis.set_major_formatter(fig_formatter)

        # Plotting Np filter process.
        Np_plot.plot(Np_DataFrame0.index, Np_DataFrame0['Np'],\
        color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
        Np_plot.plot(Np_DataFrame.index, Np_DataFrame['Np'],\
        color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
        Np_plot.set_ylabel('Np (#/cc)', fontsize=fig_ylabel_fontsize)
        Np_plot.legend(loc='best',prop={'size':fig_legend_size})
        Np_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
        Np_plot.xaxis.set_major_formatter(fig_formatter)

        # Plotting Tp filter process.
        Tp_plot.plot(Tp_DataFrame0.index, Tp_DataFrame0['Tp'],\
        color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
        Tp_plot.plot(Tp_DataFrame.index, Tp_DataFrame['Tp'],\
        color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
        Tp_plot.set_ylabel('Tp (K)', fontsize=fig_ylabel_fontsize)
        Tp_plot.legend(loc='best',prop={'size':fig_legend_size})
        Tp_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
        Tp_plot.xaxis.set_major_formatter(fig_formatter)
        
        # Plotting Ne filter process.
        if not Ne_DataFrame.empty:
            Ne_plot.plot(Ne_DataFrame0.index, Ne_DataFrame0['Ne'],\
            color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
            Ne_plot.plot(Ne_DataFrame.index, Ne_DataFrame['Ne'],\
            color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
            Ne_plot.set_ylabel('Ne (#/cc)', fontsize=fig_ylabel_fontsize)
            Ne_plot.legend(loc='best',prop={'size':fig_legend_size})
            Ne_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Ne_plot.xaxis.set_major_formatter(fig_formatter)
        else:
            Ne_plot.set_ylabel('Ne\n(unavailable)', fontsize=fig_ylabel_fontsize)
            Tp_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)

        # Plotting Ne filter process.
        if not Te_DataFrame.empty:
            Te_plot.plot(Te_DataFrame0.index, Te_DataFrame0['Te'],\
            color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
            Te_plot.plot(Te_DataFrame.index, Te_DataFrame['Te'],\
            color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
            Te_plot.set_ylabel('Te (K)', fontsize=fig_ylabel_fontsize)
            Te_plot.legend(loc='best',prop={'size':fig_legend_size})
            Te_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Te_plot.xaxis.set_major_formatter(fig_formatter)
        else:
            Te_plot.set_ylabel('Te\n(unavailable)', fontsize=fig_ylabel_fontsize)
            Tp_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)

        # Plotting Na filter process.
        if not Na_DataFrame.empty:
            Na_plot.plot(Na_DataFrame0.index, Na_DataFrame0['Na'],\
            color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
            Na_plot.plot(Na_DataFrame.index, Na_DataFrame['Na'],\
            color = 'blue', linewidth=fig_line_width, linestyle='dashed', label='Processed') # Filtered data.
            Na_plot.set_ylabel('Na (#/cc)', fontsize=fig_ylabel_fontsize)
            Na_plot.legend(loc='best',prop={'size':fig_legend_size})
            Na_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Na_plot.xaxis.set_major_formatter(fig_formatter)
        else:
            Na_plot.set_ylabel('Na\n(unavailable)', fontsize=fig_ylabel_fontsize)
            Tp_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)

        # Plotting Ta filter process.
        if not Ta_DataFrame.empty:
            Ta_plot.plot(Ta_DataFrame0.index, Ta_DataFrame0['Ta'],\
            color = '#252525', linewidth=fig_line_width, label='Original') # Original data.
            Ta_plot.plot(Ta_DataFrame.index, Ta_DataFrame['Ta'],\
            color = 'blue', linewidth=fig_line_width, linestyle='dotted', label='Processed') # FilTared data.
            Ta_plot.set_ylabel('Ta (K)', fontsize=fig_ylabel_fontsize)
            Ta_plot.legend(loc='best',prop={'size':fig_legend_size})
            Ta_plot.tick_params(axis='y', which='major', labelsize=fig_ytick_fontsize)
            Ta_plot.xaxis.set_major_formatter(fig_formatter)
        else:    
            Ta_plot.set_ylabel('Ta\n(unavailable)', fontsize=fig_ylabel_fontsize)
            Tp_plot.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)        

        # Save plot.
        fig.savefig(data_pickle_dir + '/filter_process_' + timeStart_str + '_' + timeEnd_str + '.png', format='png', dpi=300, bbox_inches='tight')

    # Setting parameters to prepare resample
    # Different datasets have different resolution and data qualities
    if (data_dict['ID'] == 'ACE') or (data_dict['ID'] == 'WIND'): 
        n_interp_limit = 5
        resampledt ='1min'
    if data_dict['ID']=='ULYSSES':
        n_interp_limit = 5
        resampledt ='4min'
    if data_dict['ID']=='PSP': 
        n_interp_limit = 3
        resampledt ='28S'
        if MutliResolution:
            resampledt ='1S'
    if data_dict['ID']=='SOLARORBITER':
        n_interp_limit = 3
        resampledt ='28S'
        if MutliResolution:
            resampledt ='4S'
    
    if isVerbose:
        print('\nResampling data into {} resolution...'.format(resampledt))

    B_OriFrame_DataFrame = process_to_resolution(B_OriFrame_DataFrame, resampledt, 
        n_interp_limit, timeStart, timeEnd, tag='B')
   
    if not V_OriFrame_DataFrame.empty:
        V_OriFrame_DataFrame = process_to_resolution(V_OriFrame_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='V')

    if not Np_DataFrame.empty:
        Np_DataFrame = process_to_resolution(Np_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='Np')

    if not Tp_DataFrame.empty:
        Tp_DataFrame = process_to_resolution(Tp_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='Tp')
  
    if not RD_DataFrame.empty:
        # Spacecraft data is interpolated without limit
        # Some datasets only have 1 hour data
        RD_DataFrame = process_to_resolution(RD_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='RD')

    if not Ne_DataFrame.empty:
        Ne_DataFrame = process_to_resolution(Ne_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='Ne')

    if not Te_DataFrame.empty:
        Te_DataFrame = process_to_resolution(Te_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='Te')

    if not Na_DataFrame.empty:
        Na_DataFrame = process_to_resolution(Na_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='Na')

    if not Ta_DataFrame.empty:
        Ta_DataFrame = process_to_resolution(Ta_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='Ta')

    if not alphaRatio_DataFrame.empty:
        alphaRatio_DataFrame = process_to_resolution(alphaRatio_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='alphaRatio')

    elif not Na_DataFrame.empty:
        NaNp = Na_DataFrame['Na']/Np_DataFrame['Np']
        alphaRatio_DataFrame = pd.DataFrame(NaNp,columns=['alphaRatio'],dtype=float)
    if data_dict['ID'] == 'PSP': 
        V_OriFrame_noVsc_DataFrame = process_to_resolution(V_OriFrame_noVsc_DataFrame, resampledt, 
            n_interp_limit, timeStart, timeEnd, tag='V_OriFrame')

    # Merge all DataFrames into one according to time index.
    if data_dict['ID']=='ACE' or 'WIND':
        timeRangeInMinutes = int((timeEnd - timeStart).total_seconds())//60
        index_datetime = np.asarray([timeStart + timedelta(minutes=x) for x in range(0, timeRangeInMinutes+1)])
    if data_dict['ID']=='ULYSSES':
        timeRangeInMinutes = int((timeEnd - timeStart).total_seconds())//60
        index_datetime = np.asarray([timeStart + timedelta(minutes=x) for x in range(0, timeRangeInMinutes+1, 4)])
    if data_dict['ID']=='PSP':
        timeRange = int((timeEnd - timeStart).total_seconds())//1
        index_datetime = np.asarray([B_OriFrame_DataFrame.index[0] + timedelta(seconds=x) for x in range(0, timeRange, 28)])
        if MutliResolution: # Usually for PSP <1s resolution
            timeRange = int((timeEnd - timeStart).total_seconds())//1
            index_datetime = np.asarray([timeStart + timedelta(seconds=x) for x in range(0, timeRange+1)])
    if data_dict['ID']=='SOLARORBITER':
        timeRange = int((timeEnd - timeStart).total_seconds())//1
        index_datetime = np.asarray([B_OriFrame_DataFrame.index[0] + timedelta(seconds=x) for x in range(0, timeRange, 28)])
        if MutliResolution: # Usually for PSP <1s resolution
            timeRange = int((timeEnd - timeStart).total_seconds())//1
            index_datetime = np.asarray([timeStart + timedelta(seconds=x) for x in range(0, timeRange+1, 4)])

    # Generate empty DataFrame according using index_datetime as index.
    GS_AllData_DataFrame = pd.DataFrame(index=index_datetime)

    GS_AllData_DataFrame = pd.concat([GS_AllData_DataFrame,\
        B_OriFrame_DataFrame, V_OriFrame_DataFrame, Np_DataFrame, Tp_DataFrame,\
        Ne_DataFrame, Te_DataFrame, Ta_DataFrame, Na_DataFrame, alphaRatio_DataFrame, RD_DataFrame], axis=1)
    # Delete dataframes that have all NaNs - null array
    GS_AllData_DataFrame.dropna(axis = 1, how = 'all', inplace = True) 
    
    if data_dict['ID'] == 'PSP': 
        GS_AllData_DataFrame = pd.concat([GS_AllData_DataFrame,V_OriFrame_noVsc_DataFrame], axis=1)

    # Save merged DataFrame into pickle file.
    # print(GS_AllData_DataFrame.index)
    GS_AllData_DataFrame.index = pd.DatetimeIndex(GS_AllData_DataFrame.index)
    if MutliResolution: 
        GS_AllData_DataFrame.to_pickle(data_pickle_dir + '/' + data_dict['ID'] +'_' + timeStart_str + '_' + timeEnd_str + '_preprocessed_high_resltn.p')
        GS_AllData_DataFrame.to_csv(data_pickle_dir + '/' + data_dict['ID'] +'_' + timeStart_str + '_' + timeEnd_str + '_preprocessed_high_resltn.csv')
    else:
        GS_AllData_DataFrame.to_pickle(data_pickle_dir + '/' + data_dict['ID'] +'_' + timeStart_str + '_' + timeEnd_str + '_preprocessed.p')
        GS_AllData_DataFrame.to_csv(data_pickle_dir + '/' + data_dict['ID'] +'_' + timeStart_str + '_' + timeEnd_str + '_preprocessed.csv')
    
    if isCheckDataIntegrity:
        if isVerbose or isVerboseR:
            print('\nChecking the number of NaNs in GS_AllData_DataFrame...')
        len_GS_AllData_DataFrame = len(GS_AllData_DataFrame)
        for key in GS_AllData_DataFrame.keys():
            num_notNaN = GS_AllData_DataFrame[key].isnull().values.sum()
            percent_notNaN = 100.0 - num_notNaN * 100.0 / len_GS_AllData_DataFrame
            if isVerbose or isVerboseR:
                print('The number of NaNs in {} is {}, integrity is {}%'.format(key, num_notNaN, round(percent_notNaN, 2)))
    if isVerbose:       
        print('\nDone.')
    return GS_AllData_DataFrame

#############################################################################################################
def findVHT(BinOrigFrame, VswinOrigFrame):
    """Calculation of the deHoffmann-Teller frame velocity VHT.
    The original nested for-loop were removed due to inefficiency."""

    # B^2 * unit matrix - Bmatleft matrix.T & Bmatright matrix.T * unit matrix >> KN matrix
    # Have to use transpose since initially Bmatleft shape is (3, 16, 3)
    # Array.Transpose.shape = (16, 3, 3) - 16 dimension 3 x 3 matrix
    # [B1^2 0       0]     [B1x B1x B1x]       [[B1x B1y B1z]   [1 0 0]]
    # [0    B1^2    0]  -  [B1y B1y B1y]  dot  [[B1x B1y B1z] * [1 0 0]]
    # [0    0    B1^2]     [B1z B1z B1z]       [[B1x B1y B1z]   [1 0 0]]
    
    # [B1^2 0       0]     [B1x B1x B1x]       [B1x 0    0]   
    # [0    B1^2    0]  -  [B1y B1y B1y]  dot  [0   B1y  0]
    # [0    0    B1^2]     [B1z B1z B1z]       [0   0  B1z]
   
    # [B1^2 0       0]     [B1xB1x B1xB1y B1xB1z]      
    # [0    B1^2    0]  -  [B1yB1x B1yB1y B1yB1z]
    # [0    0    B1^2]     [B1zB1x B1zB1y B1zB1z]
    
    # [B1^2-B1xB1x   -B1xB1y         -B1xB1z]
    # [-B1yB1x       B1^2-B1xB1x     -B1xB1z]
    # [-B1zB1x       -B1zB1y      B1^2-B1xB1x] 

    N = len(BinOrigFrame)
    B_square = np.square(BinOrigFrame).sum(axis=1) 
    KN = np.zeros((N,3,3)) # np.zeros((layer, row, column)). Right most index change first.

    Bmatleft = np.array([[BinOrigFrame['B0'], BinOrigFrame['B1'], BinOrigFrame['B2']],
                  [BinOrigFrame['B0'], BinOrigFrame['B1'], BinOrigFrame['B2']],
                  [BinOrigFrame['B0'], BinOrigFrame['B1'], BinOrigFrame['B2']]])
    Bmatright = np.array([[BinOrigFrame['B0'], BinOrigFrame['B0'], BinOrigFrame['B0']],
                  [BinOrigFrame['B1'], BinOrigFrame['B1'], BinOrigFrame['B1']],
                  [BinOrigFrame['B2'], BinOrigFrame['B2'], BinOrigFrame['B2']]])
    BmatrightUni = [Bmatright.T[i] * np.eye(3) for i in range(N)]
    BsquareUni = np.array([np.eye(3) * B_square[i] for i in range(N)])
    KN = BsquareUni - np.matmul(Bmatleft.T, BmatrightUni)

    K = np.mean(KN, axis=0)
    KVN = np.zeros((N,3))
    Vmat = np.array([VswinOrigFrame['V0'], VswinOrigFrame['V1'], VswinOrigFrame['V2']])
    KVN = np.array([np.matmul(KN[i], Vmat.T[i]) for i in range(N)])
    # Average KVN over N to get KV.
    KV = np.mean(KVN, axis=0)
    VHT = np.dot(np.linalg.inv(K), KV)

    return VHT

#############################################################################################################
def eigenMatrix(matrix_DataFrame, **kwargs):
    # Calculate the eigenvalues and eigenvectors of covariance matrix.
    
    eigenValue, eigenVector = la.eig(matrix_DataFrame) 
    # eigen_arg are eigenvalues, and eigen_vec are eigenvectors.

    # Sort the eigenvalues and arrange eigenvectors by sorted eigenvalues.
    eigenValue_i = np.argsort(eigenValue) # covM_B_eigenValue_i is sorted index of covM_B_eigenValue
    lambda3 = eigenValue[eigenValue_i[0]] # lambda3, minimum variance
    lambda2 = eigenValue[eigenValue_i[1]] # lambda2, intermediate variance.
    lambda1 = eigenValue[eigenValue_i[2]] # lambda1, maximum variance.
    eigenVector3 = pd.DataFrame(eigenVector[:, eigenValue_i[0]], columns=['minVar(lambda3)']) # Eigenvector 3, along minimum variance
    eigenVector2 = pd.DataFrame(eigenVector[:, eigenValue_i[1]], columns=['interVar(lambda2)']) # Eigenvector 2, along intermediate variance.
    eigenVector1 = pd.DataFrame(eigenVector[:, eigenValue_i[2]], columns=['maxVar(lambda1)']) # Eigenvector 1, along maximum variance.
    
    if kwargs['formXYZ']==True:
        # Form an eigenMatrix with the columns:
        # X = minimum variance direction, Y = Maximum variance direction, Z = intermediate variance direction.
        eigenMatrix = pd.concat([eigenVector3, eigenVector1, eigenVector2], axis=1)
        #print(eigenMatrix)
        eigenValues = pd.DataFrame([lambda3, lambda1, lambda2], index=['X1(min)', 'X2(max)', 'X3(inter)'], columns=['eigenValue'])
    else:
        # Form a sorted eigenMatrix using three sorted eigenvectors. Columns are eigenvectors.
        eigenMatrix = pd.concat([eigenVector3, eigenVector2, eigenVector1], axis=1)
        eigenValues = pd.DataFrame([lambda3, lambda2, lambda1], index=['lambda3', 'lambda2', 'lambda1'], columns=['eigenValue'])
    
    eigenVectorMaxVar_lambda1 = (eigenVector[:, eigenValue_i[2]])
    eigenVectorInterVar_lambda2 = (eigenVector[:, eigenValue_i[1]])
    eigenVectorMinVar_lambda3 = (eigenVector[:, eigenValue_i[0]])

    return lambda1, lambda2, lambda3, eigenVectorMaxVar_lambda1, eigenVectorInterVar_lambda2, eigenVectorMinVar_lambda3

#############################################################################################################
def formRighHandFrame(X, Z): 
    """Given two orthnormal vectors(Z and X), find the third vector(Y) 
    to form right-hand side frame.
    Z cross X = Y in right hand frame."""

    X = np.array(X)
    Z = np.array(Z)
    Y = np.cross(Z, X)
    Y = Y/(la.norm(Y)) # Normalize.
    return Y

################################################################################################################
def findXaxis(Z, V):
    """Find X axis according to Z axis and V. 
    The X axis is the projection of V on the plane perpendicular to Z axis."""

    Z = np.array(Z)
    V = np.array(V)
    # Both Z and V are unit vector representing the directions. They are numpy 1-D arrays.
    z1 = Z[0]; z2 = Z[1]; z3 = Z[2]; v1 = V[0]; v2 = V[1]; v3 = V[2]
    # V, Z, and X must satisfy two conditions. 1)The are co-plane. 2)X is perpendicular to Z. These two conditions
    # lead to two equations with three unknow. We can solve for x1, x2, and x3, in which x1 is arbitrary. Let x1
    # equals to 1, then normalize X.
    # 1) co-plane : (Z cross V) dot X = 0
    # 2) Z perpendicular to X : Z dot X = 0
    x1 = 1.0 # Arbitray.
    x2 = -((x1*(v2*z1*z1 - v1*z1*z2 - v3*z2*z3 + v2*z3*z3))/(v2*z1*z2 - v1*z2*z2 + v3*z1*z3 - v1*z3*z3))
    x3 = -((x1*(v3*z1*z1 + v3*z2*z2 - v1*z1*z3 - v2*z2*z3))/(v2*z1*z2 - v1*z2*z2 + v3*z1*z3 - v1*z3*z3))
    # Normalization.
    X = np.array([float(x1), float(x2), float(x3)])
    X = X/(la.norm(X))
    if X.dot(V) < 0:
        X = - X
    return X

################################################################################################################
def turningPoints(array):
    # Find how many turning points in an array.

    array = np.array(array)
    dx = np.diff(array)
    dx = dx[dx != 0] # if don't remove duplicate points, will miss the turning points with duplicate values.
    return np.sum(dx[1:] * dx[:-1] < 0)

################################################################################################################
def angle2matrix(theta_deg, phi_deg, VHT_inGSE):
    # To convert (theta, phi) into a matrix 
    # Usage: B_inFR = B_in_OriFrame.dot(matrix_transToFluxRopeFrame)

    factor_deg2rad = np.pi/180.0 # Convert degree to rad.
    # Direction cosines:
    # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
    # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
    # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
    # Use direction cosines to construct a unit vector.
    theta_rad = factor_deg2rad * theta_deg
    phi_rad   = factor_deg2rad * phi_deg
    # Form new Z_unitVector according to direction cosines.
    Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
    # Find X axis from Z axis and -VHT.
    X_unitVector = findXaxis(Z_unitVector, -VHT_inGSE)
    # Find the Y axis to form a right-handed coordinater with X and Z.
    Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)
    # Project B_inGSE into FluxRope Frame.
    matrix_transToFluxRopeFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
    return matrix_transToFluxRopeFrame

################################################################################################################
def directionVector2angle(V):
    # Covert a vector to angles.

    Z = np.array([0,0,1])
    X = np.array([1,0,0])
    cos_theta = np.dot(V,Z)/la.norm(V)/la.norm(Z)
    #print('cos_theta = {}'.format(cos_theta))
    V_cast2XY = np.array([V[0], V[1], 0])
    cos_phi = np.dot(V_cast2XY,X)/la.norm(V_cast2XY)/la.norm(X)
    #print('cos_phi = {}'.format(cos_phi))
    theta_deg = np.arccos(np.clip(cos_theta, -1, 1))*180/np.pi
    phi_deg = np.arccos(np.clip(cos_phi, -1, 1))*180/np.pi
    if V[1]<0:
        phi_deg = 360 - phi_deg
    return (theta_deg, phi_deg)
    
################################################################################################################
def walenTest(VA, V_remaining):
    """Find the correlation coefficient and slop between the 
    remainning velocity and Alfven speed. This function return the component-by-component 
    correlation coefficient and slop of the plasma velocities
    and the Alfven velocities."""

    # V_remaining reshaped time series of solar wind velocity. In km/s.
    # VA is the reshaped time series of Alfven wave. In km/s.
    # Make sure the input data is numpy.array.
    # Convert input to numpy array.
    V_remaining = np.array(V_remaining)
    VA = np.array(VA)
    mask = ~np.isnan(VA) & ~np.isnan(V_remaining)
    if mask.sum()>=5:
        # slope, intercept, r_value, p_value, std_err = stats.linregress(A,B)
        # scipy.stats.linregress(x, y=None). Put VA on X-axis, V_remaining on Y-axis.
        slope, intercept, r_value, p_value, std_err = stats.linregress(VA[mask], V_remaining[mask])
        # Return a numpy array.
        return slope, intercept, r_value
    else:
        return np.nan, np.nan, np.nan

################################################################################################################
def searchFluxRopeInWindow(B_DataFrame, VHT, n_theta_grid, minDuration, dt, flag_smoothA, Np_DataFrame, Tp_DataFrame, Vsw_DataFrame, Ne_DataFrame, Te_DataFrame, includeNe, includeTe):
    """Loop for all directions to calculate residue, 
    return the smallest residue and corresponding direction."""

    #t0 = datetime.now()
    print('{} - [{}~{} points] searching: ({} ~ {})'.format(time.ctime(), minDuration, 
        len(B_DataFrame), B_DataFrame.index[0], B_DataFrame.index[-1]))
    #t1 = datetime.now()
    #print((t1-t0).total_seconds())
    
    # Initialization.
    # Caution: the type of return value will be different if the initial data is updated. If updated, timeRange_temp will become to tuple, plotData_dict_temp will becomes to dict, et, al.
    time_start_temp = np.nan
    time_end_temp = np.nan
    time_turn_temp = np.nan
    turnPointOnTop_temp = np.nan
    Residue_diff_temp = np.inf
    Residue_fit_temp = np.inf
    duration_temp = np.nan
    theta_temp = np.nan
    phi_temp = np.nan

    time_start, time_end, time_turn, turnPointOnTop, \
    Residue_diff, Residue_fit, duration = getResidueForCurrentAxial(0, 0, 
        minDuration, B_DataFrame, VHT, dt, flag_smoothA, Np_DataFrame, Tp_DataFrame, 
        Vsw_DataFrame, Ne_DataFrame, Te_DataFrame, includeNe, includeTe)
    #print('For current orientation, the returned residue is {}'.format(Residue))
    #print('For current orientation, the returned duration is {}'.format(duration))
    if  Residue_diff < Residue_diff_temp:
        time_start_temp = time_start
        time_end_temp = time_end
        time_turn_temp = time_turn
        turnPointOnTop_temp = turnPointOnTop
        Residue_diff_temp = Residue_diff
        Residue_fit_temp = Residue_fit
        theta_temp = 0
        phi_temp = 0
        duration_temp = duration

    # This step loops all theta and phi except for theta = 0.
    thetaArray = np.linspace(0, 90, n_theta_grid+1)
    # print(thetaArray)
    thetaArray = thetaArray[1:]
    phiArray = np.linspace(0, 360, n_theta_grid*2+1)
    phiArray = phiArray[1:]
    for theta_deg in thetaArray: # Not include theta = 0.
        for phi_deg in phiArray: # Include phi = 0.
            time_start, time_end, time_turn, turnPointOnTop, \
            Residue_diff, Residue_fit, duration = getResidueForCurrentAxial(theta_deg, phi_deg, 
                minDuration, B_DataFrame, VHT, dt, flag_smoothA, Np_DataFrame, Tp_DataFrame, 
                Vsw_DataFrame, Ne_DataFrame, Te_DataFrame, includeNe, includeTe)
            # print('For current orientation, the returned residue is {}'.format(Residue_diff))
            #print('For current orientation, the returned duration is {}'.format(duration))
            if Residue_diff < Residue_diff_temp:
                time_start_temp = time_start
                time_end_temp = time_end
                time_turn_temp = time_turn
                turnPointOnTop_temp = turnPointOnTop
                Residue_diff_temp = Residue_diff
                Residue_fit_temp = Residue_fit
                theta_temp = theta_deg
                phi_temp = phi_deg
                duration_temp = duration
               # Tp_in_temp = Tp_in
               # Np_in_temp = Np_in

    # print('Residue_diff = {}'.format(Residue_diff_temp))
    # print('Residue_fit  = {}\n'.format(Residue_fit_temp))
    # Round some results.
    return time_start_temp, time_turn_temp, time_end_temp, \
    duration_temp, turnPointOnTop_temp, Residue_diff_temp, \
    Residue_fit_temp, (theta_temp, phi_temp), (round(VHT[0],5),round(VHT[1],5),round(VHT[2],5))

################################################################################################################
def getResidueForCurrentAxial(theta_deg, phi_deg, minDuration, B_DataFrame, VHT, dt, flag_smoothA, Np_DataFrame, Tp_DataFrame, Vsw_DataFrame, Ne_DataFrame, Te_DataFrame, includeNe, includeTe):
    # Calculate the residue for given theta and phi.

    # Physics constants.
    global mu0 #(N/A^2) magnetic constant permeability of free space vacuum permeability
    global m_proton # Proton mass. In kg.
    global factor_deg2rad # Convert degree to rad.

    time_start = np.nan
    time_end = np.nan
    time_turn = np.nan
    Residue_diff = np.inf
    Residue_fit = np.inf
    duration = np.nan
    turnPointOnTop = np.nan
    
    # Loop for half polar angle (theta(0~90 degree)), and azimuthal angle (phi(0~360 degree)) for Z axis orientations.
    # Direction cosines:
    # x = rcos(alpha) = rsin(theta)cos(phi) => cos(alpha) = sin(theta)cos(phi)
    # y = rcos(beta)  = rsin(theta)sin(phi) => cos(beta)  = sin(theta)sin(phi)
    # z = rcos(gamma) = rcos(theta)         => cos(gamma) = cos(theta)
    # Using direction cosines to form a unit vector.
    theta_rad = factor_deg2rad * theta_deg
    phi_rad   = factor_deg2rad * phi_deg
    # Form new Z_unitVector according to direction cosines.
    Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
    # Find X axis from Z axis and -VHT.
    X_unitVector = findXaxis(Z_unitVector, -VHT)
    # Find the Y axis to form a right-handed coordinater with X and Z.
    Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)

    # Project B_DataFrame & VHT into new trial Frame.
    transToTrialFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
    B_inTrialframe_DataFrame = B_DataFrame.dot(transToTrialFrame)
    VHT_inTrialframe = VHT.dot(transToTrialFrame)

    ########################################
    # Calculate alpha = (MA)**2 according to Eq. in Teh 2018
    # Calculate the remaining flow velocity in the trial frame
    Vsw_inTrialframe = Vsw_DataFrame.dot(transToTrialFrame)
    V_remaining = np.array(Vsw_inTrialframe - VHT_inTrialframe)
    # Calculate the Alfven velocity in the trial frame
    P_massDensity = Np_DataFrame['Np'] * m_proton * 1e6 # In kg/m^3.
    len_P_massDensity = len(P_massDensity)
    P_massDensity_array = np.array(P_massDensity)
    # Reshape density array to prepare the calculation of VA
    P_massDensity_array = np.reshape(P_massDensity_array, (len_P_massDensity, 1))
    # VA: the Alfven velocity
    VA_inTrialframe = np.array(B_inTrialframe_DataFrame * 1e-9) / np.sqrt(mu0 * P_massDensity_array) / 1000.0
    # Calculate Alpha = (Mach Number)**2 according to Eq. in Teh 2018
    MachNum = np.sqrt(np.square(V_remaining).sum(axis=1))/np.sqrt(np.square(VA_inTrialframe).sum(axis=1))
    AlphaMach = (MachNum.mean())**2

    # Calculate the magnitude of the magnetic field to prepare PB
    B_norm_DF = pd.DataFrame(np.sqrt(np.square(B_DataFrame).sum(axis=1)),columns=['|B|'])

    # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
    ds = - VHT_inTrialframe[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s.
    # Don't forget to convert km/s to m/s, and convert nT to T. By = B_inTrialframe_DataFrame[1]
    # A = integrate.cumtrapz(-B_inTrialframe_DataFrame[1]*1e-9, dx=ds, initial=0)
    A = integrate.cumtrapz(-(1 - AlphaMach) * B_inTrialframe_DataFrame[1] * 1e-9, dx=ds, initial=0)
    # Calculate Pt(x,0).
    # Pp = Np * kB * Tp
    if includeNe & includeTe:
        Ppe = np.array(Np_DataFrame['Np']) * 1e6 * k_Boltzmann * 1e9 * np.array(Tp_DataFrame['Tp']) \
        + np.array(Ne_DataFrame['Ne']) * 1e6 * k_Boltzmann * 1e9 * np.array(Te_DataFrame['Te'])
    elif includeTe & (not includeNe):
        Ppe = np.array(Np_DataFrame['Np']) * 1e6 * k_Boltzmann * 1e9 * (np.array(Tp_DataFrame['Tp']) + np.array(Te_DataFrame['Te']))
    else:
        Ppe = np.array(Np_DataFrame['Np']) * 1e6 * k_Boltzmann * 1e9 * np.array(Tp_DataFrame['Tp'])

    # Pb = Bz ** 2 / (2 * mu0)
    Pb = np.array((B_inTrialframe_DataFrame[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa. 
    # PB = |B| ** 2 / (2 * mu0)
    PB = np.array((B_norm_DF['|B|'] * 1e-9)**2 / (2.0 * mu0) * 1e9) # 1e9 convert unit form pa to npa. 
    # Pt = (1 - MA^2)^2 * Pb + (1 - MA^2) * Pp + MA^2 * (1 - MA^) * PB
    Pt = ((1 - AlphaMach)**2) * Pb + (1 - AlphaMach) * Ppe + (AlphaMach * (1 - AlphaMach)) * PB
    #######################################

    # Check how many turning points in original data.
    num_A_turningPoints = turningPoints(A)
    # print('num_A_turningPoints = {}'.format(num_A_turningPoints))

    '''
    if flag_smoothA == True:
        # Smooth A series to find the number of the turning point in main trend.
        # Because the small scale turning points are not important.
        # We only use A_smoothed to find turning points, when fitting Pt with A, original A is used.
        #savgol_filter_window = 9
        order = 3
        A_smoothed = savgol_filter(A, savgol_filter_window, order)
    else:
        A_smoothed = A
    '''
    
    if flag_smoothA == True:
        # Smooth A series to find the number of the turning point in main trend.
        # Because the small scale turning points are not important.
        # We only use A_smoothed to find turning points, when fitting Pt with A, original A is used.
        # First, downsample A to 20 points, then apply savgol_filter, then upsample to original data points number.
        
        #t1 = time.time()
        index_A = range(len(A))
        # Downsample A to 20 points.
        index_downsample = np.linspace(index_A[0],index_A[-1], 20)
        A_downsample = np.interp(index_downsample, index_A, A)
        # Apply savgol_filter.
        A_downsample = savgol_filter(A_downsample, 7, 3) # 7 is smooth window size, 3 is polynomial order.
        # Upsample to original data points amount.
        A_upsample = np.interp(index_A, index_downsample, A_downsample)
        # The smoothed A is just upsampled A.
        A_smoothed = A_upsample
        #t2 = time.time()
        #print('dt = {} ms'.format((t2-t1)*1000.0))
    else:
        A_smoothed = A

    # Check how many turning points in smoothed data.
    num_A_smoothed_turningPoints = turningPoints(A_smoothed)
    # print('num_A_smoothed_turningPoints = {}'.format(num_A_smoothed_turningPoints))
    # num_A_smoothed_turningPoints==0 means the A value is not double folded. It's monotonous. Skip.
    # num_A_smoothed_turningPoints > 1 means the A valuse is 3 or higher folded. Skip.
    
    # continue # Skip the rest commands in current iteration.
    if (num_A_smoothed_turningPoints==0)|(num_A_smoothed_turningPoints>1):
        #return timeRange, Residue, duration, plotData_dict, transToTrialFrame, turnPoint_dict # Skip the rest commands in current iteration.
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration # Skip the rest commands in current iteration.
    #print('Theta={}, Phi={}. Double-folding feature detected!\n'.format(theta_deg, phi_deg))
    
    # Find the boundary of A.
    A_smoothed_start = A_smoothed[0] # The first value of A.
    A_smoothed_end = A_smoothed[-1] # The last value of A.
    A_smoothed_max_index = A_smoothed.argmax() # The index of max A, return the index of first max(A).
    A_smoothed_max = A_smoothed[A_smoothed_max_index] # The max A.
    A_smoothed_min_index = A_smoothed.argmin() # The index of min A, return the index of first min(A).
    A_smoothed_min = A_smoothed[A_smoothed_min_index] # The min A.

    if (A_smoothed_min == min(A_smoothed_start, A_smoothed_end))&(A_smoothed_max == max(A_smoothed_start, A_smoothed_end)):
        # This means the A value is not double folded. It's monotonous. Skip.
        # Sometimes num_A_smoothed_turningPoints == 0 does not work well. This is double check.
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration
    elif abs(A_smoothed_min - ((A_smoothed_start + A_smoothed_end)/2)) < abs(A_smoothed_max - ((A_smoothed_start + A_smoothed_end)/2)):
        # This means the turning point is on the right side.
        A_turnPoint_index = A_smoothed_max_index
        turnPointOnRight = True
    elif abs(A_smoothed_min - ((A_smoothed_start + A_smoothed_end)/2)) > abs(A_smoothed_max - ((A_smoothed_start + A_smoothed_end)/2)):
        # This means the turning point is on the left side.
        A_turnPoint_index = A_smoothed_min_index
        turnPointOnLeft = True

    # Split A into two subarray from turning point.
    A_sub1 = A[:A_turnPoint_index+1]
    Pt_sub1 = Pt[:A_turnPoint_index+1] # Pick corresponding Pt according to index of A.
    A_sub2 = A[A_turnPoint_index:]
    Pt_sub2 = Pt[A_turnPoint_index:] # Pick corresponding Pt according to index of A.

    # Get time stamps.
    timeStamp = B_inTrialframe_DataFrame.index
    # Split time stamps into two subarray from turning point.
    timeStamp_sub1 = timeStamp[:A_turnPoint_index+1]
    timeStamp_sub2 = timeStamp[A_turnPoint_index:]
    # Keep the time of turn point and the value of Pt turn point.
    Pt_turnPoint = Pt[A_turnPoint_index]
    timeStamp_turnPoint = timeStamp[A_turnPoint_index]
    # This block is to find the time range.
    # Put two branches into DataFrame.
    Pt_vs_A_sub1_DataFrame = pd.DataFrame({'Pt_sub1':np.array(Pt_sub1).T,'timeStamp_sub1':np.array(timeStamp_sub1).T}, index=A_sub1)
    Pt_vs_A_sub2_DataFrame = pd.DataFrame({'Pt_sub2':np.array(Pt_sub2).T,'timeStamp_sub2':np.array(timeStamp_sub2).T}, index=A_sub2)

    # Sort by A. A is index in Pt_vs_A_sub1_DataFrame.
    Pt_vs_A_sub1_DataFrame.sort_index(ascending=True, inplace=True, kind='quicksort')
    Pt_vs_A_sub2_DataFrame.sort_index(ascending=True, inplace=True, kind='quicksort')
    # Trim two branches to get same boundary A value.
    # Note that, triming is by A value, not by length. After trimming, two branches may have different lengths.
    A_sub1_boundary_left = Pt_vs_A_sub1_DataFrame.index.min()
    A_sub1_boundary_right = Pt_vs_A_sub1_DataFrame.index.max()
    A_sub2_boundary_left = Pt_vs_A_sub2_DataFrame.index.min()
    A_sub2_boundary_right = Pt_vs_A_sub2_DataFrame.index.max()

    A_boundary_left = max(A_sub1_boundary_left, A_sub2_boundary_left)
    A_boundary_right = min(A_sub1_boundary_right, A_sub2_boundary_right)

    Pt_vs_A_sub1_trimmed_DataFrame = Pt_vs_A_sub1_DataFrame.iloc[Pt_vs_A_sub1_DataFrame.index.get_loc(A_boundary_left,method='nearest'):Pt_vs_A_sub1_DataFrame.index.get_loc(A_boundary_right,method='nearest')+1]
    Pt_vs_A_sub2_trimmed_DataFrame = Pt_vs_A_sub2_DataFrame.iloc[Pt_vs_A_sub2_DataFrame.index.get_loc(A_boundary_left,method='nearest'):Pt_vs_A_sub2_DataFrame.index.get_loc(A_boundary_right,method='nearest')+1]

    # Get the time range of trimmed A.
    timeStamp_start = min(Pt_vs_A_sub1_trimmed_DataFrame['timeStamp_sub1'].min(skipna=True), Pt_vs_A_sub2_trimmed_DataFrame['timeStamp_sub2'].min(skipna=True))
    timeStamp_end = max(Pt_vs_A_sub1_trimmed_DataFrame['timeStamp_sub1'].max(skipna=True), Pt_vs_A_sub2_trimmed_DataFrame['timeStamp_sub2'].max(skipna=True))
    #timeRange = [timeStamp_start, timeStamp_end]
    time_start = int(timeStamp_start.strftime('%Y%m%d%H%M%S'))
    time_end = int(timeStamp_end.strftime('%Y%m%d%H%M%S'))
    time_turn = int(timeStamp_turnPoint.strftime('%Y%m%d%H%M%S'))
    duration = int((timeStamp_end - timeStamp_start).total_seconds()/dt)+1

    # Skip if shorter than minDuration.
    if duration < minDuration:
        return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration

    # Calculate two residues respectively. Residue_fit and Residue_diff.
    # Preparing for calculating Residue_fit, the residue of all data sample w.r.t. fitted PtA curve.
    # Combine two trimmed branches.
    A_sub1_array = np.array(Pt_vs_A_sub1_trimmed_DataFrame.index)
    A_sub2_array = np.array(Pt_vs_A_sub2_trimmed_DataFrame.index)
    Pt_sub1_array = np.array(Pt_vs_A_sub1_trimmed_DataFrame['Pt_sub1'])
    Pt_sub2_array = np.array(Pt_vs_A_sub2_trimmed_DataFrame['Pt_sub2'])
    # The order must be in accordance.
    Pt_array = np.concatenate((Pt_sub1_array, Pt_sub2_array))
    A_array = np.concatenate((A_sub1_array, A_sub2_array))
    # Sort index.
    sortedIndex = np.argsort(A_array)
    A_sorted_array = A_array[sortedIndex]
    Pt_sorted_array = Pt_array[sortedIndex]

    # Fit a polynomial function (3rd order). Use it to calculate residue.
    Pt_array_float=Pt_array.astype(np.float64)
    Pt_A_coeff = np.polyfit(A_array, Pt_array_float, 3)
    Pt_A = np.poly1d(Pt_A_coeff)
    # Preparing for calculating Residue_diff, the residue get by compare two branches.
    # Merge two subset into one DataFrame.
    Pt_vs_A_trimmed_DataFrame = pd.concat([Pt_vs_A_sub1_trimmed_DataFrame, Pt_vs_A_sub2_trimmed_DataFrame], axis=1)
    # Drop timeStamp.
    Pt_vs_A_trimmed_DataFrame.drop(['timeStamp_sub1', 'timeStamp_sub2'], axis=1, inplace=True) # axis=1 for column.

    # Interpolation.
    # "TypeError: Cannot interpolate with all NaNs" can occur if the DataFrame contains columns of object dtype. Convert data to numeric type. Check data type by print(Pt_vs_A_trimmed_DataFrame.dtypes).
    for one_column in Pt_vs_A_trimmed_DataFrame:
        Pt_vs_A_trimmed_DataFrame[one_column] = pd.to_numeric(Pt_vs_A_trimmed_DataFrame[one_column], errors='coerce')
    # Interpolate according to index A.
    Pt_vs_A_trimmed_DataFrame.interpolate(method='index', axis=0, inplace=True) # axis=0:fill column-by-column
    # Drop leading and trailing NaNs. The leading NaN won't be filled by linear interpolation, however,
    # the trailing NaN will be filled by forward copy of the last non-NaN values. So, for leading NaN,
    # just use pd.dropna, and for trailing NaN, remove the duplicated values.
    Pt_vs_A_trimmed_DataFrame.dropna(inplace=True) # Drop leading NaNs.
    trailing_NaN_mask_DataFrame = (Pt_vs_A_trimmed_DataFrame.diff()!=0) # Get duplicate bool mask.
    trailing_NaN_mask = np.array(trailing_NaN_mask_DataFrame['Pt_sub1'] & trailing_NaN_mask_DataFrame['Pt_sub2'])
    Pt_vs_A_trimmed_DataFrame = Pt_vs_A_trimmed_DataFrame.iloc[trailing_NaN_mask]

    # Get Pt_max and Pt_min. They will be used to normalize Residue for both Residue_fit and Residue_diff.
    Pt_max = Pt_sorted_array.max()
    Pt_min = Pt_sorted_array.min()
    Pt_max_min_diff = abs(Pt_max - Pt_min)
    # Check if turn point is on top.
    turnPointOnTop = (Pt_turnPoint>(Pt_max-(Pt_max-Pt_min)*0.15))
    # Use two different defination to calculate Residues. # Note that, the definition of Residue_diff is different with Hu's paper. We divided it by 2 two make it comparable with Residue_fit. The definition of Residue_fit is same as that in Hu2004.
    if Pt_max_min_diff == 0:
        Residue_diff = np.inf
        Residue_fit = np.inf
    else:
        Residue_diff = 0.5 * np.sqrt((1.0/len(Pt_vs_A_trimmed_DataFrame))*((Pt_vs_A_trimmed_DataFrame['Pt_sub1'] - Pt_vs_A_trimmed_DataFrame['Pt_sub2']) ** 2).sum()) / Pt_max_min_diff
        Residue_fit = np.sqrt((1.0/len(A_array))*((Pt_sorted_array - Pt_A(A_sorted_array)) ** 2).sum()) / Pt_max_min_diff
        # Round results.
        Residue_diff = round(Residue_diff, 5)
        Residue_fit = round(Residue_fit, 5)

    # print(time_start, Residue_diff)
    return time_start, time_end, time_turn, turnPointOnTop, Residue_diff, Residue_fit, duration

################################################################################################################
def detect_flux_rope(spacecraftID, data_DF, duration_range_tuple, search_result_dir, data_dict, **kwargs):
    """This function as the main search process of flux rope. 
    Return the raw results that include records in different windows."""

    MutliResolution = False
    if 'MutliResolution' in kwargs: 
        MutliResolution = kwargs['MutliResolution']
    reverseOrder = False
    if 'reverseOrder' in kwargs: 
        reverseOrder = kwargs['reverseOrder']
    # If reverseOrder == True, run it for high resolution first.
    # Then run it for regular/default resolution 28s.

    if 'spacecraftID' in kwargs:
        spacecraftID = kwargs['spacecraftID']
    if spacecraftID == 'ACE' or 'WIND': dt = 60.0
    if spacecraftID == 'ULYSSES': dt = 240.0
    if (spacecraftID == 'PSP') & (not reverseOrder): 
        dt = 28.0
    elif (spacecraftID == 'PSP') & reverseOrder:
        dt = 1.0
    if (spacecraftID == 'SOLARORBITER') & (not reverseOrder): 
        dt = 28.0
    elif (spacecraftID == 'SOLARORBITER') & reverseOrder:
        dt = 4.0

    flag_smoothA = True
    B_DataFrame = data_DF.iloc[:,0:3]
    Vsw_DataFrame = data_DF.iloc[:,3:6] 
    Np_DataFrame = data_DF.loc[:,['Np']]
    Tp_DataFrame = data_DF.loc[:,['Tp']]
    if ('Te' in data_DF.columns) & includeTe: 
        Te_DataFrame = data_DF.loc[:,['Te']]
    else: 
        Te_DataFrame = None
        print("Attention: Te is not available/included.")
    if ('Ne' in data_DF.columns) & includeNe: 
        Ne_DataFrame = data_DF.loc[:,['Ne']]
    else: 
        Ne_DataFrame = None
        print("Attention: Ne is not available/included.")
    if 'Ta' in data_DF.columns: 
        Ta_DataFrame = data_DF.loc[:,['Ta']]
    else: Ta_DataFrame = None
    if 'Na' in data_DF.columns: 
        Na_DataFrame = data_DF.loc[:,['Na']]
    else:
        Na_DataFrame = None

    datetimeStart = data_DF.index[0]
    datetimeEnd = data_DF.index[-1]
    print('\nspacecraftID                = {}'.format(spacecraftID))
    print('Time increment dt           = {} seconds'.format(dt))

    # Multiprocessing
    num_cpus = multiprocessing.cpu_count()
    max_processes = num_cpus
    print('Totol CPU cores used        = {}'.format(num_cpus))
    # Create a multiprocessing pool with safe_lock.
    pool = multiprocessing.Pool(processes=max_processes)
    # Create a list to save result.
    results = []

    # Apply GS detection in sliding window.
    # Set searching parameters.
    n_theta_grid = 9 # theta grid number. 90/9=10, d_theta=10(degree); 90/12=7.5, d_theta=7.5(degree)
    if 'n_theta_grid' in kwargs:
        n_theta_grid = kwargs['n_theta_grid']
    print('Grid size of theta & phi    = {} & {}'.format(90/n_theta_grid, 180/n_theta_grid))
    # First integer in tuple is minimum duration threshold, second integer in tuple is searching window width.
    # duration_range_tuple = ((20,30), (30,40), (40,50), (50,60)) #
    print('Duration range tuple        = {}'.format(duration_range_tuple))
    search_result_raw_true = {}
    search_result_raw_false = {}
    totalStartTime = datetime.now()
    for duration_range in duration_range_tuple: # Loop different window width.
        startTime = datetime.now()
       
        print('\n{}'.format(time.ctime()))
        minDuration = duration_range[0]
        maxDuration = duration_range[1]
        print('Duration : {} ~ {} points.'.format(minDuration, maxDuration))

        # The maximum gap tolerance is up to 30% of total points count.
        interp_limit = int(math.ceil(minDuration*3.0/10)) # Flexible interpolation limit based on window length.
        print('interp_limit = {}'.format(interp_limit))
        # Sliding window. If half_window=15, len(DataFrame)=60, range(15,45)=[15,...,44].
        for indexFluxRopeStart in range(len(B_DataFrame) - maxDuration): # in minutes.
            indexFluxRopeEnd = indexFluxRopeStart + maxDuration - 1  # The end point is included, so -1.
            # Grab the B slice within the window. Change the slice will change the original DataFrame.
            B_inWindow = B_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1] # End is not included.
            if B_inWindow.isnull().values.sum():
                B_inWindow_copy = B_inWindow.copy(deep=True)
                B_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if B_inWindow_copy.isnull().values.sum():
                    continue
                else:
                    B_inWindow = B_inWindow_copy

            # Grab the Vsw slice within the window. Change the slice will change the original DataFrame.
            Vsw_inWindow = Vsw_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1]
            if Vsw_inWindow.isnull().values.sum():
                Vsw_inWindow_copy = Vsw_inWindow.copy(deep=True)
                Vsw_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if Vsw_inWindow_copy.isnull().values.sum():
                    continue
                else:
                    Vsw_inWindow = Vsw_inWindow_copy
            
            # Grab the Np slice within the window. Change the slice will change the original DataFrame.
            Np_inWindow = Np_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1]
            if Np_inWindow.isnull().values.sum():
                Np_inWindow_copy = Np_inWindow.copy(deep=True)
                Np_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if Np_inWindow_copy.isnull().values.sum():
                    continue
                else:
                    Np_inWindow = Np_inWindow_copy

            Tp_inWindow = Tp_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1]
            if Tp_inWindow.isnull().values.sum():
                Tp_inWindow_copy = Tp_inWindow.copy(deep=True)
                Tp_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                if Tp_inWindow_copy.isnull().values.sum():
                    continue
                else:
                    Tp_inWindow = Tp_inWindow_copy

            if Te_DataFrame is not None:
                Te_inWindow = Te_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1]
                if Te_inWindow.isnull().values.sum():
                    Te_inWindow_copy = Te_inWindow.copy(deep=True)
                    Te_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                    if Te_inWindow_copy.isnull().values.sum():
                        continue
                    else:
                        Te_inWindow = Te_inWindow_copy
            else:
                Te_inWindow = None

            if Ne_DataFrame is not None:
                Ne_inWindow = Ne_DataFrame.iloc[indexFluxRopeStart : indexFluxRopeEnd + 1]
                if Ne_inWindow.isnull().values.sum():
                    Ne_inWindow_copy = Ne_inWindow.copy(deep=True)
                    Ne_inWindow_copy.interpolate(method='time', limit=interp_limit, inplace=True)
                    if Ne_inWindow_copy.isnull().values.sum():
                       continue
                    else:
                        Ne_inWindow = Ne_inWindow_copy
            else:
                Ne_inWindow = None
            
            VHT_inGSE = findVHT(B_inWindow, Vsw_inWindow) # Very slow.
            # VHT_inGSE = np.array(Vsw_inWindow.mean())
            result_temp = pool.apply_async(searchFluxRopeInWindow, args=(B_inWindow, VHT_inGSE, n_theta_grid, 
                minDuration, dt, flag_smoothA, \
                Np_inWindow, Tp_inWindow, Vsw_inWindow, Ne_inWindow, Te_inWindow, includeNe, includeTe))
            results.append(result_temp)
            # DO NOT unpack result here. It will block IO. Unpack in bulk.

        # Next we are going to save file We have to wait for all worker processes to finish.
        # Block main process to wait for worker processes to finish. This while loop will execute almost immediately when the innner for loop goes through. The inner for loop is non-blocked, so it finish in seconds.
        while len(pool._cache)!=0:
            #print('{} - Waiting... There are {} worker processes in pool.'.format(time.ctime(), len(pool._cache)))
            time.sleep(1)
        print('{} - len(pool._cache) = {}'.format(time.ctime(), len(pool._cache)))
        print('{} - Duration range {}~{} points is completed!'.format(time.ctime(), minDuration, maxDuration))

        # Save result. One file per window size.
        results_true_tuple_list = []
        results_false_tuple_list = []
        # Unpack results. Convert to tuple, and put into list.
        for one_result in results:
            results_tuple_temp = (one_result.get())
            if not np.isinf(results_tuple_temp[5]): # Check residue.
                # if True, turn point on top.
                # 2nd & 3rd if: remove candidates that have R_dif > 0.12 or R_fit > 0.14
                if results_tuple_temp[4]: 
                # if results_tuple_temp[4] & (results_tuple_temp[5] <= 0.12) & (results_tuple_temp[6] <= 0.14): 
                    results_true_tuple_list.append(results_tuple_temp)
                else: # Turn point on bottom.
                    results_false_tuple_list.append(results_tuple_temp)

        # print(results_true_tuple_list)
        # Save results to dictionary. One key per window size.
        if MutliResolution & (not reverseOrder):
            key_temp = str(minDuration) + '~' + str(maxDuration) +'11111'
        else:
            key_temp = str(minDuration) + '~' + str(maxDuration)
        search_result_raw_true[key_temp] = results_true_tuple_list
        search_result_raw_false[key_temp] = results_false_tuple_list

        # Empty container results[].
        results = []

        endTime = datetime.now()
        time_spent_in_seconds = (endTime - startTime).total_seconds()
        print('Time spent on this window: {} seconds ({} minutes).'.format(time_spent_in_seconds, round(time_spent_in_seconds/60.0, 2)))

    # Close pool, prevent new worker process from joining.
    pool.close()
    # Block caller process until workder processes terminate.
    pool.join()

    totalEndTime = datetime.now()
    time_spent_in_seconds = (totalEndTime - totalStartTime).total_seconds()

    print('\n{} - All duration ranges are completed!'.format(time.ctime()))
    print('\nSaving search result...')
    if MutliResolution & (not reverseOrder):
        search_result_raw = {'true':search_result_raw_true, 'false':search_result_raw_false, 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}}
        search_result_raw_filename = search_result_dir + '/raw_result_28s_resltn.p'
        pickle.dump(search_result_raw, open(search_result_raw_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    else:
        search_result_raw = {'true':search_result_raw_true, 'false':search_result_raw_false, 'timeRange':{'datetimeStart':datetimeStart, 'datetimeEnd':datetimeEnd}}
        search_result_raw_filename = search_result_dir + '/raw_result.p'
        pickle.dump(search_result_raw, open(search_result_raw_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    # print('\nTotal CPU cores: {}.'.format(num_cpus))
    # print('Max number of workder process in pool: {}.'.format(max_processes))
    print('Total Time spent: {} seconds ({} minutes).\n'.format(time_spent_in_seconds, round(time_spent_in_seconds/60.0, 2)))
    print('Done.')
    return search_result_raw

################################################################################################################
def clean_up_raw_result(spacecraftID, data_DF, dataObject_or_dataPath, **kwargs):
    """This function cleans up the flux rope candidates 
    from the raw search results. Return flux rope records in a pickle file."""

    # Check input datatype:
    # If dataObject_or_dataPath is an object(dict):
    if isinstance(dataObject_or_dataPath, dict):
        # print('\nYour input is a dictionary data.')
        search_result_raw = dataObject_or_dataPath
    elif isinstance(dataObject_or_dataPath, str):
        # print('\nYour input is a path. Load the dictionary data via this path.')
        search_result_raw = pd.read_pickle(open(dataObject_or_dataPath, 'rb'))
    else:
        print('\nPlease input the correct datatype!')
        return None

    # Set default value for parameters.
    # Set minimum residue.
    min_residue_diff = 0.12
    min_residue_fit = 0.14
    # Set fitted curve quality parameters.
    max_tailPercentile = 0.3
    max_tailDiff = 0.3
    max_PtFitStd = 0.3
    # Remove discontinuity.
    isRemoveShock = False
    Vsw_std_threshold = 10000 # Max allowed standard deviation for solar wind speed.
    Vsw_diff_threshold = 10000 # Max allowed solar wind max-min difference.
    # walen test.
    walenTest_r_threshold = 0.800 # correlation coefficient.
    walenTest_k_threshold = 1.0 # slope.
    # B_mag_threshold = 5.0 # nT
    # Display control.
    isVerbose = False
    isPrintIntermediateDF = False
    SettingBLimit = False
    allowOverlap = False
    # output filename.
    output_filename = 'no_overlap'
    # output dir.
    output_dir = os.getcwd()
    
    # If keyword is specified, overwrite the default value.
    print('\nSetting parameters:')
    if 'spacecraftID' in kwargs:
        spacecraftID = kwargs['spacecraftID']
        print('spacecraftID          = {}'.format(spacecraftID))
    if spacecraftID == 'ACE' or 'WIND': dt = 60.0
    if spacecraftID == 'ULYSSES': dt = 240.0
    if spacecraftID == 'PSP': 
        dt = 28.0
        if 'the2ndDF' in kwargs:
            the2ndDF = kwargs['the2ndDF']
            dt = 1.0
            dt_2nd = 28.0
    if spacecraftID == 'SOLARORBITER': 
        dt = 28.0
        if 'the2ndDF' in kwargs:
            the2ndDF = kwargs['the2ndDF']
            dt = 4.0
            dt_2nd = 28.0
    if 'dt' in kwargs:
        dt = kwargs['dt']
        print('Time increment dt     = {} seconds'.format(dt))
    else:
        print('Time increment dt     = {} seconds'.format(dt))
        if 'the2ndDF' in kwargs: 
            print('Time increment dt2    = {} seconds'.format(dt_2nd))
    if 'min_residue_diff' in kwargs:
        min_residue_diff = kwargs['min_residue_diff']
        print('min_residue_diff      = {}'.format(min_residue_diff))
    else:
        print('min_residue_diff      = {}'.format(min_residue_diff))    
    if 'min_residue_fit' in kwargs:
        min_residue_fit = kwargs['min_residue_fit']
        print('min_residue_fit       = {}'.format(min_residue_fit))
    else:
        print('min_residue_fit       = {}'.format(min_residue_fit))    
    if 'max_tailPercentile' in kwargs:
        max_tailPercentile = kwargs['max_tailPercentile']
        print('max_tailPercentile    = {}'.format(max_tailPercentile))
    else:
        print('max_tailPercentile    = {}'.format(max_tailPercentile))    
    if 'max_tailDiff' in kwargs:
        max_tailDiff = kwargs['max_tailDiff']
        print('max_tailDiff          = {}'.format(max_tailDiff))
    else:
        print('max_tailDiff          = {}'.format(max_tailDiff))   
    if 'max_PtFitStd' in kwargs:
        max_PtFitStd = kwargs['max_PtFitStd']
        print('max_PtFitStd          = {}'.format(max_PtFitStd))
    else:
        print('max_PtFitStd          = {}'.format(max_PtFitStd))   
    if spacecraftID == 'PSP':
        print('Vsw_std_threshold     = {} km/s'.format(Vsw_std_threshold))
        print('Vsw_diff_threshold    = {} km/s'.format(Vsw_diff_threshold))
    else:
        Vsw_std_threshold = 18.0
        print('Vsw_std_threshold     = {} km/s'.format(Vsw_std_threshold))
        Vsw_diff_threshold = 60.0
        print('Vsw_diff_threshold    = {} km/s'.format(Vsw_diff_threshold))
    if 'walenTest_r_threshold' in kwargs:
        walenTest_r_threshold = kwargs['walenTest_r_threshold']
        print('walenTest_r_threshold = {}'.format(walenTest_r_threshold))
    else:
        print('walenTest_r_threshold = {}'.format(walenTest_r_threshold))
    if 'walenTest_k_threshold' in kwargs:
        walenTest_k_threshold = kwargs['walenTest_k_threshold']
        print('walenTest_k_threshold = {}'.format(walenTest_k_threshold))
    else:
        print('walenTest_k_threshold = {}'.format(walenTest_k_threshold))
    if 'SettingBLimit' in kwargs:
        SettingBLimit = kwargs['SettingBLimit']
        if SettingBLimit:
            if 'B_mag_threshold' in kwargs:
                B_mag_threshold = kwargs['B_mag_threshold']
                print('B_mag_threshold       = {} nT'.format(B_mag_threshold))
            else:
                print('Warning: B_mag_threshold is not provided')
                return None
    else:
        print('Setting |B| limit?    = {}'.format(SettingBLimit))
    if 'isVerbose' in kwargs:
        isVerbose = kwargs['isVerbose']
        print('isVerbose             = {}'.format(isVerbose))
    if 'allowOverlap' in kwargs:
        allowOverlap = kwargs['allowOverlap']
        print('allowOverlap          = {}'.format(allowOverlap))
    else:
        print('allowOverlap          = {}'.format(allowOverlap))
    if 'isPrintIntermediateDF' in kwargs:
        isPrintIntermediateDF = kwargs['isPrintIntermediateDF']
        print('isPrintIntermediateDF = {}'.format(isPrintIntermediateDF))
    else:
        print('isPrintIntermediateDF = {}'.format(isPrintIntermediateDF))
    if 'isRemoveShock' in kwargs:
        isRemoveShock = kwargs['isRemoveShock']
        print('Remove Shock?         = {}'.format(isRemoveShock))
        if isRemoveShock:
            if 'shockList_DF' in kwargs:
                shockList_DF = kwargs['shockList_DF']
                print('ShockList Loaded?     = Done')
            else:
                print('Warning: shockList_DF is not provided')
                return None    
    if 'output_dir' in kwargs:
        output_dir = kwargs['output_dir']
        print('output_dir            = {}'.format(output_dir))
    if 'output_filename' in kwargs:
        output_filename = kwargs['output_filename']
        print('output_filename       = {}'.format(output_filename))
    if 'shockList' in kwargs:
        shockList = kwargs['shockList']
        print('isRemoveShock         = {}'.format(isRemoveShock))

    # Set terminal display format.
    if isPrintIntermediateDF:
        pd.set_option('display.max_rows', 1000)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 500)


    # Get duration list.
    duration_list = search_result_raw['true'].keys()
    window_size_list = []
    for item in duration_list:
        window_size = int(item.split('~')[1])
        window_size_list.append(window_size)

    # Sort the duration_list with window_size_list by argsort.
    # The duration is in descending order, such that the longest window has a priority at Step 8.
    sorted_index_window_size_array = np.argsort(window_size_list)
    sorted_index_window_size_array = sorted_index_window_size_array[::-1]
    duration_array = np.array(list(duration_list)) # Python 3
    # duration_array = np.array(duration_list) # Python 2
    duration_list = list(duration_array[sorted_index_window_size_array])
    print('\nduration_list:')
    print(duration_list)
        
    # Get search_iteration.
    search_iteration = len(duration_list)
    # Get start and end time.
    datetimeStart = search_result_raw['timeRange']['datetimeStart']
    datetimeEnd = search_result_raw['timeRange']['datetimeEnd']
    # Create empty eventList_no_overlap DataFrame.
    eventList_DF_noOverlap = pd.DataFrame(columns=['startTime', 'turnTime', 'endTime', 'duration', 'residue_diff', 'residue_fit', 'theta_phi', 'VHT'])

    for i_iteration in range(search_iteration):
        print('\n======================================================================')
        
        # Create empty DataFrame.
        TargetedSlot = pd.DataFrame(OrderedDict((('slotStart', [datetimeStart]),('slotEnd', [datetimeEnd]))))
        duration_str_temp = duration_list[i_iteration]
        print('iteration = {}/{}: Checking records from searching window {} points.'.format(i_iteration+1, search_iteration, duration_str_temp))
        eventList_temp = search_result_raw['true'][duration_str_temp]

        # 0) Check point 0.
        # If event list not empty, put it into DataFrame.
        if not eventList_temp: # An empty list is itself considered false in true value testing.
            # Event list is empty. Skip the rest operations.
            print('\nEvent list eventList_temp is empty!')
            print('Go the the next iteration!')
            continue

        # Create headers.
        eventList_temp_Header = ['startTime', 'turnTime', 'endTime', 'duration', 'topTurn', 'residue_diff', 'residue_fit', 'theta_phi', 'VHT']
        # Convert 2-D list to DataFrame.
        eventList_DF_0_original_temp = pd.DataFrame(eventList_temp, columns=eventList_temp_Header)
        # Parse string to datetime.
        eventList_DF_0_original_temp['startTime'] = pd.to_datetime(eventList_DF_0_original_temp['startTime'], format="%Y%m%d%H%M%S")
        eventList_DF_0_original_temp['turnTime'] = pd.to_datetime(eventList_DF_0_original_temp['turnTime'], format="%Y%m%d%H%M%S")
        eventList_DF_0_original_temp['endTime'] = pd.to_datetime(eventList_DF_0_original_temp['endTime'], format="%Y%m%d%H%M%S")

        # ======================================================== S T E P. 0 ===========================================================
        print('\nStep 0. Putting candidates into targeted slots.')
        eventList_DF_0_TargetedSlot_temp = eventList_DF_0_original_temp.copy()
        # Determine whether a record is between slotStart & slotEnd
        # Generally, these two parameters are identical to the time period for searching unless specified.
        # E.g., if one runs detection for a longer time period while only needs results within a shorter time period.
        eventList_DF_0_TargetedSlot_temp = eventList_DF_0_TargetedSlot_temp.assign(keepFlag=[False]*len(eventList_DF_0_TargetedSlot_temp))
        for index, TargetedSlot in TargetedSlot.iterrows():
            keepMask = (eventList_DF_0_TargetedSlot_temp['startTime']>=TargetedSlot['slotStart'])&(eventList_DF_0_TargetedSlot_temp['endTime']<=TargetedSlot['slotEnd'])
            if(keepMask.sum()>0):
                eventList_DF_0_TargetedSlot_temp.loc[eventList_DF_0_TargetedSlot_temp[keepMask].index, 'keepFlag'] = True 
        eventList_DF_0_TargetedSlot_temp = eventList_DF_0_TargetedSlot_temp[eventList_DF_0_TargetedSlot_temp['keepFlag']==True]
        eventList_DF_0_TargetedSlot_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_0_original_temp),len(eventList_DF_0_TargetedSlot_temp)))
        # print(eventList_DF_0_TargetedSlot_temp)
        # ======================================================== S T E P. 1 ===========================================================
        # 1) Check point 1.
        if eventList_DF_0_TargetedSlot_temp.empty:
            print('DataFrame after putting candidates into targeted slots is empty!')
            print('Go the the next iteration!')
            continue

        print('\nStep 1. Removing candidates with residue_diff > {} and residue_fit > {}.'.format(min_residue_diff, min_residue_fit))
        eventList_DF_1_CheckResidue_temp = eventList_DF_0_TargetedSlot_temp[(eventList_DF_0_TargetedSlot_temp['residue_diff']<=min_residue_diff)&(eventList_DF_0_TargetedSlot_temp['residue_fit']<=min_residue_fit)]
        eventList_DF_1_CheckResidue_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_0_TargetedSlot_temp),len(eventList_DF_1_CheckResidue_temp)))
        # print(eventList_DF_1_CheckResidue_temp)
        # exit()
        # ======================================================== S T E P. 2 ===========================================================
        # 2) Check point 2.
        # After removing unqualified residue, check if eventList_DF_1_CheckResidue_temp is empty.
        if eventList_DF_1_CheckResidue_temp.empty:
            print('\nDataFrame after removing events with unqualified residues is empty!')
            print('Go the the next iteration!')
            continue

        if isRemoveShock:
            print('\nStep 2. Removing candidates containing shock.')
            eventList_DF_2_RemoveShock_temp = eventList_DF_1_CheckResidue_temp.copy()
            # Get trimmed shockList.
            shockList_trimmed_DF = shockList_DF[(shockList_DF.index>=datetimeStart)&(shockList_DF.index<=datetimeEnd)]
            spacecraftID_dict = {'WIND':'Wind', 'ACE':'ACE','ULYSSES':'Ulysses','PSP':'PSP','SOLARORBITER':'SOLARORBITER'}
            shockList_trimmed_specifiedSpacecraft_DF = shockList_trimmed_DF[(shockList_trimmed_DF['Spacecraft'].str.contains(spacecraftID_dict[spacecraftID]))]
            # Reset index. The original index is shock time. Put shock time into a new column.
            shockList_trimmed_specifiedSpacecraft_DF = shockList_trimmed_specifiedSpacecraft_DF.reset_index().rename(columns={'index':'shockTime'}).copy()
            len_shockList = len(shockList_trimmed_specifiedSpacecraft_DF)
            for index_temp, shock_record_temp in shockList_trimmed_specifiedSpacecraft_DF.iterrows():
                shockTime_temp = shock_record_temp['shockTime']
                mask_containShock = (eventList_DF_2_RemoveShock_temp['startTime']<=shockTime_temp)&(eventList_DF_2_RemoveShock_temp['endTime']>=shockTime_temp)
                if mask_containShock.sum()>0:
                    eventList_DF_2_RemoveShock_temp = eventList_DF_2_RemoveShock_temp[~mask_containShock]
            eventList_DF_2_RemoveShock_temp = eventList_DF_2_RemoveShock_temp.reset_index(drop=True)
            if isVerbose:
                print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_1_CheckResidue_temp),len(eventList_DF_2_RemoveShock_temp)))
      
        # ======================================================== S T E P. 3 ===========================================================
        # 3) Check point 3.
        if eventList_DF_2_RemoveShock_temp.empty:
            print('\nDataFrame after removing shock is empty!')
            print('Go the the next iteration!')
            continue
        print('\nStep 3. Checking candidates based on the Walen test.')
        # .assign() always returns a copy of the data, leaving the original DataFrame untouched.
        eventList_DF_3_CheckWalenTest_temp = eventList_DF_2_RemoveShock_temp.copy()
        eventList_DF_3_CheckWalenTest_temp = eventList_DF_3_CheckWalenTest_temp.assign(r = len(eventList_DF_3_CheckWalenTest_temp)*[np.nan])
        eventList_DF_3_CheckWalenTest_temp = eventList_DF_3_CheckWalenTest_temp.assign(k = len(eventList_DF_3_CheckWalenTest_temp)*[np.nan])
        eventList_DF_3_CheckWalenTest_temp = eventList_DF_3_CheckWalenTest_temp.assign(MA = len(eventList_DF_3_CheckWalenTest_temp)*[np.nan])
        eventList_DF_3_CheckWalenTest_temp['lowTail'] = [None]*len(eventList_DF_3_CheckWalenTest_temp)
        eventList_DF_3_CheckWalenTest_temp['lowTailDiff'] = [None]*len(eventList_DF_3_CheckWalenTest_temp)
        eventList_DF_3_CheckWalenTest_temp['lowStd'] = [None]*len(eventList_DF_3_CheckWalenTest_temp)
        eventList_DF_3_CheckWalenTest_temp = eventList_DF_3_CheckWalenTest_temp.assign(Vsw_std=[-1.0]*len(eventList_DF_3_CheckWalenTest_temp))
        eventList_DF_3_CheckWalenTest_temp = eventList_DF_3_CheckWalenTest_temp.assign(Vsw_diff=[-1.0]*len(eventList_DF_3_CheckWalenTest_temp))
        if SettingBLimit: eventList_DF_3_CheckWalenTest_temp['Bthreshold'] = np.nan*len(eventList_DF_3_CheckWalenTest_temp)    
        eventList_DF_3_CheckWalenTest_temp.reset_index(drop=True, inplace=True)

        for index, FR_record in eventList_DF_3_CheckWalenTest_temp.iterrows():
            # Grab the data for events from FR_record.            
            theta_deg, phi_deg = FR_record['theta_phi']            
            fluxRopeStartTime = FR_record['startTime']
            fluxRopeTurnTime   = FR_record['turnTime']
            fluxRopeEndTime   = FR_record['endTime']
            theta_deg, phi_deg = FR_record['theta_phi']
            Residue_diff = FR_record['residue_diff']
            Residue_fit = FR_record['residue_fit']
            VHT_inOriFrame = np.array(FR_record['VHT'])

            if ((spacecraftID == 'PSP') or (spacecraftID == 'SOLARORBITER')) & (len(duration_str_temp) >= 10):
                selectedRange_mask = (the2ndDF.index >= FR_record['startTime']) & (the2ndDF.index <= FR_record['endTime'])
                record_data_temp = the2ndDF.iloc[selectedRange_mask]
            else:
                selectedRange_mask = (data_DF.index >= FR_record['startTime']) & (data_DF.index <= FR_record['endTime'])
                record_data_temp = data_DF.iloc[selectedRange_mask]

            B_inOriFrame = record_data_temp.iloc[:,0:3].copy(deep=True)
            Vsw_inOriFrame = record_data_temp.loc[:,['V0','V1','V2']].copy(deep=True)
            Np_inOriFrame = record_data_temp.loc[:,['Np']].copy(deep=True)
            Tp_inOriFrame = record_data_temp.loc[:,['Tp']].copy(deep=True)
            
            # Calculate Vsw_std and Vsw_diff & Prepare to remove large Vsw fluctuations .
            Vsw_temp = np.sqrt(np.square(Vsw_inOriFrame).sum(axis=1,skipna=True))
            Vsw_diff_temp = np.nanmax(np.array(Vsw_temp)) - np.nanmin(np.array(Vsw_temp))
            eventList_DF_3_CheckWalenTest_temp.loc[index, 'Vsw_std'] = Vsw_temp.std(skipna=True)
            eventList_DF_3_CheckWalenTest_temp.loc[index, 'Vsw_diff'] = Vsw_diff_temp

            if 'Te' in record_data_temp.keys(): Te_inOriFrame = record_data_temp.loc[:,['Te']].copy(deep=True)
            if 'Ne' in record_data_temp.keys(): Ne_inOriFrame = record_data_temp.loc[:,['Ne']].copy(deep=True)
            if 'Ta' in record_data_temp.keys(): Ta_inOriFrame = record_data_temp.loc[:,['Ta']].copy(deep=True)
            if 'Na' in record_data_temp.keys(): Na_inOriFrame = record_data_temp.loc[:,['Na']].copy(deep=True)

            # If there is any NaN in Data_OriFrame, try to interpolate.
            if B_inOriFrame.isnull().values.sum():
                B_inOriFrame_copy = B_inOriFrame.copy(deep=True)
                B_inOriFrame_copy.interpolate(method='time', limit=3, inplace=True)
                B_inOriFrame_copy.bfill(inplace=True)
                B_inOriFrame_copy.ffill(inplace=True)
                if B_inOriFrame_copy.isnull().values.sum():
                    print('Too many NaNs in B. Skip this record. If this situation happens, please check.')
                    continue
                else:
                    B_inOriFrame = B_inOriFrame_copy

            if Vsw_inOriFrame.isnull().values.sum():
                Vsw_inOriFrame_copy = Vsw_inOriFrame.copy(deep=True)
                Vsw_inOriFrame_copy.interpolate(method='time', limit=3, inplace=True)
                Vsw_inOriFrame_copy.bfill(inplace=True)
                Vsw_inOriFrame_copy.ffill(inplace=True)
                if Vsw_inOriFrame_copy.isnull().values.sum():
                    print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
                    continue
                else:
                    Vsw_inOriFrame = Vsw_inOriFrame_copy
                    
            # If there is any NaN in Np_inOriFrame, try to interpolate.
            if Np_inOriFrame.isnull().values.sum():
                Np_inOriFrame_copy = Np_inOriFrame.copy(deep=True)
                Np_inOriFrame_copy.interpolate(method='time', limit=3, inplace=True)
                Np_inOriFrame_copy.bfill(inplace=True)
                Np_inOriFrame_copy.ffill(inplace=True)
                if Np_inOriFrame_copy.isnull().values.sum():
                    print('Too many NaNs in Np. Skip this record. If this situation happens, please check.')
                    continue
                else:
                    Np_inOriFrame = Np_inOriFrame_copy

            # If there is any NaN in Te_inOriFrame, try to interpolate.
            if ('Te' in record_data_temp.keys()) & includeTe:
                if Te_inOriFrame.isnull().values.sum():
                    Te_inOriFrame_copy = Te_inOriFrame.copy(deep=True)
                    Te_inOriFrame_copy.interpolate(method='time', limit=3, inplace=True)
                    Te_inOriFrame_copy.bfill(inplace=True)
                    Te_inOriFrame_copy.ffill(inplace=True)
                    if Te_inOriFrame_copy.isnull().values.sum():
                        print('Too many NaNs in Tp. Skip this record. If this situation happens, please check.')
                        continue
                    else:
                        Te_inOriFrame = Te_inOriFrame_copy

            # If there is any NaN in Ne_inOriFrame, try to interpolate.
            if ('Ne' in record_data_temp.keys()) & includeNe:
                if Ne_inOriFrame.isnull().values.sum():
                    Ne_inOriFrame_copy = Ne_inOriFrame.copy(deep=True)
                    Ne_inOriFrame_copy.interpolate(method='time', limit=3, inplace=True)
                    Ne_inOriFrame_copy.bfill(inplace=True)
                    Ne_inOriFrame_copy.ffill(inplace=True)
                    if Ne_inOriFrame_copy.isnull().values.sum():
                        print('Too many NaNs in Ne. Skip this record. If this situation happens, please check.')
                        continue
                    else:
                        Ne_inOriFrame = Ne_inOriFrame_copy

            # If there is any NaN in Ta_inOriFrame, try to interpolate.
            if 'Ta' in record_data_temp.keys():
                if Ta_inOriFrame.isnull().values.sum():
                    Ta_inOriFrame_copy = Ta_inOriFrame.copy(deep=True)
                    Ta_inOriFrame_copy.interpolate(method='time', limit=3, inplace=True)
                    Ta_inOriFrame_copy.bfill(inplace=True)
                    Ta_inOriFrame_copy.ffill(inplace=True)
                    if Ta_inOriFrame_copy.isnull().values.sum():
                        print('Too many NaNs in Ta.')
                        # continue
                    else:
                        Ta_inOriFrame = Ta_inOriFrame_copy

            # If there is any NaN in Na_inOriFrame, try to interpolate.
            if 'Na' in record_data_temp.keys():
                if Na_inOriFrame.isnull().values.sum():
                    Na_inOriFrame_copy = Na_inOriFrame.copy(deep=True)
                    Na_inOriFrame_copy.interpolate(method='time', limit=3, inplace=True)
                    Na_inOriFrame_copy.bfill(inplace=True)
                    Na_inOriFrame_copy.ffill(inplace=True)
                    if Na_inOriFrame_copy.isnull().values.sum():
                        print('Too many NaNs in NA.')
                        # continue
                    else:
                        Na_inOriFrame = Na_inOriFrame_copy

            # Transform parameters into the FR frame
            matrix_transToFluxRopeFrame = angle2matrix(theta_deg, phi_deg, np.array(VHT_inOriFrame))
            B_inFR = B_inOriFrame.dot(matrix_transToFluxRopeFrame)
            # Calculate the remaining flow velocity
            VHT_inFR = VHT_inOriFrame.dot(matrix_transToFluxRopeFrame)
            Vsw_inFR = Vsw_inOriFrame.dot(matrix_transToFluxRopeFrame)
            V_remaining = np.array(Vsw_inFR - VHT_inFR)
            V_remaining_1D = np.reshape(V_remaining, V_remaining.size)
            # Calculate the Alfven velocity
            P_massDensity = record_data_temp['Np'] * m_proton * 1e6 # In kg/m^3.
            len_P_massDensity = len(P_massDensity)
            P_massDensity_array = np.array(P_massDensity)
            P_massDensity_array = np.reshape(P_massDensity_array, (len_P_massDensity, 1))
            B_inFR = B_inOriFrame.dot(matrix_transToFluxRopeFrame)
            VA_inFR = np.array(B_inFR * 1e-9) / np.sqrt(mu0 * P_massDensity_array) / 1000.0
            VA_inFR_1D = np.reshape(VA_inFR, VA_inFR.size)
            MachNumAvg = np.mean(np.sqrt(np.square(V_remaining).sum(axis=1))/np.sqrt(np.square(VA_inFR).sum(axis=1)))   
            AlphaMach = MachNumAvg**2
            B_norm_DF = pd.DataFrame(np.sqrt(np.square(B_inOriFrame).sum(axis=1)),columns=['|B|'])

            # Calculate the Walen Test slope, intercept, and correlation coefficient
            walenTest_slope, walenTest_intercept, walenTest_r_value = walenTest(VA_inFR_1D, V_remaining_1D)
            eventList_DF_3_CheckWalenTest_temp.loc[index, 'r'] = round(walenTest_r_value, 8) # r.
            eventList_DF_3_CheckWalenTest_temp.loc[index, 'k'] = round(walenTest_slope, 8) # k.
            eventList_DF_3_CheckWalenTest_temp.loc[index, 'MA'] = MachNumAvg

            # Calculate parameters to prepare for Step 4
            if ((spacecraftID == 'PSP') or (spacecraftID == 'SOLARORBITER')) & (len(duration_str_temp) >= 10):
                ds = - VHT_inFR[0] * 1000.0 * dt_2nd
            else:
                ds = - VHT_inFR[0] * 1000.0 * dt 

            A = integrate.cumtrapz(-(1 - AlphaMach) * B_inFR[1]*1e-9, dx=ds, initial=0)

            if includeNe & includeTe:
                Ppe = np.array(Np_inOriFrame['Np']) * 1e6 * k_Boltzmann * 1e9 * np.array(Tp_inOriFrame['Tp'])\
                + np.array(Ne_inOriFrame['Ne']) * 1e6 * k_Boltzmann * 1e9 * np.array(Te_inOriFrame['Te'])
            elif includeTe & (not includeNe):
                Ppe = np.array(Np_inOriFrame['Np']) * 1e6 * k_Boltzmann * 1e9 * (np.array(Tp_inOriFrame['Tp']) + np.array(Te_inOriFrame['Te']))
            else:
                Ppe = np.array(Np_inOriFrame['Np']) * 1e6 * k_Boltzmann * 1e9 * np.array(Tp_inOriFrame['Tp'])

            Pb = np.array((B_inFR[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
            PB = np.array((B_norm_DF['|B|'] * 1e-9)**2 / (2.0*mu0) * 1e9)
            Pt = ((1 - AlphaMach)**2) * Pb + (1 - AlphaMach) * Ppe + (AlphaMach * (1 - AlphaMach)) * PB

            # Find the index of turnPoint.
            # Split A and Pt into two branches.
            index_turnTime = B_inFR.index.get_loc(fluxRopeTurnTime)
            A_sub1 = A[:index_turnTime+1]
            A_sub2 = A[index_turnTime:]
            Pt_sub1 = Pt[:index_turnTime+1]
            Pt_sub2 = Pt[index_turnTime:]
            
            A_tail1 = A[0]
            A_tail2 = A[-1]
            A_turn_point = A[index_turnTime]
            Pt_turn_point = Pt[index_turnTime]
            A_max = max(A)
            A_min = min(A)

            # Debug.
            if 0:
                print('Debug:')
                print('fluxRopeStartTime = {}'.format(fluxRopeStartTime))
                print('fluxRopeTurnTime = {}'.format(fluxRopeTurnTime))
                print('fluxRopeEndTime = {}'.format(fluxRopeEndTime))
                print('theta_deg, phi_deg = {}'.format(theta_deg, phi_deg))
                print('Residue_diff = {}'.format(Residue_diff))
                print('Residue_fit = {}'.format(Residue_fit))
                print('VHT_inOriFrame = {}'.format(VHT_inOriFrame))
                print('A_turn_point = {}'.format(A_turn_point))
                print('A_tail1 = {}'.format(A_tail1))
                print('A_tail2 = {}'.format(A_tail2))
            
            # Find the tail of A, the head is turn point.
            if (A_turn_point > A_tail1)and(A_turn_point > A_tail2):
                A_tail = min(A_tail1, A_tail2)
            elif (A_turn_point < A_tail1)and(A_turn_point < A_tail2):
                A_tail = max(A_tail1, A_tail2)
            else:
                # print('No double-folding, discard this record. When happen, please check!')
                continue
            
            z = np.polyfit(A, Pt, 3)
            Func_Pt_A = np.poly1d(z)
            A_turn_fit = A_turn_point
            Pt_turn_fit = Func_Pt_A(A_turn_point)

            # Calculate std of the residual.
            Func_Pt_A_value = Func_Pt_A(A)
            max_Func_Pt_A_value = max(Func_Pt_A_value)
            min_Func_Pt_A_value = min(Func_Pt_A_value)
            Pt_fit_std = np.std(Pt - Func_Pt_A(A))/(max_Func_Pt_A_value - min_Func_Pt_A_value)

            # Set flag.
            # Checking whether the measurement result is close to fitting result
            # Checking whether the two A values at tail are close
            # Checking whether the curve is flat
            lowTail = ((Func_Pt_A(A_tail) - min_Func_Pt_A_value)/(max_Func_Pt_A_value - min_Func_Pt_A_value)) < max_tailPercentile
            lowTailDiff = abs(A_tail1 - A_tail2)/(A_max - A_min) < max_tailDiff
            lowStd = Pt_fit_std <= max_PtFitStd 
            eventList_DF_3_CheckWalenTest_temp.loc[index, 'lowTail'] = lowTail
            eventList_DF_3_CheckWalenTest_temp.loc[index, 'lowTailDiff'] = lowTailDiff
            eventList_DF_3_CheckWalenTest_temp.loc[index, 'lowStd'] = lowStd
            if SettingBLimit: 
                Bthrehold = np.mean(B_norm_DF['|B|']) >= B_mag_threshold
                eventList_DF_3_CheckWalenTest_temp.loc[index, 'Bthreshold'] = Bthrehold

            # Debug.
            if 0:
                plt.plot(A_turn_fit, Pt_turn_fit, 'g^-')
                plt.plot(A_sub1, Pt_sub1, 'ro-', A_sub2, Pt_sub2, 'bo-', np.sort(A), Func_Pt_A(np.sort(A)),'g--')
                plt.plot(A_turn_fit, Pt_turn_fit, 'g^-')
                plt.title('diff={},  fit={}, std={} \n{}~{} lowTail={},lowTailDiff={}'.format(Residue_diff, Residue_fit, Pt_fit_std,  fluxRopeStartTime, fluxRopeEndTime, lowTail, lowTailDiff))
                plt.show()

        # Remove the records with |k| > 0.3 & |r| < 0.8 & MA > 0.9.
        # In other words, for records with |k| > 0.3, we will keep those with |r| >= 0.8 & MA <= 0.9.
        # Meanwhile, keep the records with |k| < 0.3         
        eventList_DF_3_CheckWalenTest_temp_FRFF = eventList_DF_3_CheckWalenTest_temp[
        ((abs(eventList_DF_3_CheckWalenTest_temp['k']) > 0.300) & (abs(eventList_DF_3_CheckWalenTest_temp['k']) < walenTest_k_threshold)
         & (abs(eventList_DF_3_CheckWalenTest_temp['r']) >= walenTest_r_threshold) 
         & (eventList_DF_3_CheckWalenTest_temp['MA'] <= 0.900))]
        # eventList_DF_3_CheckWalenTest_temp = eventList_DF_3_CheckWalenTest_temp[
        # (abs(eventList_DF_3_CheckWalenTest_temp['k']) < 0.300)].append(eventList_DF_3_CheckWalenTest_temp_FRFF, ignore_index=True)
        eventList_DF_3_CheckWalenTest_temp = pd.concat([eventList_DF_3_CheckWalenTest_temp[
            (abs(eventList_DF_3_CheckWalenTest_temp['k']) <= 0.300)],eventList_DF_3_CheckWalenTest_temp_FRFF], axis=0)
        # Drop 'r','k', 'MA', and 'topTurn'.
        eventList_DF_3_CheckWalenTest_temp = eventList_DF_3_CheckWalenTest_temp.drop(['r','k','MA','topTurn'], axis=1)
        eventList_DF_3_CheckWalenTest_temp = eventList_DF_3_CheckWalenTest_temp.sort_values(by='startTime')
        eventList_DF_3_CheckWalenTest_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_2_RemoveShock_temp),len(eventList_DF_3_CheckWalenTest_temp)))
        # print(eventList_DF_3_CheckWalenTest_temp)
        # ======================================================== S T E P. 4 ===========================================================
        # 4) Check point 4.
        if eventList_DF_3_CheckWalenTest_temp.empty:
            print('\nDataFrame after the Walen Test is empty!')
            print('Go the the next iteration!')
            continue

        print('\nStep 4. Cleaning candidates with bad fitting curve.')
        # Clean the records with bad fitting curve shape.
        eventList_DF_4_CheckFittingCurve_temp = eventList_DF_3_CheckWalenTest_temp.copy()
        keepMask = (eventList_DF_4_CheckFittingCurve_temp['lowTail']&
            eventList_DF_4_CheckFittingCurve_temp['lowTailDiff']&
            eventList_DF_4_CheckFittingCurve_temp['lowStd'])
        if SettingBLimit: keepMask = (eventList_DF_4_CheckFittingCurve_temp['lowTail']&
            eventList_DF_4_CheckFittingCurve_temp['lowTailDiff']&
            eventList_DF_4_CheckFittingCurve_temp['lowStd']&eventList_DF_4_CheckFittingCurve_temp['Bthreshold'])
        eventList_DF_4_CheckFittingCurve_temp = eventList_DF_4_CheckFittingCurve_temp[keepMask]
        eventList_DF_4_CheckFittingCurve_temp.reset_index(drop=True, inplace=True)        
        eventList_DF_4_CheckFittingCurve_temp = eventList_DF_4_CheckFittingCurve_temp.drop(['lowTail','lowTailDiff','lowStd'], axis=1)
        if SettingBLimit: eventList_DF_4_CheckFittingCurve_temp = eventList_DF_4_CheckFittingCurve_temp.drop('Bthreshold', axis=1)
        eventList_DF_4_CheckFittingCurve_temp = eventList_DF_4_CheckFittingCurve_temp.sort_values(by='startTime')
        eventList_DF_4_CheckFittingCurve_temp.reset_index(drop=True, inplace=True)

        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_3_CheckWalenTest_temp),len(eventList_DF_4_CheckFittingCurve_temp)))
        # print(eventList_DF_4_CheckFittingCurve_temp)

        # ======================================================== S T E P. 5===========================================================
        # 5) Check point 5.
        if eventList_DF_4_CheckFittingCurve_temp.empty:
            print('\nDataFrame after checking fitting curve is empty!')
            print('Go the the next iteration!')
            continue

        print('\nStep 5. Removing candidates with large Vsw fluctuations, Vsw_std > {} or Vsw_diff > {}.'.format(Vsw_std_threshold, Vsw_diff_threshold))
        eventList_DF_5_CheckDiscontinuity_temp = eventList_DF_4_CheckFittingCurve_temp.copy()
        mask_toBeRemoved = (eventList_DF_5_CheckDiscontinuity_temp['Vsw_std']>Vsw_std_threshold)|(eventList_DF_5_CheckDiscontinuity_temp['Vsw_diff']>Vsw_diff_threshold)
        eventList_DF_5_CheckDiscontinuity_temp = eventList_DF_5_CheckDiscontinuity_temp[~mask_toBeRemoved]
        eventList_DF_5_CheckDiscontinuity_temp = eventList_DF_5_CheckDiscontinuity_temp.drop(['Vsw_std','Vsw_diff'], axis=1)
        eventList_DF_5_CheckDiscontinuity_temp.reset_index(drop=True, inplace=True)

        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_4_CheckFittingCurve_temp),len(eventList_DF_5_CheckDiscontinuity_temp)))
        # print(eventList_DF_5_CheckDiscontinuity_temp)
        # ======================================================== S T E P. 6 ===========================================================
        # 6) Check point 6.
        if eventList_DF_5_CheckDiscontinuity_temp.empty:
            print('\nDataFrame after removing discontinuity is empty!')
            print('Go the the next iteration!')
            continue

        # Clean up the records with same turnTime.
        print('\nStep 6. Combining candidates with same turnTime (saving the one with the longest duration & smallest R_dif).')
        # Sort by turnTime.
        eventList_DF_5_CheckDiscontinuity_temp = eventList_DF_5_CheckDiscontinuity_temp.sort_values(by='turnTime')
        # Group by turnTime.
        # If candidates have the same turn time, keep the one with the longest duration
        index_min_Residue_diff_inGrouped = eventList_DF_5_CheckDiscontinuity_temp.groupby(['turnTime'], sort=False)['duration'].transform(max) == eventList_DF_5_CheckDiscontinuity_temp['duration']
        eventList_DF_6_SameTurnTime_temp1 = eventList_DF_5_CheckDiscontinuity_temp[index_min_Residue_diff_inGrouped]
        # If there are still candidates having the same turn time, keep the one with the smallest Residue_diff
        index_min_Residue_diff_inGrouped1 = eventList_DF_6_SameTurnTime_temp1.groupby(['turnTime'], sort=False)['residue_diff'].transform(min) == eventList_DF_6_SameTurnTime_temp1['residue_diff']
        eventList_DF_6_SameTurnTime_temp = eventList_DF_6_SameTurnTime_temp1[index_min_Residue_diff_inGrouped1]
        # Reset index
        eventList_DF_6_SameTurnTime_temp = eventList_DF_6_SameTurnTime_temp.sort_values(by='startTime')
        eventList_DF_6_SameTurnTime_temp.reset_index(drop=True, inplace=True)
        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_5_CheckDiscontinuity_temp),len(eventList_DF_6_SameTurnTime_temp)))

        # ======================================================== S T E P. 7 ===========================================================
        # ========================================================  Option 1  ===========================================================
        # Allow the final list has overlapping intervals
        # Append candidates after Step 6 here and go to the next itertation
        if allowOverlap:
            print('\nStep 7. Appending all events.')
            eventList_DF_9_CheckOverlap2_temp = eventList_DF_6_SameTurnTime_temp.copy()
            eventList_DF_9_CheckOverlap2_temp.drop('keepFlag', axis=1, inplace=True)
            eventList_DF_9_CheckOverlap2_temp.reset_index(drop=True, inplace=True)

            # eventList_DF_noOverlap = eventList_DF_noOverlap.append(eventList_DF_9_CheckOverlap2_temp, ignore_index=True)
            eventList_DF_noOverlap = pd.concat([eventList_DF_noOverlap,eventList_DF_9_CheckOverlap2_temp], axis=0)
            eventList_DF_noOverlap.sort_values(by='startTime', inplace=True) 
            print('Go the the next iteration!')
            continue

        # ======================================================== S T E P. 7 ===========================================================
        # ========================================================  Option 2  ===========================================================
        # Continue to handle the overlapping intervals
        # Check overlapping events and keep the one with the minimum residue_diff
        # This is done by checking whether (midpoint[i+1]-midpoint[i]) < 0.5*(length[i]+length[i+1]), i.e., average lengths
        # No need to check whether eventList_DF_6_SameTurnTime_temp is empty.
        # If eventList_DF_5_CheckFittingCurve_temp is not empty, eventList_DF_6_SameTurnTime_temp cannot be empty.
        print('\nStep 7. Removing overlapping candidates and keep the one with minimum residue.')

        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_6_SameTurnTime_temp.copy() # Default is deep copy.
        # eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp.drop(['residue_fit','theta_phi','VHT'], axis=1)
        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp.assign(midpoints = len(eventList_DF_7_AdjacentTurnTime_temp)*[np.nan])

        for index, FR_record in eventList_DF_7_AdjacentTurnTime_temp.iterrows():           
            startTime, endTime = FR_record['startTime'], FR_record['endTime']
            eventList_DF_7_AdjacentTurnTime_temp.loc[index, 'midpoints'] = startTime + (endTime - startTime)*0.5
        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp.assign(midpointsdiff=eventList_DF_7_AdjacentTurnTime_temp['midpoints'].diff())
        
        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp.assign(avg2lens = len(eventList_DF_7_AdjacentTurnTime_temp)*[np.nan])
        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp.assign(midpDiff_avgLens = len(eventList_DF_7_AdjacentTurnTime_temp)*[np.nan])
        
        # Calculate the average lengths of intervals[i] & [i+1]
        index = 0
        while(index < len(eventList_DF_7_AdjacentTurnTime_temp)-1):
            eventList_DF_7_AdjacentTurnTime_temp.loc[index+1, 'avg2lens'] = 0.5 * (eventList_DF_7_AdjacentTurnTime_temp.loc[index, 'endTime'] \
                - eventList_DF_7_AdjacentTurnTime_temp.loc[index, 'startTime']\
                + eventList_DF_7_AdjacentTurnTime_temp.loc[index+1, 'endTime']\
                - eventList_DF_7_AdjacentTurnTime_temp.loc[index+1, 'startTime'])
            index += 1
 
        # Check the overlapping: (midpoint[i+1]-midpoint[i]) < 0.5*(length[i]+length[i+1])
        index = 1
        while(index < len(eventList_DF_7_AdjacentTurnTime_temp)):
            eventList_DF_7_AdjacentTurnTime_temp.loc[index, 'midpDiff_avgLens'] = eventList_DF_7_AdjacentTurnTime_temp.loc[index, 'midpointsdiff'] - eventList_DF_7_AdjacentTurnTime_temp.loc[index, 'avg2lens'] 
            index += 1

        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp.drop(['midpointsdiff','avg2lens','midpoints'], axis=1)

        # Now keep the record with the minimum difference residue out of all overlapping candidates
        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp.assign(keepFlag=[True]*len(eventList_DF_7_AdjacentTurnTime_temp))
        index_column_keepFlag = eventList_DF_7_AdjacentTurnTime_temp.columns.get_loc('keepFlag')
        # If-else handles the situation that search results come from two different resolutions
        i_index = 1
        while(i_index < len(eventList_DF_7_AdjacentTurnTime_temp)):
            if(eventList_DF_7_AdjacentTurnTime_temp['midpDiff_avgLens'].iloc[i_index] <= timedelta(seconds=0)):
                cluster_begin_temp = i_index - 1
                while((eventList_DF_7_AdjacentTurnTime_temp['midpDiff_avgLens'].iloc[i_index] <= timedelta(seconds=0))):
                    i_index += 1
                    if (i_index > len(eventList_DF_7_AdjacentTurnTime_temp)-1):
                        break
                cluster_end_temp = i_index - 1
                min_residue_diff_index_temp = eventList_DF_7_AdjacentTurnTime_temp['residue_diff'].iloc[cluster_begin_temp:cluster_end_temp+1].idxmin()
                eventList_DF_7_AdjacentTurnTime_temp.iloc[cluster_begin_temp:cluster_end_temp+1, index_column_keepFlag] = False
                eventList_DF_7_AdjacentTurnTime_temp.loc[min_residue_diff_index_temp, 'keepFlag'] = True
            else:
                eventList_DF_7_AdjacentTurnTime_temp.iloc[i_index, index_column_keepFlag] = True
                i_index += 1

        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp[eventList_DF_7_AdjacentTurnTime_temp['keepFlag']==True]
        eventList_DF_7_AdjacentTurnTime_temp.reset_index(drop=True, inplace=True)
        eventList_DF_7_AdjacentTurnTime_temp = eventList_DF_7_AdjacentTurnTime_temp.drop('midpDiff_avgLens', axis=1)
        eventList_DF_7_AdjacentTurnTime_temp.reset_index(drop=True, inplace=True)

        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_6_SameTurnTime_temp),len(eventList_DF_7_AdjacentTurnTime_temp)))
        
        # ======================================================== S T E P. 8 ===========================================================
        # 7) Check point 7.
        if eventList_DF_7_AdjacentTurnTime_temp.empty:
            print('\nDataFrame after combining events with adjacent turning time is empty!')
            print('Go the the next iteration!')
            continue

        print('\nStep 8. Removing overlapped candidates by taking the available slots.')
        slotList_DF_temp = pd.DataFrame(columns=['slotStart', 'slotEnd'])
        if eventList_DF_noOverlap.empty:
            # Create slot.
            oneSlot_temp = pd.DataFrame(OrderedDict((('slotStart', [datetimeStart]),('slotEnd', [datetimeEnd]))))
            # Append this slot to slotList_DF_temp.
            slotList_DF_temp = pd.concat([slotList_DF_temp,oneSlot_temp], axis=0)
            # slotList_DF_temp = slotList_DF_temp.append(oneSlot_temp, ignore_index=True)
        else: 
            # Add first slot: [datetimeStart:eventList_DF_noOverlap.iloc[0]['startTime']].
            if datetimeStart<eventList_DF_noOverlap.iloc[0]['startTime']:
                # An OrderedDict is a dictionary subclass that remembers the order in which its contents are added.
                oneSlot_temp = pd.DataFrame(OrderedDict((('slotStart', [datetimeStart]),('slotEnd', [eventList_DF_noOverlap.iloc[0]['startTime']]))))
                # Append first slot to slotList_DF_temp.
                # slotList_DF_temp = slotList_DF_temp.append(oneSlot_temp, ignore_index=True)
                slotList_DF_temp = pd.concat([slotList_DF_temp,oneSlot_temp], axis=0)
            # Add last slot: [eventList_DF_noOverlap.iloc[-1]['endTime'] : datetimeEnd]
            if datetimeEnd>eventList_DF_noOverlap.iloc[-1]['endTime']:
                oneSlot_temp = pd.DataFrame(OrderedDict((('slotStart', [eventList_DF_noOverlap.iloc[-1]['endTime']]),('slotEnd', [datetimeEnd]))))
                # Append last slot to slotList_DF_temp.
                # slotList_DF_temp = slotList_DF_temp.append(oneSlot_temp, ignore_index=True)
                slotList_DF_temp = pd.concat([slotList_DF_temp,oneSlot_temp], axis=0)
            # If eventList_DF_noOverlap has more than one record, add other slots besides first one and last one.
            if len(eventList_DF_noOverlap)>1:
                # Get slots from eventList_DF_noOverlap.
                multiSlot_temp = pd.DataFrame(OrderedDict((('slotStart', list(eventList_DF_noOverlap.iloc[:-1]['endTime'])), ('slotEnd', list(eventList_DF_noOverlap.iloc[1:]['startTime'])))))
                # Append these slots to slotList_DF_temp.
                # slotList_DF_temp = slotList_DF_temp.append(multiSlot_temp, ignore_index=True)
                slotList_DF_temp = pd.concat([slotList_DF_temp,multiSlot_temp], axis=0)
            # Sort slotList_DF_temp by either slotStart or slotEnd. Because there is no overlap, both method are equivalent.
            slotList_DF_temp = slotList_DF_temp.sort_values(by='slotStart')

        # Reset index.
        slotList_DF_temp.reset_index(drop=True, inplace=True)
        eventList_DF_8_CheckOverlap2_temp = eventList_DF_7_AdjacentTurnTime_temp.copy()
        eventList_DF_8_CheckOverlap2_temp = eventList_DF_8_CheckOverlap2_temp.assign(keepFlag2=[False]*len(eventList_DF_8_CheckOverlap2_temp))
        for index, oneSlot_temp in slotList_DF_temp.iterrows():
            keepMask = (eventList_DF_8_CheckOverlap2_temp['startTime']>=oneSlot_temp['slotStart'])&(eventList_DF_8_CheckOverlap2_temp['endTime']<=oneSlot_temp['slotEnd'])
            if(keepMask.sum()>0):
                eventList_DF_8_CheckOverlap2_temp.loc[eventList_DF_8_CheckOverlap2_temp[keepMask].index, 'keepFlag2'] = True
        eventList_DF_8_CheckOverlap2_temp = eventList_DF_8_CheckOverlap2_temp[eventList_DF_8_CheckOverlap2_temp['keepFlag2']==True]
        eventList_DF_8_CheckOverlap2_temp.reset_index(drop=True, inplace=True)

        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_7_AdjacentTurnTime_temp),len(eventList_DF_8_CheckOverlap2_temp)))
        # ======================================================== S T E P. 9 ===========================================================
        # 8) Check point 8.
        if eventList_DF_8_CheckOverlap2_temp.empty:
            print('\nDataFrame after fitting into available slots is empty!')
            print('Go the the next iteration!')
            continue

        print('\nStep 9. Checking overlapped candidates again and appending all events.')
        # Clean other overlapped events, with turnTime difference longer than 5 minutes.
        eventList_DF_8_CheckOverlap2_temp = eventList_DF_8_CheckOverlap2_temp.sort_values(by='endTime')
        # Make a copy.
        eventList_DF_9_CheckOverlap2_temp = eventList_DF_8_CheckOverlap2_temp.copy()
        # Use the interval scheduling greedy algorithm to remove the overlapes.
        eventList_DF_9_CheckOverlap2_temp.drop(['keepFlag','keepFlag2'], axis=1, inplace=True)
        # Sort by endTime.
        eventList_DF_9_CheckOverlap2_temp.sort_values(by='endTime', inplace=True)
        # Reset index.
        eventList_DF_9_CheckOverlap2_temp.reset_index(drop=True, inplace=True)

        # Debug.
        if 0:
            print('eventList_DF_9_CheckOverlap2_temp:')
            print(eventList_DF_9_CheckOverlap2_temp)
        # print(eventList_DF_9_CheckOverlap2_temp)
        # Remove overlap using greedy algorithm.
        count_scheduled_record = 0
        while len(eventList_DF_9_CheckOverlap2_temp) != 0:
            # Find all records overlap with the first one, including itself.
            # Get end time of first one.
            endTime_temp = eventList_DF_9_CheckOverlap2_temp['endTime'].iloc[0]
            # Find the index of all overlapped records, including itself.
            index_df_overlap_temp = eventList_DF_9_CheckOverlap2_temp['startTime'] < endTime_temp
            # Save the first one into eventList_DF_noOverlap.
            # eventList_DF_noOverlap = eventList_DF_noOverlap.append(eventList_DF_9_CheckOverlap2_temp.iloc[0], ignore_index=True)
            eventList_DF_noOverlap = pd.concat([eventList_DF_noOverlap,eventList_DF_9_CheckOverlap2_temp.iloc[0:1]], axis=0)
            
            # Counter + 1.
            count_scheduled_record += 1
            # Remove all the overlapped records from eventList_DF_9_CheckOverlap2_temp, including itself.
            eventList_DF_9_CheckOverlap2_temp = eventList_DF_9_CheckOverlap2_temp[~index_df_overlap_temp]
        # print(eventList_DF_noOverlap)
        if isVerbose:
            print('The total records before vs after this step: {} vs {}.'.format(len(eventList_DF_8_CheckOverlap2_temp),count_scheduled_record))
        # Sort total records list by endTime.
        eventList_DF_noOverlap.sort_values(by='endTime', inplace=True)

        if isPrintIntermediateDF:
            print('\neventList_DF_0_original_temp:')
            print(eventList_DF_0_original_temp)
            print('\neventList_DF_1_CheckResidue_temp:')
            print(eventList_DF_1_CheckResidue_temp)
            print('\neventList_DF_2_RemoveShock_temp:')
            print(eventList_DF_2_RemoveShock_temp)
            print('\neventList_DF_3_CheckWalenTest_temp:')
            print(eventList_DF_3_CheckWalenTest_temp)
            print('\neventList_DF_4_CheckFittingCurve_temp:')
            print(eventList_DF_4_CheckFittingCurve_temp)
            print('\neventList_DF_5_CheckDiscontinuity_temp:')
            print(eventList_DF_5_CheckDiscontinuity_temp)
            print('\neventList_DF_6_SameTurnTime_temp:')
            print(eventList_DF_6_SameTurnTime_temp)
            print('\neventList_DF_7_AdjacentTurnTime_temp:')
            print(eventList_DF_7_AdjacentTurnTime_temp)
            print('\neventList_DF_8_CheckOverlap1_temp:')
            print(eventList_DF_8_CheckOverlap1_temp)
            print('\neventList_DF_9_CheckOverlap2_temp:')
            print(eventList_DF_9_CheckOverlap2_temp)
            print('\neventList_DF_noOverlap:')
            print(eventList_DF_noOverlap)

        # =================================================== GO TO NEXT WINDOW ========================================================
    
    # Reset index.
    eventList_DF_noOverlap.reset_index(drop=True, inplace=True)
    # print(eventList_DF_noOverlap)
    # Save DataFrame to pickle file.
    print('\nSaving eventList_DF_no_overlap to pickle file...')
    # If plotFolder does not exist, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eventList_DF_noOverlap.to_pickle(output_dir + '/' + output_filename + '.p')
    print('Done.')

    return eventList_DF_noOverlap

#########################################################################
def get_more_flux_rope_info(spacecraftID, data_DF, dataObject_or_dataPath, **kwargs):
    """This function calculates parameters of each flux rope record. 
    Return to a pickle file."""

    # Input content: dataObject_or_dataPath should be the data or path of no overlapped eventlist.
    # Check input datatype:
    # If dataObject_or_dataPath is an object(dict):
    if isinstance(dataObject_or_dataPath, pd.DataFrame):
        print('\nYour input is a DataFrame.')
        search_result_no_overlap_DF = dataObject_or_dataPath
    elif isinstance(dataObject_or_dataPath, str):
        print('\nYour input is a path. Load the dictionary data via this path.')
        search_result_no_overlap_DF = pd.read_pickle(open(dataObject_or_dataPath, 'rb'))
    else:
        print('\nPlease input the correct datatype! The input data must be a DataFrame or the path of a DataFrame!')
        return None
    
    if len(search_result_no_overlap_DF) == 0:
        print("No flux rope after cleaning.")
        print("Done.")
        exit()
    # Set default value.
    # output filename.
    output_filename = 'detailed_info'
    # output dir.
    output_dir = os.getcwd()
    # isVerbose.
    isVerbose = False
    
    print('\nDefault parameters:')
    print('output_dir_file       = {}.'.format(output_dir+output_filename))
    
    # If keyword is specified, overwrite the default value.
    if 'output_dir' in kwargs:
        output_dir = kwargs['output_dir']
        print('output_dir            = {}.'.format(output_dir))
    if 'output_filename' in kwargs:
        output_filename = kwargs['output_filename']
        print('output_filename       = {}.'.format(output_filename))
    if 'isVerbose' in kwargs:
        isVerbose = kwargs['isVerbose']
        print('isVerbose is set to {}'.format())
    if 'the2ndDF' in kwargs:
        the2ndDF = kwargs['the2ndDF']
        print('The 2nd DF is loaded  = True')
    if 'spacecraftID' in kwargs:
        spacecraftID = kwargs['spacecraftID']
        print('spacecraftID          = {}'.format(spacecraftID))
    if spacecraftID == 'ACE' or 'WIND': 
        dt = 60.0
    if spacecraftID == 'ULYSSES': 
        dt = 240.0
    if spacecraftID == 'PSP': 
        dt = 28.0
        if 'the2ndDF' in kwargs:
            the2ndDF = kwargs['the2ndDF']
            dt = 1.0
            dt_2nd = 28.0
    if spacecraftID == 'SOLARORBITER': 
        dt = 28.0
        if 'the2ndDF' in kwargs:
            the2ndDF = kwargs['the2ndDF']
            dt = 4.0
            dt_2nd = 28.0
    if 'dt' in kwargs:
        dt = kwargs['dt']
        print('Time increment dt     = {} seconds'.format(dt))
    else:
        print('Time increment dt     = {} seconds'.format(dt))
        if 'the2ndDF' in kwargs: 
            print('Time increment dt2    = {} seconds'.format(dt_2nd))
    
    # Create an empty dataframe.
    eventList_DF_detailedInfo = pd.DataFrame(columns=['startTime', 'turnTime', 'endTime', 'duration (s)', 'scaleSize (AU)',
        'residue_diff', 'residue_fit', '<|B|> (nT)', 'theta_deg', 'phi_deg', 
        'VHT_inOriFrame[0] (km/s)', 'VHT_inOriFrame[1] (km/s)', 'VHT_inOriFrame[2] (km/s)', 
        'X_unitVector[0]', 'X_unitVector[1]', 'X_unitVector[2]', 
        'Y_unitVector[0]', 'Y_unitVector[1]', 'Y_unitVector[2]', 
        'Z_unitVector[0]', 'Z_unitVector[1]', 'Z_unitVector[2]', 
        'walenTest_slope', 'walenTest_intercept', 'walenTest_r_value', 'MachNumAvg', 
        'walenTest_slope_b4reverse','walenTest_intercept_b4reverse', 'walenTest_r_value_b4reverse',
        'radialDistance (AU)', '<alphaRatio>', 'Jzmax','<VA> (km/s)',
        'crossHelicity_walen','residueEnergy_walen','crossHelicity_dv_db','residueEnergy_dv_db',
        '<|B[0]|> (nT)', '<|B[1]|> (nT)', '<|B[2]|> (nT)','<|Bx_inFR|> (nT)', '<|By_inFR|> (nT)', '<|Bz_inFR|> (nT)', 
        'B_std', 'B0_std', 'B1_std', 'B2_std', 'Bx_inFR_std', 'By_inFR_std', 'Bz_inFR_std', 
        '|B|max (nT)', '<Vsw> (km/s)', '<Beta>', '<protonBeta>', 
        '<Np> (#/cc)', '<Tp> (10^6K)', '<Ne> (#/cc)', '<Te> (10^6K)', '<Na> (#/cc)', '<Ta> (10^6K)', 
        'A_range', 'Am (T m)', 'Pt_coeff', 'Path_length', 
        'lambda1', 'lambda2', 'lambda3', 
        'eigenVectorMaxVar_lambda1[0]', 'eigenVectorMaxVar_lambda1[1]', 'eigenVectorMaxVar_lambda1[2]', 
        'eigenVectorInterVar_lambda2[0]', 'eigenVectorInterVar_lambda2[1]', 'eigenVectorInterVar_lambda2[2]', 
        'eigenVectorMinVar_lambda3[0]', 'eigenVectorMinVar_lambda3[1]', 'eigenVectorMinVar_lambda3[2]', 
        'Vswlambda1', 'Vswlambda2', 'Vswlambda3', 
        'VsweigenVectorMaxVar_lambda1[0]', 'VsweigenVectorMaxVar_lambda1[1]', 'VsweigenVectorMaxVar_lambda1[2]', 
        'VsweigenVectorInterVar_lambda2[0]', 'VsweigenVectorInterVar_lambda2[1]', 'VsweigenVectorInterVar_lambda2[2]', 
        'VsweigenVectorMinVar_lambda3[0]', 'VsweigenVectorMinVar_lambda3[1]', 'VsweigenVectorMinVar_lambda3[2]',
        'Vremaininglambda1', 'Vremaininglambda2', 'Vremaininglambda3', 
        'VremainingeigenVectorMaxVar_lambda1[0]', 'VremainingeigenVectorMaxVar_lambda1[1]', 'VremainingeigenVectorMaxVar_lambda1[2]', 
        'VremainingeigenVectorInterVar_lambda2[0]', 'VremainingeigenVectorInterVar_lambda2[1]', 'VremainingeigenVectorInterVar_lambda2[2]', 
        'VremainingeigenVectorMinVar_lambda3[0]', 'VremainingeigenVectorMinVar_lambda3[1]', 'VremainingeigenVectorMinVar_lambda3[2]'])

    print('\nCalculating detailed information of flux ropes')
    for index_FR in range(len(search_result_no_overlap_DF)):
        print('{}/{}...'.format(index_FR+1, len(search_result_no_overlap_DF)))
        oneEvent = search_result_no_overlap_DF.iloc[index_FR]
        startTime = oneEvent['startTime']
        turnTime  = oneEvent['turnTime']
        endTime  = oneEvent['endTime']
        duration  = oneEvent['duration']
        durationTemp = int((endTime - startTime).total_seconds()/dt)+1
        if 'the2ndDF' in kwargs: 
            durationTemp2 = int((endTime - startTime).total_seconds()/dt_2nd)+1
        durationPhysical = int((endTime - startTime).total_seconds())+1
        residue_diff = oneEvent['residue_diff']
        residue_fit = oneEvent['residue_fit']
        theta_deg, phi_deg = oneEvent['theta_phi']
        VHT_inOriFrame = np.array(oneEvent['VHT'])
        
        # Grab data in specific range.
        if duration == durationTemp:
            selectedRange_mask = (data_DF.index >= startTime) & (data_DF.index <= endTime)
            data_oneFR_DF = data_DF.iloc[selectedRange_mask]
        elif duration == durationTemp2:
            selectedRange_mask = (the2ndDF.index >= startTime) & (the2ndDF.index <= endTime)
            data_oneFR_DF = the2ndDF.iloc[selectedRange_mask]
        
        # Keys: Index([u'Bx', u'By', u'Bz', u'Vx', u'Vy', u'Vz', u'Np', u'Tp', u'Te'], dtype='object')
        # Get data slice.
        B_inOriFrame = data_oneFR_DF.iloc[:,0:3]
        Vsw_inOriFrame = data_oneFR_DF.loc[:,['V0','V1','V2']]
        Np = data_oneFR_DF.loc[:,['Np']]
        Tp = data_oneFR_DF.loc[:,['Tp']]
        RD = data_oneFR_DF.loc[:,['RD']]
        
        if 'alphaRatio' in data_oneFR_DF.keys(): 
            alphaRatio_mean = np.mean(np.ma.masked_invalid(np.array(data_oneFR_DF['alphaRatio'])))
        else:
            alphaRatio_mean = None
        if 'Te' in data_oneFR_DF.keys(): Te = data_oneFR_DF.loc[:,['Te']]
        if 'Ne' in data_oneFR_DF.keys(): Ne = data_oneFR_DF.loc[:,['Ne']]
        if 'Ta' in data_oneFR_DF.keys(): Ta = data_oneFR_DF.loc[:,['Ta']]
        if 'Na' in data_oneFR_DF.keys(): Na = data_oneFR_DF.loc[:,['Na']]
        AlphaDataExist = False
        if ('Ta' in data_oneFR_DF.keys())&('Na' in data_oneFR_DF.keys()): 
            AlphaDataExist = True

        # If there is any NaN in the original data frame, try to interpolate.
        if B_inOriFrame.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate B.')
            B_inOriFrame_copy = B_inOriFrame.copy()
            B_inOriFrame_copy.interpolate(method='time', limit=None, inplace=True)
            B_inOriFrame_copy.bfill(inplace=True)
            B_inOriFrame_copy.ffill(inplace=True)
            if B_inOriFrame_copy.isnull().values.sum():
                print('Too many NaNs in B. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                B_inOriFrame = B_inOriFrame_copy

        if Vsw_inOriFrame.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate Vsw.')
            Vsw_inOriFrame_copy = Vsw_inOriFrame.copy()
            Vsw_inOriFrame_copy.interpolate(method='time', limit=None, inplace=True)
            Vsw_inOriFrame_copy.bfill(inplace=True)
            Vsw_inOriFrame_copy.ffill(inplace=True)
            if Vsw_inOriFrame_copy.isnull().values.sum():
                print('Too many NaNs in Vsw. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                Vsw_inOriFrame = Vsw_inOriFrame_copy
                
        if Np.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate Np.')
            Np_copy = Np.copy()
            Np_copy.interpolate(method='time', limit=None, inplace=True)
            Np_copy.bfill(inplace=True)
            Np_copy.ffill(inplace=True)
            if Np_copy.isnull().values.sum():
                print('Too many NaNs in Np. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                Np = Np_copy

        if Tp.isnull().values.sum():
            if isVerbose:
                print('Found NaNs, interpolate Tp.')
            Tp_copy = Tp.copy()
            Tp_copy.interpolate(method='time', limit=None, iTplace=True)
            Tp_copy.bfill(iTplace=True)
            Tp_copy.ffill(iTplace=True)
            if Tp_copy.isnull().values.sum():
                print('Too many NaNs in Tp. Skip this record. If this situation happens, please check.')
                detailed_info_dict = None
                continue
            else:
                Tp = Tp_copy

        if ('Ne' in data_oneFR_DF.keys()) & includeNe:
            if Ne.isnull().values.sum():
                if isVerbose:
                    print('Found NaNs, interpolate Ne.')
                Ne_copy = Ne.copy()
                Ne_copy.interpolate(method='time', limit=None, inelace=True)
                Ne_copy.bfill(inelace=True)
                Ne_copy.ffill(inelace=True)
                if Ne_copy.isnull().values.sum():
                    print('Too many NaNs in Ne. Skip this record. If this situation happens, please check.')
                    detailed_info_dict = None
                    continue
                else:
                    Ne = Ne_copy

        if ('Te' in data_oneFR_DF.keys()) & includeTe:
            if Te.isnull().values.sum():
                if isVerbose:
                    print('Found NaNs, interpolate Te.')
                Te_copy = Te.copy()
                Te_copy.interpolate(method='time', limit=None, iTelace=True)
                Te_copy.bfill(iTelace=True)
                Te_copy.ffill(iTelace=True)
                if Te_copy.isnull().values.sum():
                    print('Too many NaNs in Te. Skip this record. If this situation happens, please check.')
                    detailed_info_dict = None
                    continue
                else:
                    Te = Te_copy

        # Use direction cosines to construct a unit vector.
        theta_rad = factor_deg2rad * theta_deg
        phi_rad   = factor_deg2rad * phi_deg

        # Form new Z_unitVector according to direction cosines.
        Z_unitVector = np.array([np.sin(theta_rad)*np.cos(phi_rad), np.sin(theta_rad)*np.sin(phi_rad), np.cos(theta_rad)])
        # Find X axis from Z axis and -VHT.
        X_unitVector = findXaxis(Z_unitVector, -VHT_inOriFrame)
        # Find the Y axis to form a right-handed coordinater with X and Z.
        Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)

        # Project B_inOriFrame into FluxRope Frame.
        matrix_transToFluxRopeFrame = angle2matrix(theta_deg, phi_deg, np.array(VHT_inOriFrame))
        B_inFR = B_inOriFrame.dot(matrix_transToFluxRopeFrame)
        VHT_inFR = VHT_inOriFrame.dot(matrix_transToFluxRopeFrame)
        Vsw_inFR = Vsw_inOriFrame.dot(matrix_transToFluxRopeFrame)

        # Apply walen test on the result(in optimal frame).
        # Proton mass density. Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
        P_massDensity = Np * m_proton * 1e6 # In kg/m^3.
        len_P_massDensity = len(P_massDensity)
        P_massDensity_array = np.array(P_massDensity)
        P_massDensity_array = np.reshape(P_massDensity_array, (len_P_massDensity, 1))
                
        # Alfven speed. Multiply by 1e-9 to convert nT to T. Divided by 1000.0 to convert m/s to km/s.
        VA_inFR = np.array(B_inFR * 1e-9) / np.sqrt(mu0 * P_massDensity_array) / 1000.0
        VA_inFR_1D = np.reshape(VA_inFR, VA_inFR.size)
        V_remaining = np.array(Vsw_inFR - VHT_inFR)
        V_remaining_1D = np.reshape(V_remaining, V_remaining.size)
        # Call walen test function.
        walenTest_slope_b4reverse, walenTest_intercept_b4reverse, walenTest_r_value_b4reverse = walenTest(VA_inFR_1D, V_remaining_1D)

        # Check if Bz has negative values, if does, flip Z-axis direction.
        num_Bz_lt0 = (B_inFR[2]<0).sum()
        num_Bz_gt0 = (B_inFR[2]>0).sum()
        # If the negative Bz is more than positive Bz, filp.
        if (num_Bz_lt0 > num_Bz_gt0):
            # Reverse the direction of Z-axis.
            print('Reverse the direction of Z-axis!')
            Z_unitVector = -Z_unitVector
            # Recalculat theta and phi with new Z_unitVector.
            theta_deg, phi_deg = directionVector2angle(Z_unitVector)
            # Refind X axis frome Z axis and -Vsw.
            X_unitVector = findXaxis(Z_unitVector, -VHT_inOriFrame)
            # Refind the Y axis to form a right-handed coordinater with X and Z.
            Y_unitVector = formRighHandFrame(X_unitVector, Z_unitVector)
            # Reproject B_inOriFrame_DataFrame into flux rope (FR) frame.
            matrix_transToFluxRopeFrame = np.array([X_unitVector, Y_unitVector, Z_unitVector]).T
            B_inFR = B_inOriFrame.dot(matrix_transToFluxRopeFrame)


        # Project VHT_inOriFrame into FluxRope Frame.
        VHT_inFR = VHT_inOriFrame.dot(matrix_transToFluxRopeFrame)
        # Project Vsw_inFR into FluxRope Frame.
        Vsw_inFR = Vsw_inOriFrame.dot(matrix_transToFluxRopeFrame)
        # Alfven speed. Multiply by 1e-9 to convert nT to T. Divided by 1000.0 to convert m/s to km/s.
        VA_inFR = np.array(B_inFR * 1e-9) / np.sqrt(mu0 * P_massDensity_array) / 1000.0
        VA_inFR_1D = np.reshape(VA_inFR, VA_inFR.size)
        V_remaining = np.array(Vsw_inFR - VHT_inFR)
        V_remaining_1D = np.reshape(V_remaining, V_remaining.size)
        V_remaining_avg = np.array(Vsw_inFR - np.mean(Vsw_inFR, axis=0))
        # Call walen test function.
        walenTest_slope, walenTest_intercept, walenTest_r_value = walenTest(VA_inFR_1D, V_remaining_1D)

        # Calculate the covariance matrix of Magnetic field.
        covM_B_inOriFrame = B_inOriFrame.cov()
        lambda1, lambda2, lambda3, eigenVectorMaxVar_lambda1, eigenVectorInterVar_lambda2, eigenVectorMinVar_lambda3 = eigenMatrix(covM_B_inOriFrame, formXYZ=True)
        covM_Vsw_inOriFrame = Vsw_inOriFrame.cov()
        Vswlambda1, Vswlambda2, Vswlambda3, VsweigenVectorMaxVar_lambda1, VsweigenVectorInterVar_lambda2, VsweigenVectorMinVar_lambda3 = eigenMatrix(covM_Vsw_inOriFrame, formXYZ=True)
        Vremaining_inOriFrame = Vsw_inOriFrame - VHT_inOriFrame
        covM_Vremaining_inOriFrame = Vremaining_inOriFrame.cov()
        Vremaininglambda1, Vremaininglambda2, Vremaininglambda3, VremainingeigenVectorMaxVar_lambda1, VremainingeigenVectorInterVar_lambda2, VremainingeigenVectorMinVar_lambda3 = eigenMatrix(covM_Vremaining_inOriFrame, formXYZ=True)

        size_inMeter = - VHT_inFR[0] * 1000.0 * durationPhysical # Space increment along X axis. Convert km/s to m/s.
        size_inAU = size_inMeter/149597870700

        MachNumAvg = np.mean(np.sqrt(np.square(V_remaining).sum(axis=1))/np.sqrt(np.square(VA_inFR).sum(axis=1)))   
        AlphaMach = MachNumAvg**2
        B_norm_DF = pd.DataFrame(np.sqrt(np.square(B_inOriFrame).sum(axis=1)),columns=['|B|'])
        
        if duration == durationTemp: 
            ds = - VHT_inFR[0] * 1000.0 * dt # Space increment along X axis. Convert km/s to m/s. m/minutes.
        elif duration == durationTemp2: 
            ds = - VHT_inFR[0] * 1000.0 * dt_2nd
        # Calculate A(x,0) by integrating By. A(x,0) = Integrate[-By(s,0)ds, {s, 0, x}], where ds = -Vht dot unit(X) dt.
        A = integrate.cumtrapz(-(1 - AlphaMach) * B_inFR[1]*1e-9, dx=ds, initial=0)
        # Calculate Pt(x,0). Pt(x,0)=p(x,0)+Bz^2/(2mu0). By = B_inFR[2]           
        if includeNe & includeTe:
            Ppe = np.array(Np['Np']) * 1e6 * k_Boltzmann * 1e9 * np.array(Tp['Tp']) \
            + np.array(Ne['Ne']) * 1e6 * k_Boltzmann * 1e9 * np.array(Te['Te'])
        elif includeTe & (not includeNe):
            Ppe = np.array(Np['Np']) * 1e6 * k_Boltzmann * 1e9 * (np.array(Tp['Tp'])+np.array(Te['Te']))
        else:
            Ppe = np.array(Np['Np']) * 1e6 * k_Boltzmann * 1e9 * np.array(Tp['Tp'])
        
        Pb = np.array((B_inFR[2] * 1e-9)**2 / (2.0*mu0) * 1e9) # 1e9 convert unit form pa to npa.
        PB = np.array((B_norm_DF['|B|'] * 1e-9)**2 / (2.0*mu0) * 1e9)
        Pt = ((1 - AlphaMach)**2) * Pb + (1 - AlphaMach) * Ppe + (AlphaMach * (1 - AlphaMach)) * PB
        
        # Find the index of turnPoint.
        index_turnTime = B_inFR.index.get_loc(turnTime)
        # Split A and Pt into two branches.
        A_sub1 = A[:index_turnTime+1]
        A_sub2 = A[index_turnTime:]
        Pt_sub1 = Pt[:index_turnTime+1]
        Pt_sub2 = Pt[index_turnTime:]
        
        z = np.polyfit(A, Pt, 3)
        Func_Pt_A = np.poly1d(z)
        Func_Jz = np.polyder(Func_Pt_A) 
        Pt_coeff = list(z)
        
        A_physical = A / (1-AlphaMach)
        A_range = [min(A_physical), max(A_physical)]
        A_norm = A_range/max(A_physical)
        if abs(min(A_physical)) > abs(max(A_physical)):
            Am = min(A_physical)
        else:
            Am = max(A_physical)
        Path_length = ds * durationPhysical # The lenght of spacecraft trajectory across the flux rope.
        Jz = Func_Jz(A_range)
        Jzmax = np.nanmax(Jz)


        # Subscripts 0, 1, 2 refer to (X, Y, Z) in GSE or (R, T, N) in RTN
        B_magnitude_max = B_norm_DF['|B|'].max(skipna=True)
        B_inOriFrame = pd.concat([B_inOriFrame, B_norm_DF], axis=1)
        B_std_Series = B_inOriFrame.std(axis=0,skipna=True,numeric_only=True)
        B_abs_mean_Series = B_inOriFrame.abs().mean(axis=0,skipna=True,numeric_only=True)
        B_mean_Series = B_inOriFrame.mean(axis=0,skipna=True,numeric_only=True)
        B_abs_mean = round(B_abs_mean_Series['|B|'],4)
        B0_abs_mean = round(B_abs_mean_Series[0],4)
        B1_abs_mean = round(B_abs_mean_Series[1],4)
        B2_abs_mean = round(B_abs_mean_Series[2],4)
        B_std = round(B_std_Series['|B|'],4)
        B0_std = round(B_std_Series[0],4)
        B1_std = round(B_std_Series[1],4)
        B2_std = round(B_std_Series[2],4)
        
        # B_inFR.
        B_inFR_std_Series = B_inFR.std(axis=0,skipna=True,numeric_only=True)
        B_inFR_abs_mean_Series = B_inFR.abs().mean(axis=0,skipna=True,numeric_only=True)
        Bx_inFR_abs_mean = round(B_inFR_abs_mean_Series[0],4)
        By_inFR_abs_mean = round(B_inFR_abs_mean_Series[1],4)
        Bz_inFR_abs_mean = round(B_inFR_abs_mean_Series[2],4)
        Bx_inFR_std = round(B_inFR_std_Series[0],4)
        By_inFR_std = round(B_inFR_std_Series[1],4)
        Bz_inFR_std = round(B_inFR_std_Series[2],4)
        
        if spacecraftID == 'PSP':
            Vsw_inOriFrame_noVsc = data_oneFR_DF.loc[:,['VR_noVsc','VT_noVsc','VN_noVsc']]
            Vsw_norm_DF = pd.DataFrame(np.sqrt(np.square(Vsw_inOriFrame).sum(axis=1)),columns=['|Vsw|'])
            Vsw_magnitude_mean = Vsw_norm_DF['|Vsw|'].mean(skipna=True) 
        else:
            Vsw_norm_DF = pd.DataFrame(np.sqrt(np.square(Vsw_inOriFrame).sum(axis=1)),columns=['|Vsw|'])
            Vsw_magnitude_mean = Vsw_norm_DF['|Vsw|'].mean(skipna=True)   
        VA_mean = np.mean(np.sqrt(np.square(VA_inFR).sum(axis=1)))
        radialDistance = np.mean(np.array(RD['RD']))
        
        # Get Plasma Beta statistical properties.
        Tp_mean = np.mean(np.ma.masked_invalid(np.array(Tp['Tp']))) # Exclude nan and inf.
        Tp_mean = Tp_mean/1e6
        Tp_mean = round(Tp_mean, 6)

        # Original Np is in #/cc ( cc = cubic centimeter).
        Np_mean = float(Np.mean(skipna=True, numeric_only=True))# In #/cc
        # Calculate Te_mean. Divided by 1e6 to convert unit to 10^6K.
        if 'Te' in data_oneFR_DF.keys(): Te_mean = float(Te.mean(skipna=True, numeric_only=True))/1e6
        else: Te_mean = None
        if 'Ta' in data_oneFR_DF.keys(): Ta_mean = float(Ta.mean(skipna=True, numeric_only=True))/1e6
        else: Ta_mean = None
        if 'Ne' in data_oneFR_DF.keys(): Ne_mean = float(Ne.mean(skipna=True, numeric_only=True))/1e6
        else: Ne_mean = None
        if 'Na' in data_oneFR_DF.keys(): Na_mean = float(Na.mean(skipna=True, numeric_only=True))/1e6
        else: Na_mean = None
        
        # Get Other statistical properties.
        # Using the method of the Walen test, ie., Vsw-VHT vs VA
        Bx_inFR_VA = np.array(B_inFR[0] * 1e-9)/np.sqrt(mu0 * np.array(P_massDensity['Np'])) / 1000.0 # km/s
        By_inFR_VA = np.array(B_inFR[1] * 1e-9)/np.sqrt(mu0 * np.array(P_massDensity['Np'])) / 1000.0 # km/s
        Bz_inFR_VA = np.array(B_inFR[2] * 1e-9)/np.sqrt(mu0 * np.array(P_massDensity['Np'])) / 1000.0 # km/s

        dv_db_mean_walen = (V_remaining[:,0]*Bx_inFR_VA+V_remaining[:,1]*By_inFR_VA+V_remaining[:,2]*Bz_inFR_VA).mean()
        dv_sq2_mean_walen = (V_remaining[:,0]**2+V_remaining[:,1]**2+V_remaining[:,2]**2).mean()
        db_sq2_mean_walen = (Bx_inFR_VA**2+By_inFR_VA**2+Bz_inFR_VA**2).mean()
        crossHelicity_walen = 2*dv_db_mean_walen/(dv_sq2_mean_walen+db_sq2_mean_walen)
        residueEnergy_walen = (dv_sq2_mean_walen-db_sq2_mean_walen)/(dv_sq2_mean_walen+db_sq2_mean_walen)

        # Calculate with the dv = V - <V> & db = VA - <VA>
        Bx_inFR_dVA = Bx_inFR_VA - Bx_inFR_VA.mean()
        By_inFR_dVA = By_inFR_VA - By_inFR_VA.mean()
        Bz_inFR_dVA = Bz_inFR_VA - Bz_inFR_VA.mean()

        dv_db_mean = (V_remaining_avg[:,0]*Bx_inFR_dVA+V_remaining_avg[:,1]*By_inFR_dVA+V_remaining_avg[:,2]*Bz_inFR_dVA).mean() 
        dv_sq2_mean = (V_remaining_avg[:,0]**2+V_remaining_avg[:,1]**2+V_remaining_avg[:,2]**2).mean()
        db_sq2_mean = (Bx_inFR_dVA**2+By_inFR_dVA**2+Bz_inFR_dVA**2).mean()
        crossHelicity_dv_db=2*dv_db_mean/(dv_sq2_mean+db_sq2_mean)
        residueEnergy_dv_db=(dv_sq2_mean-db_sq2_mean)/(dv_sq2_mean+db_sq2_mean)

        # Calculate plasma Dynamic Pressure PD.
        # Original Np is in #/cc ( cc = cubic centimeter). Multiply by 1e6 to convert cc to m^3.
        if AlphaDataExist: 
            Pa = np.array(Na['Na']) * 1e6 * k_Boltzmann * 1e9 * np.array(Ta['Ta'])
        else:
            Pa = 0
        # The previous Ppe includes Pe
        Pp = np.array(Np['Np']) * 1e6 * k_Boltzmann * 1e9 * np.array(Tp['Tp'])
        Beta = (Pa + Ppe)/PB  
        Beta_mean = np.mean(np.ma.masked_invalid(Beta)) # Exclude nan and inf.
        Beta_p = Pp/PB
        Beta_p_mean = np.mean(np.ma.masked_invalid(Beta_p))

        detailed_info_dict = {'startTime':startTime, 'turnTime':turnTime, 'endTime':endTime, 
        'duration (s)':durationPhysical, 'scaleSize (AU)':size_inAU,'residue_diff':residue_diff, 'residue_fit':residue_fit, '<|B|> (nT)':B_abs_mean,
        'theta_deg':theta_deg, 'phi_deg':phi_deg, 
        'VHT_inOriFrame[0] (km/s)':VHT_inOriFrame[0], 'VHT_inOriFrame[1] (km/s)':VHT_inOriFrame[1], 'VHT_inOriFrame[2] (km/s)':VHT_inOriFrame[2], 
        'X_unitVector[0]':X_unitVector[0], 'X_unitVector[1]':X_unitVector[1], 'X_unitVector[2]':X_unitVector[2], 
        'Y_unitVector[0]':Y_unitVector[0], 'Y_unitVector[1]':Y_unitVector[1], 'Y_unitVector[2]':Y_unitVector[2], 
        'Z_unitVector[0]':Z_unitVector[0], 'Z_unitVector[1]':Z_unitVector[1], 'Z_unitVector[2]':Z_unitVector[2], 
        'walenTest_slope':walenTest_slope, 'walenTest_intercept':walenTest_intercept, 'walenTest_r_value':walenTest_r_value, 'MachNumAvg':MachNumAvg, 
        'walenTest_slope_b4reverse':walenTest_slope_b4reverse, 'walenTest_intercept_b4reverse':walenTest_intercept_b4reverse, 'walenTest_r_value_b4reverse':walenTest_r_value_b4reverse,
        'radialDistance (AU)':radialDistance, '<alphaRatio>':alphaRatio_mean, 'Jzmax':Jzmax, '<VA> (km/s)':VA_mean,
        'crossHelicity_walen':crossHelicity_walen, 'residueEnergy_walen':residueEnergy_walen,
        'crossHelicity_dv_db':crossHelicity_dv_db, 'residueEnergy_dv_db':residueEnergy_dv_db,
        '<|B[0]|> (nT)':B0_abs_mean, '<|B[1]|> (nT)':B1_abs_mean, '<|B[2]|> (nT)':B2_abs_mean, 
        '<|Bx_inFR|> (nT)':Bx_inFR_abs_mean, '<|By_inFR|> (nT)':By_inFR_abs_mean, '<|Bz_inFR|> (nT)':Bz_inFR_abs_mean, 
        'B_std':B_std, 'B0_std':B0_std, 'B1_std':B1_std, 'B2_std':B2_std, 
        'Bx_inFR_std':Bx_inFR_std, 'By_inFR_std':By_inFR_std, 'Bz_inFR_std':Bz_inFR_std, 
        '|B|max (nT)':B_magnitude_max, '<Vsw> (km/s)':Vsw_magnitude_mean, '<Beta>':Beta_mean, '<protonBeta>':Beta_p_mean,
        '<Np> (#/cc)':Np_mean, '<Tp> (10^6K)':Tp_mean, '<Ne> (#/cc)':Ne_mean, '<Te> (10^6K)':Te_mean, '<Na> (#/cc)':Na_mean, '<Ta> (10^6K)':Ta_mean,  
        'A_range':A_range, 'Am (T m)':Am, 'Pt_coeff':Pt_coeff, 'Path_length':Path_length,  
        'lambda1':lambda1, 'lambda2':lambda2, 'lambda3':lambda3, 
        'eigenVectorMaxVar_lambda1[0]':eigenVectorMaxVar_lambda1[0], 'eigenVectorMaxVar_lambda1[1]':eigenVectorMaxVar_lambda1[1], 'eigenVectorMaxVar_lambda1[2]':eigenVectorMaxVar_lambda1[2], 
        'eigenVectorInterVar_lambda2[0]':eigenVectorInterVar_lambda2[0], 'eigenVectorInterVar_lambda2[1]':eigenVectorInterVar_lambda2[1], 'eigenVectorInterVar_lambda2[2]':eigenVectorInterVar_lambda2[2], 
        'eigenVectorMinVar_lambda3[0]':eigenVectorMinVar_lambda3[0], 'eigenVectorMinVar_lambda3[1]':eigenVectorMinVar_lambda3[1], 'eigenVectorMinVar_lambda3[2]':eigenVectorMinVar_lambda3[2], 
        'Vswlambda1':Vswlambda1, 'Vswlambda2':Vswlambda2, 'Vswlambda3':Vswlambda3, 
        'VsweigenVectorMaxVar_lambda1[0]':VsweigenVectorMaxVar_lambda1[0], 'VsweigenVectorMaxVar_lambda1[1]':VsweigenVectorMaxVar_lambda1[1], 'VsweigenVectorMaxVar_lambda1[2]':VsweigenVectorMaxVar_lambda1[2], 
        'VsweigenVectorInterVar_lambda2[0]':VsweigenVectorInterVar_lambda2[0], 'VsweigenVectorInterVar_lambda2[1]':VsweigenVectorInterVar_lambda2[1], 'VsweigenVectorInterVar_lambda2[2]':VsweigenVectorInterVar_lambda2[2], 
        'VsweigenVectorMinVar_lambda3[0]':VsweigenVectorMinVar_lambda3[0], 'VsweigenVectorMinVar_lambda3[1]':VsweigenVectorMinVar_lambda3[1], 'VsweigenVectorMinVar_lambda3[2]':VsweigenVectorMinVar_lambda3[2],
        'Vremaininglambda1':Vremaininglambda1, 'Vremaininglambda2':Vremaininglambda2, 'Vremaininglambda3':Vremaininglambda3, 
        'VremainingeigenVectorMaxVar_lambda1[0]':VremainingeigenVectorMaxVar_lambda1[0], 'VremainingeigenVectorMaxVar_lambda1[1]':VremainingeigenVectorMaxVar_lambda1[1], 'VremainingeigenVectorMaxVar_lambda1[2]':VremainingeigenVectorMaxVar_lambda1[2], 
        'VremainingeigenVectorInterVar_lambda2[0]':VremainingeigenVectorInterVar_lambda2[0], 'VremainingeigenVectorInterVar_lambda2[1]':VremainingeigenVectorInterVar_lambda2[1], 'VremainingeigenVectorInterVar_lambda2[2]':VremainingeigenVectorInterVar_lambda2[2], 
        'VremainingeigenVectorMinVar_lambda3[0]':VremainingeigenVectorMinVar_lambda3[0], 'VremainingeigenVectorMinVar_lambda3[1]':VremainingeigenVectorMinVar_lambda3[1], 'VremainingeigenVectorMinVar_lambda3[2]':VremainingeigenVectorMinVar_lambda3[2]}

        # Append detailed_info_dict to FR_detailed_info_DF.
        if not (detailed_info_dict is None):
            detailed_info_dict_df = pd.DataFrame([detailed_info_dict])
            eventList_DF_detailedInfo = pd.concat([eventList_DF_detailedInfo,detailed_info_dict_df], axis=0)
            eventList_DF_detailedInfo.reset_index(drop=True, inplace=True)

    # Add one more attribute: wait_time (time interval between two flux ropes.)
    # waitTime: start - start
    startTimeSeries = eventList_DF_detailedInfo['startTime'].copy()
    endTimeSeries = eventList_DF_detailedInfo['startTime'].copy()
    # Drop first record in startTime list.
    startTimeSeries.drop(startTimeSeries.index[[0]], inplace=True)
    # Reset index.
    startTimeSeries.reset_index(drop=True, inplace=True)
    # Drop last record in endTime list.
    endTimeSeries.drop(endTimeSeries.index[[-1]], inplace=True)
    # Reset index.
    endTimeSeries.reset_index(drop=True, inplace=True)
    # Calculate wait time.
    waitTime = startTimeSeries - endTimeSeries
    # Convert wait time to list.
    waitTimeList = []

    for record in waitTime:
        waitTime_temp = int(record.total_seconds())
        waitTimeList.append(waitTime_temp)
    # Add np.nan as first element.
    waitTimeList = [np.nan] + waitTimeList
    eventList_DF_detailedInfo.insert(32, 'waitTime_SS', waitTimeList) 
    # This command is able to specify column position.
    # print('\nwaitTime Start-Start is added.')
    
    # Add one more attribute: wait_time1 (time interval between two flux ropes.)
    # wait1Time = start{i+1} - end{i}
    startTimeSeries1 = eventList_DF_detailedInfo['startTime'].copy()
    endTimeSeries1 = eventList_DF_detailedInfo['endTime'].copy()
    # Drop first record in startTime list.
    startTimeSeries1.drop(startTimeSeries1.index[[0]], inplace=True)
    # Reset index.
    startTimeSeries1.reset_index(drop=True, inplace=True)
    # Drop last record in endTime list.
    endTimeSeries1.drop(endTimeSeries1.index[[-1]], inplace=True)
    # Reset index.
    endTimeSeries1.reset_index(drop=True, inplace=True)
    # Calculate wait time.
    wait1Time = startTimeSeries1 - endTimeSeries1
    # Convert wait time to list.
    wait1TimeList = []
    for record in wait1Time:
        wait1Time_temp = int(record.total_seconds())
        wait1TimeList.append(wait1Time_temp)
    # Add np.nan as first element.
    wait1TimeList = [np.nan] + wait1TimeList
    # Add one new column 'wait1Time' to dataframe
    eventList_DF_detailedInfo.insert(33, 'waitTime_SE', wait1TimeList) 
    # print('\nwaitTime Start-End is added.')

    # Save DataFrame to pickle file.
    print('\nSaving the detailed info of flux rope events to pickle file...')
    # If plotFolder does not exist, create it.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    eventList_DF_detailedInfo.to_pickle(output_dir + '/' + output_filename + '.p')
    # print('Done.')

    return eventList_DF_detailedInfo

###############################################################################
def plot_time_series_data(all_data_frame, spacecraftID, **kwargs):
    """This function plots time-series parameters of flux rope based on the detection. 
    It also prepares an additional panels for the reconstruction. 
    Return to show a figure."""

    # Check keywords.
    # Get flux rope label info.
    func = None
    if 'func' in kwargs: func = kwargs['func']
    if ('start' in kwargs) & ('end' in kwargs): 
        start = kwargs['start']
        end = kwargs['end']
        selectedRange_mask = (all_data_frame.index >= start) & (all_data_frame.index <= end)
        data_DF = all_data_frame.iloc[selectedRange_mask]
    else:
        data_DF = all_data_frame
    if 'spacecraftID' in kwargs:
        spacecraftID = kwargs['spacecraftID']
        print('spacecraftID          = {}'.format(spacecraftID))
    if 'fluxRopeList_DF' in kwargs:
        isLabelFluxRope = True
        fluxRopeList_DF = kwargs['fluxRopeList_DF']
    else:
        isLabelFluxRope = False
    # Get shock label info.
    if 'shockTimeList' in kwargs:
        shockTimeList = kwargs['shockTimeList']
        if shockTimeList: # If not empty.
            isLabelShock = True
        else:
            isLabelShock = False
    else: 
        isLabelShock = False

    # Check x interval setting
    fig_x_interval = 10
    if 'fig_x_interval' in kwargs:
        fig_x_interval = kwargs['fig_x_interval']
    
    # Plot function cannot handle the all NaN array.
    for column in data_DF:
        if data_DF[column].isnull().all():
            data_DF[column].fillna(value=0, inplace=True)
        
    # Make plots.
    # Physics constants.
    mu0 = 4.0 * np.pi * 1e-7 #(N/A^2) magnetic constant permeability of free space vacuum permeability.
    m_proton = 1.6726219e-27 # Proton mass. In kg.
    factor_deg2rad = np.pi/180.0 # Convert degree to rad.
    k_Boltzmann = 1.3806488e-23 # Boltzmann constant, in J/K.
    # Set plot format defination.
    #fig_formatter = mdates.DateFormatter('%m/%d %H:%M') # Full format is ('%Y-%m-%d %H:%M:%S').
    fig_formatter = mdates.DateFormatter('%H:%M') # Full format is ('%Y-%m-%d %H:%M:%S').
    fig_hour_locator = dates.HourLocator(interval=fig_x_interval)
    fig_title_fontsize = 11
    fig_ylabel_fontsize = 8
    fig_ytick_fontsize = 7
    fig_xtick_fontsize = 7
    fig_linewidth=1

    # Convert time format in DataFrame to datetime.
    Time_series = pd.to_datetime(data_DF.index)
    # Get range start and end time from data_DF.
    rangeStart = data_DF.index[0]
    rangeEnd = data_DF.index[-1]
        
    # Create Figure Title from date.
    if rangeStart.strftime('%Y/%m/%d') == rangeEnd.strftime('%Y/%m/%d'):
        rangeStart_str = rangeStart.strftime('%m/%d/%Y %H:%M:%S')
        rangeEnd_str = rangeEnd.strftime('%H:%M:%S')
    else:
        rangeStart_str = rangeStart.strftime('%m/%d/%Y %H:%M:%S')
        rangeEnd_str = rangeEnd.strftime('%m/%d/%Y %H:%M:%S')
    
    figureTitle = 'Time Interval '+ rangeStart_str + ' ~ ' + rangeEnd_str + ' (' + spacecraftID + ')'
    plotFileName = spacecraftID + '_' + rangeStart.strftime('%Y%m%d%H%M%S') + '_' + rangeEnd.strftime('%Y%m%d%H%M%S')

    if func == 'GSR':
        fig, ax = plt.subplots(4, 1, sharex=True,figsize=(6, 5.5))
        Bfield = ax[0]
        delta_lambda = ax[1]
        Beta_ratio = ax[2]
        Vsw = ax[3]
    else:
        fig, ax = plt.subplots(5, 1, sharex=True,figsize=(6, 5.5))
        Bfield = ax[0]
        Vsw = ax[1]
        Density = ax[2]
        Temp = ax[3]
        Beta_ratio = ax[4]

    # 1) Plot WIN magnetic field.
    Bfield.xaxis.set_major_formatter(fig_formatter) # Set xlabel format.
    Bfield.set_title(figureTitle, fontsize = fig_title_fontsize) # All subplots will share this title.
    Bfield.set_ylabel(r'$B$ (nT)',fontsize=fig_ylabel_fontsize) # Label font size.
    Bfield.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    Bfield.yaxis.set_major_locator(MaxNLocator(4))
    B0, = Bfield.plot(data_DF.index,data_DF.iloc[:,0],'-r',linewidth=fig_linewidth)
    B1, = Bfield.plot(data_DF.index,data_DF.iloc[:,1],'-g',linewidth=fig_linewidth)
    B2, = Bfield.plot(data_DF.index,data_DF.iloc[:,2],'-b',linewidth=fig_linewidth)
    WIN_Btotal = (data_DF.iloc[:,0]**2 + data_DF.iloc[:,1]**2 + data_DF.iloc[:,2]**2)**0.5
    lBl, = Bfield.plot(data_DF.index,WIN_Btotal,color='black',linewidth=fig_linewidth)
    Bfield.axhline(0, color='black',linewidth=0.5,linestyle='dashed') # Zero line, must placed after data plot
    # Only plot color span do not label flux rope in this panel.
    if isLabelFluxRope:
        for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
            startTime_temp = oneFluxRopeRecord['startTime']
            endTime_temp = oneFluxRopeRecord['endTime']
            # Plot color span.
            Bfield.axvspan(startTime_temp, endTime_temp, color='gray', alpha=0.2, lw=0)
            # Plot boundary line.
            Bfield.axvline(startTime_temp, color='black', linewidth=0.2)
            Bfield.axvline(endTime_temp, color='black', linewidth=0.2)
    # Indicate shock Time. No label.
    if isLabelShock:
        for shockTime in shockTimeList:
            Bfield.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
    
    if (spacecraftID == 'ACE') or (spacecraftID == 'WIND'):
        Bfield.legend(handles=[B0,B1,B2,lBl],labels=['$B_X$','$B_Y$','$B_Z$','|B|'],
                loc='center left',prop={'size':5}, bbox_to_anchor=(1.01, 0.5))
    if (spacecraftID == 'ULYSSES') or (spacecraftID == 'PSP') or (spacecraftID == 'SOLARORBITER'):
        Bfield.legend(handles=[B0,B1,B2,lBl],labels=['$B_R$','$B_T$','$B_N$','|B|'],
                loc='center left',prop={'size':5}, bbox_to_anchor=(1.01, 0.5))

    # 2) Plot WIND solar wind bulk speed
    Vsw.set_ylabel(r'$V_{sw}$ (km/s)', fontsize=fig_ylabel_fontsize)
    Vsw.yaxis.set_major_locator(MaxNLocator(4))
    Vsw.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    Vsw.tick_params(axis='x', labelsize=8) # Tick font size.
    WIN_Vtotal = (data_DF.iloc[:,3]**2 + data_DF.iloc[:,4]**2 + data_DF.iloc[:,5]**2)**0.5
    Vsw.plot(data_DF.index,WIN_Vtotal,linewidth=fig_linewidth,color='black')

    # Plot color span and label flux rope in this panel.
    if isLabelFluxRope:
        # Get y range, used for setting label position.
        y_min, y_max = Vsw.axes.get_ylim()
        for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
            startTime_temp = oneFluxRopeRecord['startTime']
            endTime_temp = oneFluxRopeRecord['endTime']
            # Set label horizontal(x) position.
            label_position_x = startTime_temp+(endTime_temp-startTime_temp)/2
            # Set label verticle(y) position.
            if not (index+1)%2: # Odd number.
                label_position_y = y_min + (y_max - y_min)/4.0 # One quater of the y range above y_min.
            else: # Even number.
                label_position_y = y_min + 3*(y_max - y_min)/4.0 # One quater of the y range above y_min.
            # Set label text.
            label_text = str(index+1)
            # Plot color span.
            Vsw.axvspan(startTime_temp, endTime_temp, color='gray', alpha=0.2, linewidth=0)
            # Plot boundary line.
            Vsw.axvline(startTime_temp, color='black', linewidth=0.2)
            Vsw.axvline(endTime_temp, color='black', linewidth=0.2)
            # Place label.
            Vsw.text(label_position_x, label_position_y, label_text, fontsize = 8, horizontalalignment='center', 
                verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'None', 'facecolor':'white', 'alpha':0.6})
            Vsw.text(label_position_x, label_position_y, label_text, fontsize = 8, horizontalalignment='center', 
                verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'black', 'facecolor':'None'})
    # Indicate shock Time. And label it.
    if isLabelShock:
        # Get y range, used for setting label position.
        y_min, y_max = Vsw.axes.get_ylim()
        for shockTime in shockTimeList:
            # Set label horizontal(x) position.
            label_position_x = shockTime
            # Set label verticle(y) position.
            label_position_y = y_min + (y_max - y_min)/2.0 # Half of the y range above y_min.
            # Set label text.
            shockTime_str = shockTime.strftime('%H:%M')
            label_text = 'SHOCK\n'+shockTime_str
            Vsw.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
            Vsw.text(label_position_x, label_position_y, label_text, fontsize = 7,horizontalalignment='center', 
                verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'None', 'facecolor':'white', 'alpha':0.6})
            Vsw.text(label_position_x, label_position_y, label_text, fontsize = 7,horizontalalignment='center', 
                verticalalignment='center',bbox={'boxstyle':'round','pad':0.2, 'edgecolor':'black', 'facecolor':'None'})
    
    if func == 'GSR':
        Bdelta=(np.arccos(data_DF.iloc[:,2]/WIN_Btotal)-np.pi/2)*180/np.pi
        Blambda=np.arccos(data_DF.iloc[:,0]/np.sqrt(data_DF.iloc[:,0]**2+data_DF.iloc[:,1]**2))
        Blambda[data_DF.iloc[:,1] < 0]=2*np.pi-Blambda[data_DF.iloc[:,1] < 0]
        Blambda=Blambda*180/np.pi

        # 3) Plot delta & lambda of the magnetic field
        delta_lambda.set_ylabel('Latitude (deg)', fontsize=fig_ylabel_fontsize)
        delta_lambda.yaxis.set_major_locator(MaxNLocator(4))
        delta_lambda.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        # delta_lambda.set_ylim([0,1])
        delta_lambda.plot(data_DF.index, Bdelta, linewidth=fig_linewidth,color='black',label = r'${\delta}$')
        # delta_lambda.legend(loc = 'upper left',prop={'size':5})
        delta_lambda.set_ylim([-90,90])

        delta_lambda_twin = delta_lambda.twinx()
        delta_lambda_twin.yaxis.set_major_locator(MaxNLocator(5))
        delta_lambda_twin.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        delta_lambda_twin.set_ylabel('Longitude (deg)', color='b', fontsize=fig_ylabel_fontsize)
        delta_lambda_twin.plot(data_DF.index, Blambda, color='b', linewidth=fig_linewidth)
        delta_lambda_twin.set_ylim([0,360])
        for ticklabel in delta_lambda_twin.get_yticklabels(): # Set label color to red
            ticklabel.set_color('b')
        # delta_lambda_twin.legend(labels=[r'${\lambda}$'], loc='upper right',prop={'size':5})
        # delta_lambda.xaxis.set_major_locator(fig_hour_locator)            
        delta_lambda.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)
    else:
        # 3) Plot WIN Proton number density, and alpha/proton ratio.
        Density.set_ylabel(r'$n_{p}$ (#/cc)', fontsize=fig_ylabel_fontsize)
        Density.yaxis.set_major_locator(MaxNLocator(4))
        Density.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        # Density.set_ylim([0,30])
        Density.plot(data_DF.index,data_DF['Np'],linewidth=fig_linewidth,color='black',label=r'$n_p$')
        if 'Ne' in data_DF.columns:
            Density.plot(data_DF.index,data_DF['Ne'],linewidth=fig_linewidth,color='red',label=r'$n_e$') 
        Density.legend(loc='upper left',prop={'size':5})
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                Density.axvspan(startTime_temp, endTime_temp, color='gray', alpha=0.2, lw=0)
                Density.axvline(startTime_temp, color='black', linewidth=0.2)
                Density.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                Density.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        Density_twin = Density.twinx()
        # Density_twin.set_ylim([0,0.1])
        Density_twin.yaxis.set_major_locator(MaxNLocator(4))
        Density_twin.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Density_twin.set_ylabel(r'$n_{\alpha}$', color='b', fontsize=fig_ylabel_fontsize) 
        #print(ratio)
        if 'Na' in data_DF.columns:
            Density_twin.plot(data_DF.index, data_DF['Na'], color='b', linewidth=fig_linewidth) 
            Density_twin.legend(labels=[r'$n_{\alpha}$'], loc='upper right',prop={'size':5})
        else:
            Density_twin.set_ylabel(r'$n_{\alpha}~(no~data)$', color='b', fontsize=fig_ylabel_fontsize)
        for ticklabel in Density_twin.get_yticklabels(): # Set label color to green
            ticklabel.set_color('b')

        # 4) Plot WIN Proton temperature, and plasma beta.
        Temp.set_ylabel(r'$T_{p}$ ($10^6$K)', fontsize=fig_ylabel_fontsize)
        Temp.yaxis.set_major_locator(MaxNLocator(4))
        Temp.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Tp_inMK = data_DF['Tp']/1e6
        # Temp.set_ylim([0,0.28])
        Temp.plot(Tp_inMK.index, Tp_inMK, linewidth=fig_linewidth,color='black',label=r'$T_p$')
        if 'Te' in data_DF.columns:
            Te_inMK = data_DF['Te']/1e6
            Temp.plot(Te_inMK.index, Te_inMK,linewidth=fig_linewidth,color='red',label=r'$T_e$')
        Temp.legend(loc='upper left',prop={'size':5})
        # Only plot color span do not label flux rope in this panel.
        if isLabelFluxRope:
            for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
                startTime_temp = oneFluxRopeRecord['startTime']
                endTime_temp = oneFluxRopeRecord['endTime']
                # Plot color span.
                Temp.axvspan(startTime_temp, endTime_temp, color='gray', alpha=0.2, lw=0)
                Temp.axvline(startTime_temp, color='black', linewidth=0.2)
                Temp.axvline(endTime_temp, color='black', linewidth=0.2)
        # Indicate shock Time. No label.
        if isLabelShock:
            for shockTime in shockTimeList:
                Temp.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
        # Set double x axis for beta.
        Temp_twin = Temp.twinx()
        Temp_twin.yaxis.set_major_locator(MaxNLocator(5))
        Temp_twin.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        Temp_twin.set_ylabel(r'$T_{\alpha}$', color='b', fontsize=fig_ylabel_fontsize)
        # Temp_twin.set_ylim([0,2.2]) #Bata.
        if 'Ta' in data_DF.columns:
            Ta_inMK = data_DF['Ta']/1e6
            Temp_twin.plot(Ta_inMK.index, Ta_inMK, color='b', linewidth=fig_linewidth)
            Temp_twin.legend(labels=[r'$T_{\alpha}$'], loc='upper right',prop={'size':5})
        else:
            Temp_twin.set_ylabel(r'$T_{\alpha}~(no~data)$', color='b', fontsize=fig_ylabel_fontsize)
        for ticklabel in Temp_twin.get_yticklabels(): # Set label color to red
            ticklabel.set_color('b')
        Temp.xaxis.set_major_locator(fig_hour_locator)            
        Temp.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)

    # 4) Plot plasma beta & alpha to proton ratio.
    if ('Tp' in data_DF.columns) & ('Np' in data_DF.columns):    
        Beta_p = data_DF['Np']*1e6*k_Boltzmann*data_DF['Tp']/(np.square(WIN_Btotal*1e-9)/(2.0*mu0))
        Beta_ratio.set_ylabel(r'${\beta}$', fontsize=fig_ylabel_fontsize)
        Beta_ratio.yaxis.set_major_locator(MaxNLocator(4))
        Beta_ratio.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
        # Beta_ratio.set_ylim([0,1])
        protonbeta, = Beta_ratio.plot(Beta_p.index, Beta_p, linewidth=fig_linewidth,color='black')
        Beta_ratio.legend(handles=[protonbeta],labels=[r'${\beta}_p$'],loc='upper left',prop={'size':5})
    else:
        Beta_ratio.set_ylabel(r'$no~np/Tp~data$', fontsize=fig_ylabel_fontsize)
    if ('Te' in data_DF.columns) & ('Ne' in data_DF.columns):
        Beta = (data_DF['Np']+data_DF['Ne'])*1e6*k_Boltzmann*(data_DF['Tp']+data_DF['Te'])\
        /(np.square(WIN_Btotal*1e-9)/(2.0*mu0))
        plasmabeta, = Beta_ratio.plot(Beta.index, Beta, linewidth=fig_linewidth,color='red',label = r'${\beta}$')
        Beta_ratio.legend(handles=[protonbeta,plasmabeta],labels=[r'${\beta}_p$',r'${\beta}$'],loc='upper left',prop={'size':5})

    # Only plot color span do not label flux rope in this panel.
    if isLabelFluxRope:
        for index, oneFluxRopeRecord in fluxRopeList_DF.iterrows():
            startTime_Beta_ratio = oneFluxRopeRecord['startTime']
            endTime_Beta_ratio = oneFluxRopeRecord['endTime']
            # Plot color span.
            Beta_ratio.axvspan(startTime_Beta_ratio, endTime_Beta_ratio, color='gray', alpha=0.2, lw=0)
            Beta_ratio.axvline(startTime_Beta_ratio, color='black', linewidth=0.2)
            Beta_ratio.axvline(endTime_Beta_ratio, color='black', linewidth=0.2)
    # Indicate shock Time. No label.
    if isLabelShock:
        for shockTime in shockTimeList:
            Beta_ratio.axvline(shockTime, color='black', linewidth=1, linestyle='dashed')# Shock Time.
    # Set double x axis for beta.
    Beta_ratio_twin = Beta_ratio.twinx()
    Beta_ratio_twin.yaxis.set_major_locator(MaxNLocator(5))
    Beta_ratio_twin.tick_params(axis='y', labelsize=fig_ytick_fontsize) # Tick font size.
    Beta_ratio_twin.set_ylabel(r'$n_{\alpha}/n_p$', color='b', fontsize=fig_ylabel_fontsize)
    # Beta_ratio_twin.set_ylim([0,2.2]) #Bata.
    if 'alphaRatio' in data_DF.columns:
        Beta_ratio_twin.plot(data_DF.index, data_DF['alphaRatio'], color='b', linewidth=fig_linewidth)
        Beta_ratio_twin.legend(labels=[r'$n_{\alpha}/n_p$'], loc='upper right',prop={'size':5})
    else:
        Beta_ratio_twin.set_ylabel(r'$n_{\alpha}/n_p~(no~data)$', color='b', fontsize=fig_ylabel_fontsize)
    for ticklabel in Beta_ratio_twin.get_yticklabels(): # Set label color to red
        ticklabel.set_color('b')
    # Beta_ratio.xaxis.set_major_locator(fig_hour_locator)            
    Beta_ratio.tick_params(axis='x', which='major', labelsize=fig_xtick_fontsize)

    # If adjust interval in the GS reconstruction
    adjustInterval=False
    if 'adjustInterval' in kwargs: 
        adjustInterval = kwargs['adjustInterval']
    if adjustInterval:
        print("\n---------------------------------------------")
        print('\nClick to select preferred starting and ending times...')
        select_boundary = plt.ginput(2)
        new_start_temp = select_boundary[0][0]
        new_end_temp = select_boundary[1][0]
        new_start_original = matplotlib.dates.num2date(new_start_temp, tz=None).strftime('%Y-%m-%d %H:%M:%S')
        new_end_original = matplotlib.dates.num2date(new_end_temp, tz=None).strftime('%Y-%m-%d %H:%M:%S')
        print("Selected starting time   = ", new_start_original)
        print("Selected ending time     = ", new_end_original)
        
        # Selected timestamps may not exist in the original dataframe
        # due to resolution. Thus, adjust them to be the most adjacent ones.
        data_DF_num = matplotlib.dates.date2num(data_DF.index)
        new_start_data_dif = abs(new_start_temp - data_DF_num)
        new_end_data_dif = abs(new_end_temp - data_DF_num)
        new_start = np.argwhere(new_start_data_dif == min(new_start_data_dif))[0][0]
        new_end = np.argwhere(new_end_data_dif == min(new_end_data_dif))[0][0]
        print("Adjusted starting time   = ", data_DF.index[new_start])
        print("Adjusted ending time     = ", data_DF.index[new_end])

        # Plot boundaries of selected interval
        Bfield.axvline(data_DF.index[new_start], color='black', linewidth=1, linestyle='dashed')
        Bfield.axvline(data_DF.index[new_end], color='black', linewidth=1, linestyle='dashed')
        Vsw.axvline(data_DF.index[new_start], color='black', linewidth=1, linestyle='dashed')
        Vsw.axvline(data_DF.index[new_end], color='black', linewidth=1, linestyle='dashed')
        delta_lambda.axvline(data_DF.index[new_start], color='black', linewidth=1, linestyle='dashed')
        delta_lambda.axvline(data_DF.index[new_end], color='black', linewidth=1, linestyle='dashed')
        Beta_ratio.axvline(data_DF.index[new_start], color='black', linewidth=1, linestyle='dashed')
        Beta_ratio.axvline(data_DF.index[new_end], color='black', linewidth=1, linestyle='dashed')
        
    if 'output_dir' in kwargs:
        output_dir = kwargs['output_dir']
        # Save to two places.
        print('\nSaving plot: {}...'.format(plotFileName))
        fig.savefig(output_dir + '/' + plotFileName + '.png', format='png', dpi=300, bbox_inches='tight')
        #fig.savefig(single_plot_dir + '/' + plotFileName + '.png', format='png', dpi=300, bbox_inches='tight')
        saved_filename = output_dir + '/' + plotFileName + '.png'
        print('Done.')
        return saved_filename
    else:
        return data_DF_num[new_start], data_DF_num[new_end]

###############################################################################
def find_duration_tuple(duration_range):
    # Spilt the user-specified duration range to tuple.

    duration = list(np.arange(10,60,10)) + list(np.arange(60,200,20)) + list(np.arange(200,400,40))
    duration_list = [(duration[i],duration[i+1]) for i in range(len(duration)-1)]

    # duration list left boundary
    duration_list_left = abs(np.array([duration_list[i][0] for i in range(len(duration_list))])-duration_range[0])
    # duration list right boundary
    duration_list_right = abs(np.array([duration_list[i][1] for i in range(len(duration_list))])-duration_range[1])
    # Find index of boundaries
    index_left = np.argwhere(duration_list_left == duration_list_left.min())[0][0]
    index_right = np.argwhere(duration_list_right == duration_list_right.min())[0][0]
    duration_range_tuple = tuple(duration_list[index_left:index_right+1])

    return duration_range_tuple

###############################################################################
def detection(rootDir,spacecraftID,timeStart,timeEnd,**kwargs):
    """Here is the main function that controls all processes during the GS detection."""

    # Terminal output format.
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 300)

    global includeNe, includeTe
    # Read detection settings
    if 'Search' in kwargs:
        Search = kwargs['Search']
    if 'CombineRawResult' in kwargs:
        CombineRawResult = kwargs['CombineRawResult']
    if 'GetMoreInfo' in kwargs:
        GetMoreInfo = kwargs['GetMoreInfo']
    if 'LabelFluxRope' in kwargs:
        LabelFluxRope = kwargs['LabelFluxRope']
    # Threshold on the magnitude of the magnetic field
    B_mag_thres = 0.0
    if 'B_mag_threshold' in kwargs:
        B_mag_thres = kwargs['B_mag_threshold']
    includeTe = False
    if 'includeTe' in kwargs:
        includeTe = kwargs['includeTe']
    includeNe = False
    if 'includeNe' in kwargs:
        includeNe = kwargs['includeNe']
    if ((spacecraftID == 'ACE') or (spacecraftID == 'ULYSSES') or 
        (spacecraftID == 'SOLARORBITER')) & (includeNe or includeTe):
        print("Error! ACE, ULYSSES, OR SOLARORBITER does not have electron data.")
        print("Please set includeTe = False and includeNe = False.")
        exit()
    # Indicate duration range
    duration = None
    if 'duration' in kwargs:
        duration = kwargs['duration']
    else:
        print("Error! Please indicate duration range, e.g., duration=(10,20).")
        exit()
    duration_range_tuple = find_duration_tuple(duration)
    allowIntvOverlap = False
    if allowIntvOverlap in kwargs:
        allowIntvOverlap = kwarg['allowIntvOverlap']
    # Load shock file
    shockList_DF_path = None
    if 'shockList_DF_path' in kwargs:
        shockList_DF_path = kwargs['shockList_DF_path']
    # xtick interval.
    fig_x_interval = 1 # In hours.
    if 'fig_x_interval' in kwargs:
        fig_x_interval = kwargs['fig_x_interval']
    

    print('\nStart Time: {}'.format(timeStart))
    print('End   Time: {}'.format(timeEnd))
    
    # Get parameters from input datetime.
    timeStart_str = timeStart.strftime('%Y%m%d%H%M%S')
    timeEnd_str = timeEnd.strftime('%Y%m%d%H%M%S')

    # Create folders for detection and results
    # Create case folder.
    case_dir = rootDir + 'detection_results/' + \
    spacecraftID + '_' + timeStart_str + '_' + timeEnd_str
    # csv filename.
    csv_filename = spacecraftID + '_' + timeStart_str + '_' + timeEnd_str + '.csv'
    # If case_dir folder does not exist, create it.
    if not os.path.exists(case_dir): os.makedirs(case_dir)
    # Create data_cache folder to save downloaded original spacecraft data
    data_cache_dir = case_dir + '/data_cache'
    # If data_cache folder does not exist, create it.
    if not os.path.exists(data_cache_dir): os.makedirs(data_cache_dir)
    # Create data_pickle_dir to save preprocessed data file.
    data_pickle_dir = case_dir + '/data_pickle'
    # If data_pickle folder does not exist, create it.
    if not os.path.exists(data_pickle_dir): os.makedirs(data_pickle_dir)
    # Create search_result_dir to save detection raw and processed results.
    search_result_dir = case_dir + '/detection_files'
    # If search_result folder does not exist, create it.
    if not os.path.exists(search_result_dir): os.makedirs(search_result_dir)
    # Create case result dir to save the final flux rope results including csv and figure.
    case_result_dir = case_dir + '/flux_rope_result'
    # If case_result folder does not exist, create it.
    if not os.path.exists(case_result_dir): os.makedirs(case_result_dir)

    # Start detection
    # 1) Download data from specified spacecraft with specified start and end time.
    print('\n######################################################################')
    print('######################## Downloading data ############################')
    print('######################################################################')
    data_dict = download_data(spacecraftID, data_cache_dir, timeStart, timeEnd)

    # 2) Preprocess data. Put all variables into DataFrame.
    print('\n######################################################################')
    print('##################### Prepare data for detection #####################')
    print('######################################################################')
    pickle_path_name = data_pickle_dir + '/' + spacecraftID +'_' + \
        timeStart_str + '_'+ timeEnd_str + '_preprocessed.p'
    # Try to load a preprocessed file if existed
    if (os.path.isfile(pickle_path_name)):
        print('\nLoading existed preprocessed data from \
            \n {}'.format(data_pickle_dir))
        data_DF = pd.read_pickle(open(pickle_path_name, 'rb'))
    # If not, process a new one
    else:
        print('\nProcessing data...')
        data_DF = preprocess_data(data_dict, data_pickle_dir, isPlotFilterProcess=True)
    
    # If there exists data with multiple resolution
    # Preprocess the second data file here
    # Usually higher resolution is for PSP & Solar Orbiter, i.e., 1s and 4s
    # Now the above data_DF resolution is 28s.
    if data_dict['MutliResolution']:
        print('\n######################################################################')
        print('\nPrepare data for the second resolution for detection.')
        pickle_path_name_2nd = data_pickle_dir + '/' + spacecraftID +'_' + \
        timeStart_str + '_'+ timeEnd_str + '_preprocessed_high_resltn.p'
        # Try to load a preprocessed file if existed
        if (os.path.isfile(pickle_path_name_2nd)):
            print('\nLoading existed preprocessed data from \
            \n {}'.format(data_pickle_dir))
            data_DF_high_resltn = pd.read_pickle(open(pickle_path_name_2nd, 'rb'))
       # If not, process a new one
        else:
            print('\nProcessing data...')
            data_DF_high_resltn = preprocess_data(data_dict, data_pickle_dir, 
                MutliResolution=True, isPlotFilterProcess=True)

    # If all the rest settings are False,
    # it means download and process only.
    if ((Search==False) & (CombineRawResult==False) &(GetMoreInfo==False) & (LabelFluxRope==False)):
        print("\nDone.")
        exit()

    # 3) Detect flux ropes.
    print('\n######################################################################')
    print('################## Search for flux rope candidates ###################')
    print('######################################################################')
    if Search:
        # if n_theta_grid=12, d_theta_deg = 90/12 = 7.5, d_phi_deg = 360/24 = 15
        # Determine whether it is multiple resolution first
        # For such a case, must run higher resolution first,
        # which results in records with shorter duration.
        # Then run for 28s resolution, such that records with longer
        # duration will survive. 

        if data_dict['MutliResolution']:
            # This line is run for 1s or 4s data,
            # thus loading "data_DF_high_resltn".
            # An additional controller reverseOrder is used. 
            # This helps using different output names. 
            search_result_raw = detect_flux_rope(spacecraftID, 
                data_DF_high_resltn, 
                duration_range_tuple, search_result_dir, data_dict, 
                n_theta_grid=12, reverseOrder=True, MutliResolution=True)
            print('\n######################################################################')
            print('\nSearch for flux ropes with the second resolution.')
            # This line is run for 28s data, thus loading "data_DF".
            search_result_raw_2nd = detect_flux_rope(spacecraftID, 
                data_DF, duration_range_tuple, search_result_dir, data_dict, 
                n_theta_grid=12, MutliResolution=True)
        # If not multiple resolution, detection flux rope with data_DF (28s for PSP & SolO)
        else:
            search_result_raw = detect_flux_rope(spacecraftID, data_DF, 
                duration_range_tuple, search_result_dir, data_dict, n_theta_grid=12)
    # If search has already been done, load the raw result directly.
    else:
        print('\nLoading existed searching results from \
            \n {}'.format(search_result_dir))
    
    # 4) Clean up records.     
    print('\n######################################################################')
    print('#################### Filter flux rope candidates #####################')
    print('######################################################################')
    if CombineRawResult:
        shockList_DF = pd.read_pickle(open(shockList_DF_path, 'rb'))    
        # If multiple resolution,
        # need to merge two raw result pickle files.
        # In order to avoid messing up the raw result between multiple resolutions 
        # and not multiple ones, here use subscript 28s_resltn to represent 28s 
        # results for PSP & SolO. The raw result with high resolution is "raw_result". 
        # For single-resolution spacecraft (WIND/ACE/Ulysses), 
        # "raw_result" is with their original resolution.
        if data_dict['MutliResolution']:
            search_result_raw = pd.read_pickle(open(search_result_dir + '/raw_result.p', 'rb'))
            search_result_raw1 = pd.read_pickle(open(search_result_dir + '/raw_result_28s_resltn.p', 'rb'))
            searchResultsTrue = {}
            searchResultsTrue.update(search_result_raw['true'])
            searchResultsTrue.update(search_result_raw1['true'])
            searchResultsFalse = {}
            searchResultsFalse.update(search_result_raw['false'])
            searchResultsFalse.update(search_result_raw1['false'])
            searchResultsRange = {}
            searchResultsRange.update(search_result_raw['timeRange'])
            searchResultsRange.update(search_result_raw1['timeRange'])
            search_result_raw_combined = {'true':searchResultsTrue, 'false':searchResultsFalse, 
            'timeRange':{'datetimeStart':timeStart, 'datetimeEnd':timeEnd}}
            # For mutltiple resolution, the 2nd data file is 28s, i.e., data_DF
            search_result_no_overlap_DF = clean_up_raw_result(spacecraftID, data_DF_high_resltn, 
                search_result_raw_combined, B_mag_threshold=B_mag_thres, walenTest_k_threshold=1.0, 
                min_residue_diff=0.12, min_residue_fit=0.14, output_dir=search_result_dir, 
                isVerbose=True, isRemoveShock=True, shockList_DF=shockList_DF, 
                allowOverlap=allowIntvOverlap, the2ndDF=data_DF)
        else:
            # For single resolution, there is only one data file, i.e., data_DF
            search_result_raw_filename = search_result_dir + '/raw_result.p'
            search_result_no_overlap_DF = clean_up_raw_result(spacecraftID, data_DF, 
                search_result_raw_filename, B_mag_threshold=B_mag_thres, walenTest_k_threshold=1.0, 
                min_residue_diff=0.12, min_residue_fit=0.14, output_dir=search_result_dir, 
                isVerbose=True, isRemoveShock=True, shockList_DF=shockList_DF, 
                allowOverlap=allowIntvOverlap)
    # If existed a combined result, load it.
    else:
        print('\nLoading existed combined results from \
            \n {}'.format(search_result_dir))
        search_result_no_overlap_DF = pd.read_pickle(open(search_result_dir 
            + '/no_overlap.p', 'rb'))

    print('\n######################################################################')
    print('################## Calculate flux rope information ###################')
    print('######################################################################')
    # 5) Get more flux rope information.
    if GetMoreInfo:
        if data_dict['MutliResolution']:
            # For mutltiple resolution, the 2nd data file is 28s, i.e., data_DF
            search_result_detail_info_DF = get_more_flux_rope_info(spacecraftID, 
                data_DF_high_resltn, search_result_no_overlap_DF, 
                output_dir=search_result_dir, the2ndDF = data_DF)
        else:
            # For single resolution, there is only one data file, i.e., data_DF
            search_result_detail_info_DF = get_more_flux_rope_info(spacecraftID, 
                data_DF, search_result_no_overlap_DF, output_dir=search_result_dir)
        # Save the result to csv file.
        print('Saving {}...'.format(csv_filename))
        search_result_detail_info_DF.to_csv(path_or_buf=case_result_dir + '/' + csv_filename)
        print('Done.')
    # If done already, load an existed file.
    else:
        print('\nLoad a file containing flux rope info from \
            \n {}'.format(search_result_dir))
        search_result_detail_info_DF = pd.read_pickle(open(search_result_dir 
            + '/detailed_info.p', 'rb'))

    # 6) Plot time series data for given time range. Label flux ropes.
    print('\n######################################################################')
    print('########################### Plot flux rope ###########################')
    print('######################################################################')
    if LabelFluxRope:
        fig = plot_time_series_data(data_DF, spacecraftID, output_dir=case_result_dir, 
            fluxRopeList_DF=search_result_detail_info_DF, fig_x_interval=fig_x_interval)
    else: 
        fig = plot_time_series_data(data_DF, spacecraftID, output_dir=case_result_dir, 
            fig_x_interval=fig_x_interval)

    return




