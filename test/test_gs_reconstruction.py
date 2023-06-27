import os
import numpy as np
import pandas as pd
import datetime
import FluxRopeDetection as detect
import obtainAxis as axis
import ReconstructionMisc as gsmisc

rootDir = os.getcwd() +'/examples/'

def test_gs_reconstruction():
    inputFileName = '2018_selected_events.p'
    SFR_detection_list = pd.read_pickle(open(rootDir + inputFileName,'rb'))
    gsmisc.reconstruction(rootDir,
    spacecraftID='WIND',
    FR_list=SFR_detection_list,eventNo=53,
    # adjustAxis is an interative feature
    # Change to True if need for test
    adjustAxis=False,
    # TimeStart & End do not matter here
    # Will extract timestamps from inputFileName
    timeStart=datetime.datetime(2018,8,24,11,29,0),
    timeEnd=datetime.datetime(2018,8,24,17,10,0),
    includeTe=False, 
    includeNe=False,
    saveFig=True, 
    plotJz=True,
    plotHodogram=True, 
    checkHT=True,
    plotWalenRelation=True, 
    plotSpacecraftTimeSeries=True,
    # adjustInterval is an interative feature
    # Change to True if need for test
    adjustInterval=False, 
    checkPtAFitting=True,
    grid_x=15, grid_y=131,
    get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)