import os
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import datetime
import FluxRopeDetection

rootDir = os.getcwd() +'/examples/'

def test_gs_reconstruction():
    shockList = rootDir + '/IPShock_ACE_or_WIND_or_Ulysses_1996_2016_DF.p' 
    FluxRopeDetection.detection(rootDir,spacecraftID='WIND',
    timeStart=datetime.datetime(2018,8,24,11,29,0),
    timeEnd=datetime.datetime(2018,8,24,17,10,0),
    duration=(10,30),
    includeTe=False,includeNe=False,
    Search=True,CombineRawResult=True,GetMoreInfo=True,
    LabelFluxRope=True,B_mag_threshold=5.0,
    shockList_DF_path=shockList,FilterProcess=False,
    allowIntvOverlap=False)
    
    # Show time-series plot for the current interval
    # The shaded area represents flux rope.
    plt.show()