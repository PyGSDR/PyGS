import os
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import datetime
import ReconstructionMisc as gsmisc

rootDir = os.getcwd() +'/test/'

timeStart=datetime.datetime(2005,8,28,8,36,0)
timeEnd=datetime.datetime(2005,8,28,8,49,0)
alldata, longer_data_interval = gsmisc.download_process_data(timeStart,
    timeEnd,file_dir=rootDir,spacecraftID='WIND',func='simple_VHT')
VHT = gsmisc.findVHT(alldata,spacecraftID='WIND',func='simple_VHT',checkHT=True)
print(VHT)
 
# Test HT frame
timeStart=datetime.datetime(2005,8,28,8,36,0)
timeEnd=datetime.datetime(2005,8,28,8,49,0)
z_axis = np.array([0.469846310393,-0.171010071663,0.866025403784])
alldata, longer_data_interval = gsmisc.download_process_data(timeStart,
    timeEnd,file_dir=rootDir,spacecraftID='WIND')
HT_frame = gsmisc.clouddata(timeStart,timeEnd,alldata,
    func='HT_frame',fluxropeAxis=z_axis, 
    spacecraftID='WIND')
print(HT_frame)
# First column: X_UnitVector
# Second column: Y_UnitVector
# Third column: Z_UnitVector
