import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ReconstructionMisc as gsmisc

rootDir = os.getcwd() +'/examples/'

def test_HT_frame():
    timeStart=datetime.datetime(2018,8,24,11,29,0)
    timeEnd=datetime.datetime(2018,8,24,17,10,0)
    z_axis = np.array([0.15038373318,-0.852868531952,-0.5])
    alldata, longer_data_interval = gsmisc.download_process_data(timeStart,
        timeEnd,file_dir=rootDir,spacecraftID='WIND')
    HT_frame = gsmisc.clouddata(timeStart,timeEnd,alldata,
        func='HT_frame',fluxropeAxis=z_axis, spacecraftID='WIND')
    assert np.shape(HT_frame) == (3,3)
