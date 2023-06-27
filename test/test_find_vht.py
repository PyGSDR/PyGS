import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import ReconstructionMisc as gsmisc

rootDir = os.getcwd() + '/examples/'

def test_vht():
    timeStart=datetime.datetime(2018,8,24,11,29,0)
    timeEnd=datetime.datetime(2018,8,24,17,10,0)
    alldata, longer_data_interval = gsmisc.download_process_data(timeStart,
        timeEnd,file_dir=rootDir,spacecraftID='WIND',func='simple_VHT')
    VHT = gsmisc.findVHT(alldata,spacecraftID='WIND',func='simple_VHT',checkHT=True)
    
    assert len(VHT) == 3
    plt.show()
