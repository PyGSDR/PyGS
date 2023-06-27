import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import ReconstructionMisc as gsmisc

rootDir = os.getcwd() +'/examples/'

def test_mvab():
    timeStart=datetime.datetime(2018,8,24,11,29,0)
    timeEnd=datetime.datetime(2018,8,24,17,10,0)
    alldata, longer_data_interval = gsmisc.download_process_data(timeStart,
        timeEnd,file_dir=rootDir,spacecraftID='WIND')
    B_DataFrame = gsmisc.clouddata(timeStart,timeEnd,alldata,spacecraftID='WIND',func='mvab')
    X = gsmisc.MVAB(B_DataFrame, plotFigure=True, plotHodogram=True)
    assert np.shape(X) == (3,3)
    plt.show()