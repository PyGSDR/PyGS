import os
import datetime
import ReconstructionMisc as gsmisc

timeStart=datetime.datetime(2005,8,28,8,36,0)
timeEnd=datetime.datetime(2005,8,28,8,49,0)
alldata, longer_data_interval = gsmisc.download_process_data(timeStart,
    timeEnd,file_dir=rootDir,spacecraftID='WIND')
B_DataFrame = gsmisc.clouddata(timeStart,timeEnd,alldata,func='mvab',spacecraftID='WIND')
X = gsmisc.MVAB(B_DataFrame, plotHodogram=True)
print(X)
