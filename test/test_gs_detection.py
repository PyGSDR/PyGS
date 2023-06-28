import matplotlib.pyplot as plt
import datetime
from PyGS import FluxRopeDetection

rootDir = '/home/ychen/Desktop/PyGS' + '/examples/'

def test_gs_reconstruction():
    shockList = rootDir + '/IPShock_ACE_or_WIND_or_Ulysses_1996_2016_DF.p' 
    
    FluxRopeDetection.detection(rootDir,spacecraftID='WIND',
    timeStart=datetime.datetime(2018,10,31,18,0,0),
    timeEnd=datetime.datetime(2018,10,31,22,0,0),
    duration=(10,30),
    includeTe=False,includeNe=False,
    Search=True,CombineRawResult=True,GetMoreInfo=True,
    LabelFluxRope=True,B_mag_threshold=5.0,
    shockList_DF_path=shockList,FilterProcess=False,
    allowIntvOverlap=False)
    
    # Show time-series plot for the current interval
    # The shaded area represents flux rope.
    plt.show()