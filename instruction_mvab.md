# Instruction on obtaining the Minimum Variance Analysis of the Magnetic Field (MVAB)    

To start, one will need to specify the directory to save the downloaded and processed data files,    
spacecraft mission, as well as starting and ending times.
```python
import os
import datetime
import matplotlib.pyplot as plt
from PyGS import ReconstructionMisc as GSRM

timeStart=datetime.datetime(2018,8,24,11,29,0) # starting    
timeEnd=datetime.datetime(2018,8,24,17,10,0) # ending    

# download and process the data file
alldata, longer_data_interval = gsmisc.download_process_data(timeStart, timeEnd,
                                                             file_dir=rootDir, spacecraftID='WIND')    
# Get the magnetic field data from processed file                                                         
B_DataFrame = gsmisc.clouddata(timeStart,timeEnd,alldata,func='mvab',spacecraftID='WIND')    

# Get MVAB frame    
MVAB_frame = gsmisc.MVAB(B_DataFrame, plotHodogram=True)    
print(MVAB_frame)

plt.show()
```
On your terminal:
> [[ 0.53546172  0.05153188  0.84298589]    
 [-0.3504958   0.92168323  0.16629107]    
 [ 0.76839666  0.38450552 -0.5115878 ]]
> 
> Figure: Hodograms of the magnetic field.      
> **Subscripts 1, 2, and 3 correspond to the maximum, intermediate, and minimum variance in the magnetic field.*   
> <img width="600" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_hodogram.png">
