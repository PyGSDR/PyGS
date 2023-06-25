# Instruction on obtaining the Minimum Variance Analysis of the Magnetic Field (MVAB)    

To start, one will need to specify the directory to save the downloaded and processed data files,    
spacecraft mission, as well as starting and ending times.
```python
import os
import datetime
import matplotlib.pyplot as plt
import ReconstructionMisc as GSRM

timeStart=datetime.datetime(2005,8,28,8,36,0) # starting    
timeEnd=datetime.datetime(2005,8,28,8,49,0) # ending    

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
> [[ 0.35721525 -0.70017385  0.61818593]    
 [-0.8405972  -0.52952433 -0.11401901]    
 [-0.40717761  0.47891603  0.77771834]]    

Figure: Hodograms of the magnetic field.    
**Subscripts 1, 2, and 3 correspond to the maximum, intermediate, and minimum variance in the magnetic field.*   
<img width="600" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/test_mvab.png">
