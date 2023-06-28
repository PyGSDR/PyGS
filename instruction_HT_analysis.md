# Instruction of the de Hoffmann-Teller (HT) analysis
The de Hoffmann-Teller (HT) related analysis in PyGS includes two functions:     
finding the HT frame velocity and build the HT frame with a given axial direction.     


## 1. Finding the HT frame velocity - VHT
To find the VHT, one will need to specify the directory to save the downloaded and processed data files,    
spacecraft mission, as well as starting and ending times.    
For example, calculate VHT during **08/24/2018 11:29 - 17:10** via the **WIND** spacecraft dataset.    

```python
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from PyGS import ReconstructionMisc as GSRM
     
rootDir = os.getcwd() +'/test/' # directory to where you save test folder
timeStart=datetime.datetime(2018,8,24,11,29,0) # starting
timeEnd=datetime.datetime(2018,8,24,17,10,0) # ending
    
# download and process the data file
alldata, longer_data_interval = GSRM.download_process_data(timeStart, timeEnd, file_dir=rootDir,
                                                           spacecraftID='WIND', func='simple_VHT') 
# Calculate VHT
VHT = gsmisc.findVHT(alldata,spacecraftID='WIND',func='simple_VHT',checkHT=True)
print("VHT is =", VHT)

plt.show()
```
On your terminal: 
> The HT frame is good when the correlation coefficient is close to 1.    
> For the current time interval, it is 0.9997792532099188.     
> VHT is = [-360.61966463   -6.51443291    1.5279859]

Figure: Comparison between -VHT x B and -VHT(t) x B    
<img width="400" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_checkHT.png">


## 2. Building the HT frame with a given axial direction
Since this is a flux rope interval, we use its axis as the axial direction,    
i.e., z_axis = [0.15038373318,-0.852868531952,-0.5].    
Similarly, specify the directory to save the downloaded and processed data files,     
spacecraft mission, axis, as well as starting and ending times.    
```python
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
from PyGS import ReconstructionMisc as GSRM

timeStart=datetime.datetime(2018,8,24,11,29,0) # starting   
timeEnd=datetime.datetime(2018,8,24,17,10,0) # ending    
z_axis = np.array([0.15038373318,-0.852868531952,-0.5]) # given a direction    

# download and process the data file
alldata, longer_data_interval = GSRM.download_process_data(timeStart, timeEnd,
                                                           file_dir=rootDir, spacecraftID='WIND')
# build the HT frame
HT_frame = GSRM.clouddata(timeStart, timeEnd, alldata, func='HT_frame', 
                          fluxropeAxis=z_axis, spacecraftID='WIND')
print(HT_frame)

plt.show()
```
On your terminal:   
> [[ 0.9885453   0.01276428  0.15038373]    
 [ 0.13625192 -0.50403441 -0.85286853]    
 [ 0.06491232  0.86358925 -0.5       ]]    

**The first to third columns represent X_UnitVector, Y_UnitVector, and Z_UnitVector respectively.*
