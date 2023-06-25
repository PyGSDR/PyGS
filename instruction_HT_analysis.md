# Instruction of the de Hoffmann-Teller (HT) analysis
The de Hoffmann-Teller (HT) related analysis in PyGS includes two functions:     
finding the HT frame velocity and build the HT frame with a given axial direction.     


## 1. Finding the HT frame velocity - VHT
To find the VHT, one will need to specify the directory to save the downloaded and processed data files,    
spacecraft mission, as well as starting and ending times.    
For example, calculate VHT during **08/28/2005 08:36 - 08:49** via the **WIND** spacecraft dataset.    

```python
import os
import numpy as np
import datetime
import ReconstructionMisc as GSRM
     
rootDir = os.getcwd() +'/test/' # directory
timeStart=datetime.datetime(2005,8,28,8,36,0) # starting
timeEnd=datetime.datetime(2005,8,28,8,49,0) # ending
    
# download and process the data file
alldata, longer_data_interval = GSRM.download_process_data(timeStart, timeEnd, file_dir=rootDir,
                                                           spacecraftID='WIND', func='simple_VHT') 
# Calculate VHT
VHT = gsmisc.findVHT(alldata,spacecraftID='WIND',func='simple_VHT',checkHT=True)
print("VHT is =", VHT)

plt.show()
```
On your terminal: 
> VHT is = [-389.04530776    0.95885129    3.96096323]


## 2. Building the HT frame with a given axial direction
Since this is a flux rope interval, we use its axis as the axial direction,    
i.e., z_axis = [0.469846310393,-0.171010071663,0.866025403784].    
Similarly, specify the directory to save the downloaded and processed data files,     
spacecraft mission, axis, as well as starting and ending times.    
```python
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import ReconstructionMisc as GSRM

timeStart=datetime.datetime(2005,8,28,8,36,0) # starting
timeEnd=datetime.datetime(2005,8,28,8,49,0) # ending
z_axis = np.array([0.469846310393,-0.171010071663,0.866025403784]) # given a direction

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
> [[ 0.88273743  0.00436813  0.46984631]    
 [ 0.0861651   0.98149434 -0.17101007]    
 [-0.46189849  0.19144135  0.8660254 ]]


**The first to third columns represent X_UnitVector, Y_UnitVector, and Z_UnitVector respectively.*
