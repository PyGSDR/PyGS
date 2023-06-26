# Example of the GS Reconstruction (GSR) outputs
In a separate file, we show the step-by-step instruction for GSR.     
The corresponding outputs are presented here.

- Example 1
- Using ...
```python

import pickle
import pandas as pd
import datetime
from ReconstructionMisc import reconstruction

rootDir = '/home/ychen/Desktop/PyGS/'
inputFileName = 'events/2005_selected_events.p'
SFR_detection_list = pd.read_pickle(open(rootDir + inputFileName,'rb'))

reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list,eventNo=4, adjustAxis=True, 
    get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)
```

The full function is the above <code>reconstruction(...)</code>, which is the only function that needs to run for GSR.    
Notice that this function has to **run twice with adjusted parameters** to have the final GSR results (see Section 3).    

The following describes each controller/parameter inside this function.      
**For non-Python users, simply copy the above lines into the Python window and press enter. Remember to change the rootDir first.*

## 1. Initializing
**rootDir**: set a directory where you would like to save files, such as downloaded & preprocessed data, GS reconstructed figures, etc.    
e.g., ```rootDir = '/Users/Tom_and_Jerry/'```    

**spacecraftID**: specify the spacecraft ID, e.g., 'WIND', 'ACE', 'ULYSSES','PSP','SOLARORBITER'    
e.g., ```spacecraftID='WIND'```

## 2. Select the interval & obtain the FR axis
Two options to designate starting and ending times of an interval.     
**If both options exist in reconstruction, will prioritize timestamps from the detection result.*    

- If use **detection** results, indicate the source of detection result as "FR_list"    
  and the event sequence number as "eventNo" (see below).    
  - E.g., the test event is the **first record** (No.0) in **2001_selected_events.p**.
    - In ```reconstruction(...)```, set: ```FR_list = SFR_detection_list, eventNo=0```,
    - and add below lines before ```reconstruction(...)``` to load the detection result.
    - **This will automatically extract timestamps of the first event from the detection result.* 
```python
inputFileName = 'events/2001_selected_events.p'
SFR_detection_list = pd.read_pickle(open(rootDir + inputFileName,'rb')) 
```

- If use User-selected interval, specify **timeStart & timeEnd**.
  - E.g., ```timeStart = datetime(2005,8,28,0,23,0), timeEnd=datetime(2005,8,28,0,32,0)```    
  

**Flux rope axis**: Two options to obtain the FR axis.    
- If from **detection** results:
  - ```FR_list = SFR_detection_list, eventNo=0, adjustAxis=False```       
  - **This will use the axis of the first event from the detection.*     
  - **If would like to adjust the interval while still using this axis, set adjustInterval = True.*    
- If need to adjust axis from **reconstruction**: ```adjustAxis=True```    

## 3. Start reconstruction
**Has to run twice to have the final GSR results.*<br>
**First round**: ```grid_x=15, grid_y=131, get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0```   
**Second round**: e.g., ```grid_x=15, grid_y=131, get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0```    
**or** ```grid_x=15, grid_y=131, get_Ab=-1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0```    
**The sign of get_Ab depends on flux rope chirality. Right/left-handed corresponds to +/- 1.*    

**grid_x**: default setting = 15, the grid in the x direction in FR frame. No need to specify if using the default setting.   
**grid_y**: default setting = 131, the grid in the y direction in FR frame. No need to specify if using the default setting.    
**get_Ab**: *MUST INDICATE when running*    
initial setting = 0, select the boundary of A. If satisfied, set it to 1.    
**pressureSwitch**: *MUST INDICATE when running*    
initial setting = 0 to fit Bz'(A') first.     
If satisfied with the current results, set to 1 to include other pressure terms to have the final reconstruction.     
**polyOrder**: usually set to be 2 or 3, the order of polynomials.    
**dmid**: initial setting = 0, where the spacecraft path is at.     
**dAl0**: initial setting = 0.0, adjust to any numbers in [0,1] to change the left boundary/percentage of extrapolation.    
**dAr0**: initial setting = 0.0, adjust to any numbers in [0,1] to change the right boundary/percentage of extrapolation.    

** All initial settings may need adjustments case by case,    
while the default settings can be used without indicating again unless one would like to adjust.    
*** By this step, the GSR is completed.    

## 4. Selective settings and Add-on features
Set to be True if would like to implement additional functions.    

**includeTe**: include the electron temperature in the transverse pressure Pt'.    
**includeNe**: include the electron number density in the transverse pressure Pt'.    
**saveFig**: save figures.   
**plotJz**: plot the map of the axial current density jz.    
**plotHodogram**: plot hodograms of the magnetic field via the MVAB.    
**plotWalenRelation**: plot the Walen relation between V-remaining and VA.    
**plotSpacecraftTimeSeries**: plot the time-series data from spacecraft.    
**adjustInterval**: adjust the boundary of an interval, i.e., starting and/or ending times.    
*Default setting is to show the current interval with +/- 10 points.*    
**checkPtAFitting**: check whether the extrapolation percentages (dAl0 & dAr0) are well selected.    
**helicityTwist**: estimate the relative helicity and average twist.     
**checkHT**: check whether the current HT frame is well-found.    
