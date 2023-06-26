# Instruction for GS detection (GSD)    
Here is a step-by-step instruction for the Grad-Shafranov-based detection. 
## Main command line
```python
import datetime
from FluxRopeDetection import detection

rootDir = '/Users/Tom_and_Jerry/' 
shockList = rootDir + 'IPShock_ACE_or_WIND_or_Ulysses_1996_2016_DF.p'

detection(rootDir, spacecraftID='PSP',
          timeStart=datetime(2018,10,31,18,0,0), timeEnd=datetime(2018,10,31,20,0,0),
          duration=(10,30), includeTe=True, includeNe=True,
          Search=True, CombineRawResult=True, GetMoreInfo=True,
          LabelFluxRope=True, B_mag_threshold=25.0, shockList_DF_path=shockList, allowIntvOverlap=False)
```

The main function is the above ```detection(...)```, which is the only function that needs to run for GSD.    
In addition, one has to specify the directory of 'rootDir' and the source of shock list.    
The final results will be a time-series plot and a csv file including flux rope parameters.     

## Description
The following describes each controller/parameter inside this function:<br><br>
**rootDir**: set a directory where you would like to save files,     
such as downloaded & preprocessed data, GS detection results, etc.    
E.g., ```rootDir = '/Users/Tom_and_Jerry/' ```

**spacecraftID**: specify the spacecraft ID, e.g., 'WIND', 'ACE', 'ULYSSES','PSP','SOLARORBITER'.<br>
E.g., ```spacecraftID='WIND' ```

**timeStart & timeEnd**: specify the time of the searching interval.  
E.g., ```timeStart=datetime(2018,10,31,18,0,0), timeEnd=datetime(2018,10,31,20,0,0)```    
**Better less than a half day*.

**duration**: specify the duration range for detection.    
**The lower limit is 10, and the upper limit is 360*.    
E.g., ```duration=(10,30)```    
10 is the lower limit, and 30 is upper limit.    
During detection, it will be spilt into (10, 20) & (20, 30), or split with increments 10/20/40 points.    

**includeTe**: include the electron temperature in the transverse pressure Pt'.    
E.g., ``` includeTe=True```    

**includeNe**: include the electron number density in the transverse pressure Pt'.    
E.g., ```includeNe=True```    
 
**Search**: the default setting is True to search flux ropes in all search windows.    
Can be set to False if already have a raw result from the search.    
In such a case, a raw pickle file will be loaded.      
E.g., ```Search=True```  

**CombineRawResult**: the default setting is True to combine results via all search windows.    
Can be set to False if already have a combined result.    
In such a case, a combined pickle file will be loaded.        
E.g., ```CombineRawResult=True```    

**GetMoreInfo**: the default setting is True to calculate the average parameters, etc., within the flux rope interval.    
Can be set to False if already have detailed info.    
In such a case, a detailed info pickle file will be loaded.        
E.g., ```GetMoreInfo=True```  

**LabelFluxRope**: set to True if want to label flux rope in the final time-series plot.    
E.g., ```LabelFluxRope=True```    

**B_mag_threshold**: set a limit on the magnitude of the magnetic field (nT) to remove small fluctuations.    
E.g., ```B_mag_threshold=25.0```
  
**shockList_DF_path**: indicate the shock file to remove flux rope records containing shocks

**allowIntvOverlap**: set to True if allow flux rope intervals to be overlapped with adjacent events.    
E.g., ```allowIntvOverlap=False```