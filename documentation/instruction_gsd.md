# Instruction for GS detection (GSD)    
Here is a step-by-step instruction for the Grad-Shafranov-based detection. 
## Main Function
```python
import datetime
from PyGS.FluxRopeDetection import detection

rootDir = '/Users/Tom_and_Jerry/' 
shockList = rootDir + 'IPShock_ACE_or_WIND_or_Ulysses_1996_2016_DF.p'

if __name__ == "__main__":
    detection(rootDir,spacecraftID='WIND',
        timeStart=datetime(2018,10,31,18,0,0), timeEnd=datetime(2018,10,31,22,0,0),
        duration=(10,30), includeTe=True, includeNe=False,
        Search=True, CombineRawResult=True, GetMoreInfo=True,
        LabelFluxRope=True, B_mag_threshold=5.0,
        shockList_DF_path=shockList,
        allowIntvOverlap=False)
```

The main function is the above ```detection(...)```, which is the only function that needs to run for GSD.    
In addition, one has to specify the directory of 'rootDir' and the source of shock list.    
The **final results** will be a time-series plot and a csv file including flux rope parameters.     

## Description
The following describes each controller/parameter inside this function:<br><br>

- **rootDir**: set a directory where you would like to save files, such as downloaded & preprocessed data, GS detection results, etc.    
    - ```rootDir = '/Users/Tom_and_Jerry/' ```

- **spacecraftID**: specify the spacecraft ID, e.g., 'WIND', 'ACE', 'ULYSSES','PSP','SOLARORBITER'.<br>
    - ```spacecraftID='WIND' ```

- **timeStart & timeEnd**: specify the time of the searching interval.  
    - ```timeStart=datetime(2018,10,31,18,0,0), timeEnd=datetime(2018,10,31,20,0,0)```    
    - Better less than a half day.

- **duration**: specify the duration range for detection.
    - ```duration=(10,30)```
        - Here 10 is the lower limit, and 30 is the upper limit.
        - During detection, it will be spilt into (10, 20) & (20, 30). 
    - For general use, the lower limit is 10, and the upper limit is 360.        
        - All windows will be split with increments 10/20/40 points.    

- **includeTe**: include the electron temperature in the transverse pressure Pt'.    
    - ``` includeTe=True```
    - Will remind if a spacecraft dataset does not have Te data

- **includeNe**: include the electron number density in the transverse pressure Pt'.    
    - ```includeNe=True```
    - Will remind if a spacecraft dataset does not have Ne data 
 
- **Search**: the default setting is True to search flux ropes in all search windows.    
    - ```Search=True``` 
    - Can be set to False if already have a raw result from the search.    
    - In such a case, a raw pickle file will be loaded.      
 
- **CombineRawResult**: the default setting is True to combine results via all search windows.    
    - ```CombineRawResult=True```    
    - Can be set to False if already have a combined result.    
    - In such a case, a combined pickle file will be loaded.        

- **GetMoreInfo**: the default setting is True to calculate the flux rope parameters.   
    - ```GetMoreInfo=True```   
    - Can be set to False if already have detailed info.    
    - In such a case, a detailed info pickle file will be loaded.        

- **LabelFluxRope**: set to True if want to label flux rope in the final time-series plot.    
    - ```LabelFluxRope=True```
    - If would like to download and process data only:
        - set ```Search=False, CombineRawResult=False, GetMoreInfo=False, LabelFluxRope=False```.    

- **B_mag_threshold**: set a limit on the magnitude of the magnetic field (nT) to remove small fluctuations.    
    - ```B_mag_threshold=5.0```
  
- **shockList_DF_path**: indicate the shock file to remove flux rope records containing shocks.

- **allowIntvOverlap**: set to True if allow flux rope intervals to be overlapped with adjacent events.    
    - ```allowIntvOverlap=False```

## Example of Detection results
- Flux rope event list
<img width="500" src="https://github.com/PyGSDR/PyGS/raw/main/example_figures/detection_FR_event_list.png">

- Flux rope time-series plot
<img width="500" src="https://github.com/PyGSDR/PyGS/raw/main/example_figures/detection_FR_time_series.png">
