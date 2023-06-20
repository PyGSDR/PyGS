# Instruction for GS detection (GSD)<br>
Here is a step-by-step instruction for the Grad-Shafranov-based detection. 

<code>rootDir = '/home/ychen/Desktop/PyGS/'
shockList = rootDir + 'IPShock_ACE_or_WIND_or_Ulysses_1996_2016_DF.p'</code>

<code>detection(rootDir,spacecraftID='PSP',
    timeStart=datetime(2018,10,31,18,0,0),
    timeEnd=datetime(2018,10,31,20,0,0),
    duration=(10,30),
    includeTe=True,includeNe=True,
    Search=True,CombineRawResult=True,GetMoreInfo=True,
    LabelFluxRope=True,B_mag_threshold=25.0,
    shockList_DF_path=shockList,
    allowIntvOverlap=False)</code>

The main function is the above detection(...), which is the only function that needs to run for GSR.<br>
In addition, one has to specify the directory of 'rootDir' and the source of shock list.<br>
The final results will be a time-series plot and a csv file including flux rope parameters. <br>

The following describes each controller/parameter inside this function:<br><br>
**rootDir**: <br> set a directory where you would like to save files, 
such as downloaded & preprocessed data, GS detection results, etc.<br>
e.g., <code>rootDir = '/Users/Tom_and_Jerry/' </code>

**spacecraftID**: specify the spacecraft ID, e.g., 'WIND', 'ACE', 'ULYSSES','PSP','SOLARORBITER'.<br>
e.g., <code>spacecraftID='WIND'</code>

**timeStart & timeEnd**: specify time of searching interval.<br>
**Better less than a half day*.

**duration**: specify the duration range for detection.<br>
**The lower limit is 10, and the upper limit is 360*.<br>
E.g., <code>duration=(10,30)</code><br>
10 is the lower limit, and 30 is upper limit.<br>
During detection, it will be spilt into (10, 20) & (20, 30), or split with increments 10/20/40 points.<br>


**includeTe**: include the electron temperature in the transverse pressure Pt'.

**includeNe**: include the electron number density in the transverse pressure Pt'.

**Search**: default setting is True to search flux ropes in all 
search windows.<br>
Can be set to False if already have a raw result from search.
In such a case, a raw pickle file will be loaded.

**CombineRawResult**: default setting is True to combine results via all 
search windows.<br>
Can be set to False if already have a combined result.
In such a case, a combined pickle file will be loaded.

**GetMoreInfo**: default setting is True to calculate the average parameters, etc., 
within flux rope interval.<br>
Can be set to False if already have detailed info.
In such a case, a detailed info pickle file will be loaded.

**LabelFluxRope**: set to True if want to label flux rope in the final 
time-series plot.

**B_mag_threshold**: set a limit on the magnitude of the magnetic field to remove
small fluctuations.
  
**shockList_DF_path**: indicate the shock file to remove flux rope records
containing shocks

**allowIntvOverlap**: set to True if allow flux rope intervals to be overlapped
with adjacent events
