# Instruction for GS Reconstruction (GSR)
Here is a step-by-step instruction for the Grad-Shafranov Reconstruction (GSR).<br><br> 
<code>SFR_detection_list = pd.read_pickle(open('/Users/Tom_and_Jerry/2001_selected_events.p','rb'))</code><br>
**The above line is optional, see instruction below.*<br><br>
<code>rootDir = '/Users/Tom_and_Jerry/' 
reconstruction(rootDir, spacecraftID='WIND',
FR_list=SFR_detection_list, eventNo=1, adjustAxis=True,
timeStart=datetime(2005,8,28,0,24,0),timeEnd=datetime(2005,8,28,0,32,0),
includeTe=False, includeNe=False, saveFig=False, plotJz=False,
plotHodogram=False, checkHT=False, plotWalenRelation=False, plotSpacecraftTimeSeries=False,
adjustInterval=False, checkPtAFitting=False,
grid_x=15, grid_y=131, get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)</code><br> 

The full function is the above <code>reconstruction(...)</code>, which is the only function that needs to run for GSR.    
The following describes each controller/parameter inside this function.    
*Additional lines might need to be added.*   
*For non-Python users, simply copy the above lines into the Python window and press enter. Remember to change the rootDir first.

## 1. Initializing
<p><strong>rootDir</strong>: set a directory where you would like to save files, such as downloaded & preprocessed data, GS reconstructed figures, etc.<br>
e.g., <code>rootDir = '/Users/Tom_and_Jerry/'</code> </p>
<p><strong>spacecraftID</strong>: specify the spacecraft ID, e.g., 'WIND', 'ACE', 'ULYSSES','PSP','SOLARORBITER'<br> 
e.g., <code>spacecraftID='WIND'</code></p>

## 2. Select the interval & obtain the FR axis
Two options to designate starting and ending times of an interval. <br>
- If use <strong>detection</strong> results, indicate the source of detection result as "FR_list" and event sequence number as "eventNo" (see below). <br>
e.g., test event is the first record in 2001_selected_events.p. <br>
<code>inputListDir = 'Detection/'
inputFileName = '2001_selected_events.p'
SFR_detection_list = pd.read_pickle(open(rootDir + inputListDir + inputFileName,'rb'))</code><br><br>
In <code>reconstruction(...)</code>, set:<br>
<code>FR_list = SFR_detection_list, eventNo=0 </code><br>
**This will automatically extract timestamps for the first event from the detection result.*

- If use User-selected interval, specify <strong>timeStart & timeEnd</strong><br>
e.g., <code>timeStart = datetime(2005,8,28,0,23,0), timeEnd=datetime(2005,8,28,0,32,0)</code><br>
**If both options exist in reconstruction, will prioritize timestamps from the detection result.*

Flux rope axis: Two options to obtain the FR axis. <br>
- If from <strong>detection</strong> results: 
  <code>FR_list = SFR_detection_list, eventNo=0, adjustAxis=False </code><br>
  **This will use the axis of the first event from the detection.* <br>
  **If would like to adjust the interval while still using this axis, set adjustInterval = True.* <br>
- If need to adjust axis from <strong>reconstruction</strong>:
  <code>adjustAxis=True </code><br>

## 3. Start reconstruction
*Has to run twice to have the final GSR results.*<br>
**First round**: e.g., <code>grid_x=15, grid_y=131, get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0</code><br>
**Second round**: e.g., <code>grid_x=15, grid_y=131, get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0</code><br>
**or** <code>grid_x=15, grid_y=131, get_Ab=-1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0</code><br>
**The sign of get_Ab depends on flux rope chirality. Right/left-handed corresponds to +/- 1.*

<strong>grid_x</strong>: default setting = 15, the grid in the x direction in FR frame. No need to specify if using the default setting.<br>
<strong>grid_y</strong>: default setting = 131, the grid in the y direction in FR frame. No need to specify if using the default setting.<br>
<strong>get_Ab</strong>: *MUST INDICATE when running*<br>
initial setting = 0, select the boundary of A. If satisfied, set it to 1.<br>
<strong>pressureSwitch</strong>: *MUST INDICATE when running*<br>
initial setting = 0 to fit Bz'(A') first. <br>
If satisfied with the current results, set to 1 to include other pressure terms to have the final reconstruction. <br>
<strong>polyOrder</strong>: usually set to be 2 or 3, the order of polynomials.<br>
<strong>dmid</strong>: initial setting = 0, where the spacecraft path is at. <br>
<strong>dAl0</strong>: initial setting = 0.0, adjust to any numbers in [0,1] to change the left boundary/percentage of extrapolation.<br>
<strong>dAr0</strong>: initial setting = 0.0, adjust to any numbers in [0,1] to change the right boundary/percentage of extrapolation.<br>

** All initial settings may need adjustments case by case,    
while the default settings can be used without indicating again unless one would like to adjust. <br>
*** By this step, the GSR is completed.<br>

## 4. Selective settings and Add-on features
Set to be True if would like to implement additional functions<br>

<strong>includeTe</strong>: include the electron temperature in the transverse pressure Pt'.<br>
<strong>includeNe</strong>: include the electron number density in the transverse pressure Pt'.<br>
<strong>saveFig</strong>: save figures.<br>
<strong>plotJz</strong>: plot the map of the axial current density jz.<br>
<strong>plotHodogram</strong>: plot hodograms of the magnetic field via the MVAB.<br>
<strong>plotWalenRelation</strong>: plot the Walen relation between V-remaining and VA.<br>
<strong>plotSpacecraftTimeSeries</strong>: plot the time-series data from spacecraft.<br>
<strong>adjustInterval</strong>: adjust the boundary of an interval, i.e., starting and/or ending times.<br>
*Default setting is to show the current interval with +/- 10 points.*<br>
<strong>checkPtAFitting</strong>: check whether the extrapolation percentages (dAl0 & dAr0) are well selected.<br>
<strong>checkHT</strong>: check whether the current HT frame is well-found.<br>
