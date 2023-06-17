# Instruction for GS Reconstruction (GSR)
Here is a step-by-step instruction for the Grad-Shafranov Reconstruction (GSR).<br><br> 
<code>rootDir = '/Users/Tom_and_Jerry/'    
reconstruction(rootDir,spacecraftID='WIND',timeStart=datetime(2005,8,28,0,23,0),timeEnd=datetime(2005,8,28,0,32,0),
FR_list=SFR_detection_list, eventNo=1, selectedAxis=False, includeTe=False, saveFig=False, plotJz=False,
checkMVAaxis=False, plotHodogram=False, checkHT=False, plotWalenRelation=False, plotSpacecraftTimeSeries=False,
adjustInterval=False, checkPtAFitting=False,
grid_x=15, grid_y=131, get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)</code><br> 

The main function is the above <code>reconstruction(...)</code>, which is the only function that needs to run for GSR.    
The following describes each controller/parameter inside this function.    
*Additional lines might need to be added.*   
*For non-Python users, simply copy the above lines into the Python window and press enter. Remember to change the rootDir first.

## 1. Initializing
<p><strong>rootDir</strong>: set a directory where you would like to save files, such as downloaded & preprocessed data, GS reconstructed figures, etc.<br>
e.g., <code>rootDir = '/Users/Tom_and_Jerry/'</code> </p>
<p><strong>spacecraftID</strong>: specify the spacecraft ID, e.g., 'WIND', 'ACE', 'ULYSSES','PSP','SOLARORBITER'<br> 
e.g., <code>spacecraftID='WIND'</code></p>

## 2. Select interval & obtain FR axis
<strong>timeStart & timeEnd</strong>: designate starting and ending times of an interval.<br>
e.g., <code>timeStart = datetime(2005,8,28,0,23,0),timeEnd=datetime(2005,8,28,0,32,0)</code><br><br>
*No need to do so if reconstructing SFR from GS detection results.* <br>
*For such records, only indicate the source of detection result as "FR_list" and event sequence number as "eventNo" (see below).*

Flux rope axis: need to specify where the FR axis will be obtained. <br>
<strong>FR_list & eventNo</strong>: the axis is from the detection result.    
Thus, the event list (pickle file) has to be specified as well as the event sequence number.<br>
<strong>selectedAxis</strong>: the axis is obtained from GSR based on the minimum residue. <br>
*Notice that the controller "selectedAxis=True" is in conflict with FR_list & eventNo.*<br>

e.g., 
<code>inputListDir = 'Detection/'
inputFileName = '2001_selected_events.p'
SFR_detection_list = pd.read_pickle(open(rootDir + inputListDir + inputFileName,'rb'))</code><br><br>
In <code>reconstruction(...)</code>, set:<br>
<code>FR_list = SFR_detection_list, eventNo=0, selectedAxis=False</code><br>
**or**, <code>selectedAxis = True, # FR_list = SFR_detection_list, eventNo=0</code><br>

## 3. Start reconstruction
*Has to run twice to have the final GSR results.*<br>
**First round**: e.g., <code>grid_x=15, grid_y=131, get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0</code><br>
**Second round**: e.g., <code>grid_x=15, grid_y=131, get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0</code><br>

<strong>grid_x</strong>: default setting = 15, the grid in the x direction in FR frame.<br>
<strong>grid_y</strong>: default setting = 131, the grid in the y direction in FR frame.<br>
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

## 4. Selective settings
Set to be True if would like to implement additional functions<br>

<strong>includeTe</strong>: include the electron temperature in the transverse pressure Pt'.<br>
<strong>saveFig</strong>: save figures.<br>
<strong>plotJz</strong>: plot the map of the axial current density jz.<br>
<strong>plotHodogram</strong>: plot hodograms of the magnetic field via the MVAB.<br>
<strong>plotWalenRelation</strong>: plot the Walen relation between V-remaining and VA.<br>
<strong>plotSpacecraftTimeSeries</strong>: plot the time-series data from spacecraft.<br>
<strong>adjustInterval</strong>: adjust the boundary of an interval, i.e., starting and/or ending times.<br>
*Default setting is to show the current interval with +/- 10 points.*<br>
<strong>checkPtAFitting</strong>: check whether the extrapolation percentages (dAl0 & dAr0) are well selected.<br>
<strong>checkMVAaxis</strong>: check Pt'(A') in MVA and minimum residue directions.<br>
*This is only shown with selectedAxis=True.*<br>
<strong>checkHT</strong>: check whether the current HT frame is well-found.<br>
