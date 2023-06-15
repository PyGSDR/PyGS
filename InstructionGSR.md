# Information and instruction for PyGSDR

*italic*
**bold**
> citaion
>> built in citation
> - bulleted points

## Instruction for GS Detection (GSD)

## Instruction for GS Reconstruction (GSR)

1. Initializing:
**rootDir**: 
e.g., rootDir = '/Users/Tom_and_Jerry/'
- Set a directory where you would like to save files, 
    such as downloaded & preprocessed data, GS reconstructed figures, etc.

spacecraftID: 
e.g., spacecraftID='WIND'
- Specify the spacecraft ID, e.g., 'WIND', 'ACE', 'ULYSSES','PSP','SOLARORBITER'

--------------------------------------------------

2. SELECT INTERVAL & OBTAIN FR AXIS:
timeStart & timeEnd:
e.g., timeStart = datetime(2005,8,28,0,23,0),
      timeEnd=datetime(2005,8,28,0,32,0)
- Designate starting and ending times of an interval
* No need to do so if reconstructing SFR from GS detection results. 
    For such records, only indicate the source of detection result 
    as "FR_list" and event sequence number as "eventNo".

2.1 FR AXIS
e.g., 
inputListDir = 'Detection/'
inputFileName = '2001_selected_events.p'
SFR_detection_list = pd.read_pickle(open(rootDir + inputListDir + inputFileName,'rb'))
FR_list = SFR_detection_list, eventNo=0, selectedAxis=False,

OR

selectedAxis = True, # FR_list = filepath_filename, eventNo=0,

- Specify where the FR axis will be obtained. 
    (1) FR_list & eventNo: the axis is from the detection result. 
    Thus, the event list (pickle file) has to be specified as well as 
    the event sequence number. 
    (2) selectedAxis: the axis is obtained from GSR based on the minimum residue.
* Notice that the controller "selectedAxis=True" is in conflict with FR_list & eventNo.

--------------------------------------------------

3. START RECONSTRUCTION:
* Has to run twice to have the final GSR results.
First round:
e.g., get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0
Second round:
e.g., get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0

grid_x: default setting = 15, the grid in the x direction in FR frame.
grid_y: default setting = 131, the grid in the y direction in FR frame.
get_Ab:  initial setting = 0, select the boundary of A. 
    If satisfied, set it to 1.
    * MUST indicate when running
pressureSwitch: initial setting = 0 to fit Bz'(A') first. 
    If satisfied with the current results, set to 1 to include 
    other pressure terms to have the final reconstruction.
    * MUST indicate when running
polyOrder: usually set to be 2 or 3, the order of polynomials.
dmid: initial setting = 0, where the spacecraft path is at. 
dAl0: initial setting = 0.0, adjust to any numbers in [0,1] 
    to change the left boundary/percentage of extrapolation.
dAr0: initial setting = 0.0, adjust to any numbers in [0,1] 
    to change the right boundary/percentage of extrapolation.

** All initial settings may need adjustments case by case, 
    while the default settings can be used without 
    indicating again unless one would like to adjust.
*** By this step, the GSR is completed.

--------------------------------------------------

4. SELECTIVE SETTINGS:
* Set to be True if would like to implement additional functions

includeTe: include the electron temperature in the transverse pressure Pt'.
saveFig: save figures.
plotJz: plot the map of the axial current density jz.
plotHodogram: plot hodograms of the magnetic field via the MVAB.
plotWalenRelation: plot the Walen relation between V-remaining and VA.
plotSpacecraftTimeSeries: plot the time-series data from spacecraft.
adjustInterval: adjust the boundary of an interval, i.e., starting and/or ending times.
* Default setting is to show the current interval with +/- 10 points.
checkPtAFitting: check whether the extrapolation percentages (dAl0 & dAr0) are well selected.
checkMVAaxis: check Pt'(A') in MVA and minimum residue directions.
    * This is only shown with selectedAxis=True.
checkHT: check whether the current HT frame is well-found.