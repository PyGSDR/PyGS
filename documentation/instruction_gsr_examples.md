# Instruction for GS reconstruction (GSR) and examples 
Here is a step-by-step instruction for the Grad-Shafranov Reconstruction (GSR).    
This document also presents examples corresponding to changes in the main command line.    

I. The full expression of the GSR main function:    
  ```python
  reconstruction(rootDir, spacecraftID='WIND', 
                 FR_list=SFR_detection_list, eventNo=0,
                 timeStart=datetime(2018,8,28,0,24,0), timeEnd=datetime(2018,8,28,0,32,0), 
                 adjustAxis=False, 
                 grid_x=15, grid_y=131, 
                 get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                 includeTe=False, includeNe=False, saveFig=False, plotJz=False, 
                 plotHodogram=False, checkHT=False, plotWalenRelation=False, 
                 plotSpacecraftTimeSeries=False, adjustInterval=False, 
                 checkPtAFitting=False, helicityTwist=False)
  ```
II. This instruction will include:     
- Initializing (lines before the main function and the first three rows in the above main function)   
- Obtaining the flux rope axis (the 4th row)   
  - from the detection result    
  - from reconstruction     
- Round 1 for reconstruction
  - The 5th shows the default settings, and the 6th row shows initial settings
  - May need to change during this round   
- Round 2 for reconstruction (may need to change parameters in the 5th & 6th rows again) 
- Add-on features (the 7th - 10th rows)   

III. Notes:    
- The GSR needs to run twice to have the final results.    
- Except for the optional flux rope axis, the rest steps in II are common for both options.    
- The final outputs will include a set of figures showing flux rope features.    

IV. For **non-Python users** or those who are not very familiar with Python:    
- Simply copy the command lines in block to your terminal or into the script file "yourfile.py".
    - for the latter, it means a file when you run ```$ python3 yourfile.py```    
- Please also pay attention to warnings & information printed on the terminal   

---
## 1. Initializing
Initialization includes all lines before the main function ```reconstruction(...)``` as well as the first three rows inside it.
- **rootDir**: set a directory where you would like to save files, such as downloaded & preprocessed data, GS reconstructed figures, etc.    
  - E.g., ```rootDir = '/Users/Tom_and_Jerry/'```
- **spacecraftID**: specify the spacecraft ID, e.g., 'WIND', 'ACE', 'ULYSSES', 'PSP', 'SOLARORBITER'.    
  - E.g., ```spacecraftID='WIND'```
- **Flux rope timestamps**: can be either from detection or user-specified
  - If **detection**: 
    - Specify the source of event list from the detection, i.e., ```inputFileName```
    - Here we use a record 08/24/2018 11:29:00 - 17:10:00 UT via the WIND spacecraft.    
      - It is the first event in the detection file selected_events.p, i.e., ```eventNo = 0```
    - The starting and ending times will be extracted automatically.
    - No need to indicate timestamps in ```reconstruction(...)```.    
    <br>

    ```python
    import pickle
    import pandas as pd
    import datetime
    from PyGS.ReconstructionMisc import reconstruction

    rootDir = '/home/ychen/Desktop/PyGS/examples/' # specify rootDir
    inputFileName = 'selected_events.p' # specify input file's name
    # load flux rope list from detection result
    SFR_detection_list = pd.read_pickle(open(rootDir + inputFileName,'rb')) 

    # In reconstruction (...), specify spacecraft ID, FR_list, and eventNo.
    # The second line shows the initial settings of the GSR. MUST INCLUDE THE FIRST TWO.
    reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)

    ```
  - If **user-specified** or the detection result is unavailable:
    - One needs to specify ```timeStart``` and ```timeEnd```.
    - Data will be downloaded and processed automatically.
    <br>
    
    ```python
    import pickle
    import pandas as pd
    import datetime
    from PyGS.ReconstructionMisc import reconstruction

    rootDir = '/home/ychen/Desktop/PyGS/examples/' # specify rootDir
    
    # In reconstruction (...), specify spacecraft ID, timeStart, and timeEnd.
    # The second line shows the initial settings of the GSR. MUST INCLUDE THE FIRST TWO.
    reconstruction(rootDir, spacecraftID='WIND', 
                   timeStart=datetime(2018,8,24,11,29,0), timeEnd=datetime(2018,8,24,17,10,0),
                   get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)

    ```
  - **Attention**:
    - If both options are included, i.e., ```FR_list=SFR_detection_list, eventNo=0, timeStart=datetime(2018,8,28,0,24,0), timeEnd=datetime(2018,8,28,0,32,0)```, will prioritize timestamps from the detection result and neglect ```timeStart and timeEnd```.
  
## 2. Obtaining the flux rope axis  
In this section, we will introduce how to proceed the reconstruction with the detection result and how to select the flux rope axis via the reconstruction.

### 2.1 From the detection result
- Skip the rest of Section 2. Go to Section 3.
  - Since we use the detection result, the flux rope axis will be extracted automatically as well.
  - Actually, the reconstruction is already started by copying the code block in Section 1.
- If specifying timestamps manually in the last step, go to Section 2.2 to obtain a new axis.

### 2.2 From reconstruction 
- Keep those lines above the main function ```reconstruction(...)```.    
- Turn on ```adjustAxis``` in ```reconstruction(...)```.
- Notice that this process will only be implemented with initial settings ```get_Ab = 0``` and ```pressureSwitch = 0```.
  - If either setting is ```1```, it will automatically load the saved z-axis file. 
- It will pop up four figures to let you decide whether this is a good axis.
- Provide an answer on the terminal to proceed (see detailed info below).
- Here, we would like to use timestamps of detection event No.0 while getting a new axis.

  ```python
  import pickle
  import pandas as pd
  import datetime
  from PyGS.ReconstructionMisc import reconstruction

  rootDir = '/home/ychen/Desktop/PyGS/examples/' # specify rootDir
  inputFileName = 'selected_events.p' # specify input file's name
  # load flux rope list from detection result
  SFR_detection_list = pd.read_pickle(open(rootDir + inputFileName,'rb')) 

  # In reconstruction (...), specify spacecraft ID, FR_list, and eventNo.
  # The second line shows the initial settings of the GSR. MUST INCLUDE THE FIRST TWO
  # The default setting of adjustAxis is False if not specified.
  reconstruction(rootDir, spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                 get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                 adjustAxis=True)
  ``` 
- Outputs:
  - Left: The residue map & Right: Pt'(A') in different axial directions:
    > Left: The big black dot represents the axis with the minimum residue.    
    > Users select any point within the contour.    
    > The cross and number represent the sequence of the current trial, i.e., how many clicks by far.    
    > The next three figures pop up, and with them, users will determine whether this is a good axis.    
    > The right Figure shows two branches of Pt'(A') in different directions (see legends).    
    > **If it appears blank, it means the current selection is not acceptable. Try a different one.*    
    > <img width="300" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/axis_residue_map.png">
    > <img width="300" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/axis_multi_directions.png">
  - Parameters along the spacecraft path: A', B in flux rope frame, pressures, and plasma beta.
    > Figure shows several parameters along the spacecraft path with the selected axial direction.    
    > For a flux rope structure, A' has one and only one extremum, and B_inFR has one or more rotating components.    
    > <img width="500" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/axis_parameters_alongsc.png">
  - Four pressures versus A'.
    > Panels show Pt'(A'), Bz'(A'), p'(A'), and PBz'(A').    
    > For a flux rope, it must have a good double-folding pattern between two branches of Pt'(A').    
    > Bz'(A') curves usually also have the double-folding pattern except for those large-beta flux ropes.    
    > <img width="500" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/axis_4_pressures.png">
    
- On the terminal, it will ask you to answer "y" or "n" to **Is the optimal invariance direction found**?
  - If "n", figures will be closed automatically except for the residue map.
    - On the residue map, click on an alternative point.
    - The rest three figures will show up and let you decide again.
    - There will be 5 trials for each run, i.e., you can answer "n" five times.
    - The temporary residue map will be saved if the 5th result is still unsatisfied.
  - If "y", all figures will be closed automatically and the axis will be saved to your ```rootDir```.    
    - Figure 1 in Section 3.1 will appear (see below).
    - Now, it means you are ready for getting reconstruction results.

- In the meanwhile, the terminal will show information below:
  - Trial number
  - Three components of the minimum residue axis and selected axis.
  - Instruction on how to select an axis.
  - Figure information

- The above procedure also proceeds with user-specified time intervals
  
## 3. Round 1 for reconstruction
Durinng two rounds in Sections 3.1 & 3.2, we will mainly adjust the 5th & 6th rows in ```reconstruction(...)```.
- Explanation of parameters in these two rows:
  - **grid_x**: No need to specify if using the default setting
    - default setting = 15, the grid in the x direction in FR frame. 
  - **grid_y**: No need to specify if using the default setting
    - default setting = 131, the grid in the y direction in FR frame. 
  - **get_Ab**: MUST INDICATE when running
    - initial setting = 0, select the boundary of A. If satisfied, set it to 1.
  - **pressureSwitch**: MUST INDICATE when running
    - initial setting = 0 to fit Bz'(A') first.
    - If satisfied with the current results, set to 1 or -1 to include other pressure terms to have the final reconstruction.
  - **polyOrder**: usually set to be 2 or 3, the order of polynomials.
  - **dmid**: initial setting = 0, the position where the spacecraft path is at.
  - **dAl0**: initial setting = 0.0
    - adjust to any numbers in [0,1] to change the left boundary/percentage of extrapolation.
  - **dAr0**: initial setting = 0.0
    - adjust to any numbers in [0,1] to change the right boundary/percentage of extrapolation.

- Round 1: runs with initial settings
  - Here we only present results with the axis from the detection.
    - Those with the axis from reconstruction will follow the same procedure. 
  - Since lines above the main function ```reconstruction(...)``` remain the same, we will focus on changes in controllers within
```reconstruction(...)```. Correspondingly, the code example will only display for ```reconstruction(...)```. 
For non-Python users, remember to copy all those lines before this function when running.

    ```python
    # Using default grids, thus no need to specify grid_x or grid_y.
    reconstruction(rootDir, spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)
    ```    

    - Outputs:    
      > **Figure 1**: Four parameters (Pt', Bz', Ppe', and PBz') versus A'.    
      > Initially, there is no cyan vertical dashed line on the second panel.
      > Users need to select it by manually clicking the cross point of two curves, i.e., the boundary of A'.
      > Take the second panel as the main reference.
      > If it does not have a cross point, try the first panel.
      > The reconstruction is then processed and Figures 2 & 3 will pop up.    
      > <img width="500" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_pressures.png">   

      > **Figures 2 & 3**: The reconstructed cross-sectional map & Pt' versus A'.    
      > These two figures are temporary results obtained from the GSR (not the final yet).         
      > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_cross_section.png">
      > <img width="220" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_PtA.png">

  - Can also change ```dmid```, ```dAl0```, and ```dAr0``` if necessary.
    - ```dmid = 0```: set the spacecraft path at 0.
    -  ```dAl0``` and ```dAr0```: Usually can turn on ```checkPtAFitting``` to see whether they need changes.

       ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   checkPtAFitting=True)
        ```
    - Outputs:
      > Figure shows the normalized Pt'(A') and fitting curves.    
      > Since the left and right boundaries look good. We decide to keep ```dAl0=0.0, dAr0=0.0```.     
      > <img width="250"        src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_check_PtA.png">    
    - Can be adjusted multiple times until satisfied.
      - For **non-Python users**, you will manually close or quit all figures to rerun the above command line.
      - If satisfied with the current results, go to Section 4.
      
## 4. Round 2 for reconstruction
```python
reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
               get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)
```
- The major change is to add the rest pressure terms in the extended GS equation, i.e., turning on ```pressureSwitch```.    
- Since we are also satisfied with the current boundary of A' and would not like to select it again, we also change ```get_Ab```.
  - ```get_Ab=1``` for right-handed flux rope (anti-clockwise rotated white arrows)
  - ```get_Ab=-1``` for left-handed flux rope (clockwise rotated white arrows)
  - **If the axis was obtained from Section 2.2*, non-zero value of ```get_Ab``` and ```pressureSwitch = 1``` will directly load the saved axis. 

    Outputs:
    > Figure 1: Four pressures versus A'.    
    > Since the first round has ```pressureSwitch = 0```, the thermal pressure (3rd panel) was 0.    
    > Now, all pressures are included.
    > 
    > <img width="500" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_4_pressures.png">   

    > Figures 2 & 3: The reconstructed cross-sectional map & Pt' versus A'.    
    > These two figures are the **final** results from the GSR.    
    > Left: the black contours and color background represent the transverse Bt and the axial ﬁeld Bz.
    > The closed transverse ﬁeld-line regions and the gradient of the unipolar Bz confirm the ﬂux rope conﬁguration.
    > More info about these figures can be found in any reference papers related to the GSR technique.   
    > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_cross_section.png">
    > <img width="220" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_PtA.png">


- Again, check whether ```dmid```, ```dAl0```, and ```dAr0``` needs adjustments.  
  - ```dmid = -10```
    - Now, we would like to move the spacecraft path upward.
      > Comparison between before (left) and after moving (right):    
      > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_cross_section.png">
      > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_cross_section_dmid.png">        
  - ```dAl0 = 0.0, dAr0 = 0.0```
    - Outputs:    
      > Figure shows the normalized curves.    
      > So, still feel good, and thus no changes.     
      > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_check_PtA.png">

- Since we are okay with the current results, the GSR for this event is complete by this step.
  - If unsatisfied, re-run the main function and adjust the above parameters.

## 5. Add-on features (recommend turning on after Round 2)
The following are add-on features, which are independent and optional.     
The default setting is False, i.e., they will not be implemented.    
Change to True if would like to have any features.    

- **saveFig**: save figures.
   
- **plotJz**: plot the map of the axial current density jz.
    - In ```reconstruction(...)```, add ```plotJz = True```.    

        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   plotJz=True)
        ```
        > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_Jz.png">    
    
- **plotHodogram**: plot hodograms of the magnetic field via the MVAB.
    - In ```reconstruction(...)```, add ```plotHodogram = True```.    
        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   plotHodogram=True)
        ```
        > Subscripts 1, 2, and 3 correspond to the maximum, intermediate, and minimum variance in the magnetic field.    
        > <img width="500" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_hodogram.png">
        
- **plotWalenRelation**: plot the Walen relation between V-remaining and VA.
    - In ```reconstruction(...)```, add ```plotWalenRelation = True```.    
        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   plotWalenRelation=True)
        ```
        > The Walen relation between the remaining flow and Alfven velocities.    
        > Red, green, and blue circles represent components (R, T, N) or (X, Y, Z).    
        > <img width="300" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_walen_relation.png">  

- **plotSpacecraftTimeSeries**: plot the time-series data from spacecraft.
   - In ```reconstruction(...)```, add ```plotSpacecraftTimeSeries = True```.    
        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   plotSpacecraftTimeSeries=True)
        ```
       > Time series plot from the original spacecraft dataset.    
       > <img width="500" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_spacecraftTimeSeries.png">  
        
- **adjustInterval**: adjust the boundary of an interval, i.e., starting and/or ending times.    
    - The default setting is to show the current interval with +/- 10 points.
    - Needs to add during the first round. The second round will load a saved boundary file directly.    
    - In ```reconstruction(...)```, add ```adjustInterval = True```.    

       ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   adjustInterval=True)
        ```
       > Time series plot with the new interval marked by two vertical dashed lines.    
       > They are adjusted based on the resolution of processed data.    
       >
       > On Terminal, it shows:    
       > Selected starting time   =  2018-08-24 11:34:28    
       > Selected ending time     =  2018-08-24 16:58:25    
       > Adjusted starting time   =  2018-08-24 11:34:00    
       > Adjusted ending time     =  2018-08-24 16:58:00    
       > <img width="500" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_adjustInterval.png">
       
- **checkPtAFitting**: as shown in the previous section.
     
- **helicityTwist**: estimate the relative helicity and average twist (shown on terminal).
    
- **checkHT**: check whether the current HT frame is well-found.
     - In ```reconstruction(...)```, add ```checkHT = True```.    

        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   checkHT=True)
        ```
        > Comparison between -VHT x B and -VHT(t) x B.    
        > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_checkHT.png">  
        
- **includeTe**: include the electron temperature in the transverse pressure Pt'.
  
- **includeNe**: include the electron number density in the transverse pressure Pt'.
  
- **Other Calculations and derived parameters** of a flux rope: mostly shown on the terminal.    
      - Parameters for the Grad-Shafranov reconstruction, i.e., the aforementioned 6th row     
      - Figure information      
      - The maximum of the axial magnetic field **Bz**   = 7.403238594695631 nT      
      - The maximum of the axial current density **jz**  = 3.241255038840692e-12 A/m^2    
      - Estimated axial current                          = 64515916.57123131 A    
      - Estimated **toroidal magnetic flux**             = 179317855413.55267 Wb    
      - Estimated **poloidal magnetic flux** at 1 AU     = 1144618485766.9966 Wb    
      - Estimated **relative helicity per unit length**  = 0.00031573680494749977 nT^2AU^2         
      - Estimated **average twist per AU**               = 4.839753149634445
