# Example of the GS Reconstruction (GSR) outputs  
This file presents examples corresponding to changes in the main command line ```reconstruction(...)```.    
It includes the full process using the axis from the detection result and that from reconstruction.    
**Remember the GSR needs to run twice to have the final results.*

Example:    
Flux rope 08/24/2018 11:29:00 - 17:10:00 UT via the WIND spacecraft     
No.53 in 2018_selected_events.p.    
**Since we aim to reconstruct a detection result here, the starting and ending times will be extracted from the detection result as well. One can also specify any time following the instruction.*

## Using the flux rope axis from the detection result
### 1. Round 1
```python
import pickle
import pandas as pd
import datetime
from ReconstructionMisc import reconstruction

rootDir = '/home/ychen/Desktop/PyGS/' # specify rootDir
inputFileName = 'events/2018_selected_events.p' # specify input file's name
# load flux rope list from detection result
SFR_detection_list = pd.read_pickle(open(rootDir + inputFileName,'rb')) 

# In reconstruction (...), specify spacecraft ID, FR_list, and eventNo.
# The second line shows the initial settings of the GSR.
reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
               get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)
```    

Outputs:    
> Figure 1: Four pressures versus A'.    
> Initially, there is no cyan vertical dashed line on the second panel.    
> Users need to select the boundary by manually clicking the cross point of two curves on the second (priority) or first panel.    
> Then Figures 2 & 3 will pop up.    
> <img width="600" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_pressures.png">   

> Figures 2 & 3: The reconstructed cross-sectional map & Pt' versus A'.    
> These two figures represent the default results obtained from the GSR (not the final yet).      
> <img width="300" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_cross_section.png">
> <img width="270" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_PtA.png">

### 2. Round 2
**Since lines above ```reconstruction(...)``` remain the same.    
Here only shows changes inside ```reconstruction(...)```.*    

```python
reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
               get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0)
```
I. The major change from the first round to the second round is to add the rest pressure terms in the extended GS equation, i.e., turning on ```pressureSwitch```.    
II. Since we are also satisfied with the current boundary of A' and would not like to select it again, we also set ```get_Ab=1```.    

Outputs:
> Figure 1: Four pressures versus A'.    
> Since the first round has ```pressureSwitch = 0```, the thermal pressure (3rd panel) was 0.    
> Now, all pressures are included.
> <img width="600" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_4_pressures.png">   

> Figures 2 & 3: The reconstructed cross-sectional map & Pt' versus A'.    
> These two figures are the **final** results from the GSR.      
> <img width="300" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_cross_section.png">
> <img width="270" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_PtA.png">


III. Additional changes can be made on ```dmid```, ```dAl0```, and ```dAr0``` if necessary.    
- III-1, ```dmid = -10```
    - This change will move the spacecraft upward.    
      > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_cross_section_dmid.png">
- III-2, ```dAl0 = 0.0, dAr0 = 0.0```
    - These two parameters control the percentages of extrapolation at left and right boundaries.
    - Usually can turn on ```checkPtAFitting``` to see whether they need changes.    


       ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   checkPtAFitting=True,)
        ```
    - These two figures show the normalized curves during the first (left) and second (right) round.    

      > <img width="250"        src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/first_round_check_PtA.png">
      > <img width="250"         src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_check_PtA.png">

### 3. Add-on features (recommend turning on after Round 2)
- **saveFig**: save figures.   
- **plotJz**: plot the map of the axial current density jz.
    - In ```reconstruction(...)```, add ```plotJz = True```.    
        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   plotJz=True)
        ```
        > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_Jz.png">    
    
- **plotHodogram**: plot hodograms of the magnetic field via the MVAB.
    - In ```reconstruction(...)```, add ```plotHodogram = True```.    
        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   plotHodogram=True)
        ```
        > <img width="400" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_hodogram.png">  
- **plotWalenRelation**: plot the Walen relation between V-remaining and VA.
    - In ```reconstruction(...)```, add ```plotWalenRelation = True```.    
        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   plotWalenRelation=True)
        ```
        > <img width="300" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_walen_relation.png">  

- **plotSpacecraftTimeSeries**: plot the time-series data from spacecraft.
   - In ```reconstruction(...)```, add ```plotSpacecraftTimeSeries = True```.    
        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   plotSpacecraftTimeSeries=True)
        ```
        > <img width="400" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_spacecraftTimeSeries.png">  
        
- **adjustInterval**: adjust the boundary of an interval, i.e., starting and/or ending times.    
    - *Default setting is to show the current interval with +/- 10 points.*    

- **checkPtAFitting**: shown above.
     
- **helicityTwist**: estimate the relative helicity and average twist.     
- **checkHT**: check whether the current HT frame is well-found.
     - In ```reconstruction(...)```, add ```checkHT = True```.    
        ```python
        reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
                   get_Ab=1, pressureSwitch=1, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                   checkHT=True)
        ```
        > <img width="250" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_checkHT.png">  
        
- **includeTe**: include the electron temperature in the transverse pressure Pt'.
- **includeNe**: include the electron number density in the transverse pressure Pt'.
- **Other Calculations and derived parameters** of a flux rope: mostly shown on the terminal.    
      - Parameters for the Grad-Shafranov reconstruction, e.g., the aforementioned second line    
      - Figure information      
      - The maximum of the axial magnetic field Bz   = 7.403238594695631 nT      
      - The maximum of the axial current density jz  = 3.241255038840692e-12 A/m^2    
      - Estimated toroidal magnetic flux             = 180038399696.94977 Wb    
      - Estimated poloidal magnetic flux at 1 AU     = 1152801302589.8374 Wb    
      - Estimated relative helicity                  = 0.00031948627788541526 nT^2/AU^2    
      - Estimated relative helicity per unit length  = 0.00046711112871995975 nT^2/AU^3    
      - Estimated average twist                      = 9.85648174146297e-27   


---

## Using the flux rope axis from the detection result
### Round 1
**Have to start over with the initial settings.*

```python
import pickle
import pandas as pd
import datetime
from ReconstructionMisc import reconstruction

rootDir = '/home/ychen/Desktop/PyGS/' # specify rootDir
inputFileName = 'events/2018_selected_events.p' # specify input file's name
# load flux rope list from detection result
SFR_detection_list = pd.read_pickle(open(rootDir + inputFileName,'rb')) 

# In reconstruction (...), specify spacecraft ID, FR_list, and eventNo.
# The second line shows the initial settings of the GSR.
# To obtain an axis from GSR, set **adjustAxis=True**
reconstruction(rootDir,spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=53,
               get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
               adjustAxis=True)
```
