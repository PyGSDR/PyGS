# PyGS
Magnetic flux rope detection &amp; reconstruction based on the extended Grad-Shafranov equation.

## Introduction
This package consists of two key products to serve as a set of comprehensive tools for the investigation of magnetic flux ropes (FRs) in space plasmas based on in-situ spacecraft measurements. See [fluxrope.info](http://www.fluxrope.info) for details. 
- The Grad-Shafranov (GS)-based detection (GSD):
  - Automatedly identify flux ropes and output their parameters
  - Support the purpose of the statistical analysis
  - Rely on the generalized version of the GS equation ([Teh 2018](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-018-0802-z))
  - Applicable to FRs with a broad definition including both static and dynamic structures
  - Applicable to PSP, Solar Orbiter, Ulysses, ACE, and WIND spacecraft datasets
  - Enhanced and streamlined from the original automated GSD (see [Dr. Jinlei Zheng's GitHub](https://github.com/AlexJinlei/Magnetic_Flux_Rope_Detection))
  - Final outputs:
    - an event list including selected flux rope parameters
    - a time-series plot showing flux rope intervals and selected plots showing part of the analysis results

- The Grad-Shafranov (GS) type reconstruction (GSR, [Hu & Sonnuerup 2002](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001JA000293)):
  - Visualize and characterize the 2D magnetic field configuration from 1D time-series data
  - Confirm the SFR detection results
  - Derive flux rope parameters, e.g., the poloidal and toroidal magnetic fluxes, the relative helicity, average twist, etc.
  - Useful for case studies
  - Final outputs: a set of figures characterizing flux rope properties

## Python & Dependencies
Python 3 | Numpy | SciPy | pandas | Matplotlib | ai.cdas | SpacePy

## Installations
For non-Python users*:
```shell
# Download PyGS & find where the file is, usually in downloads folder
# Also download "examples" folder since it includes necessary inputs for testing.
# On your terminal:
tar -zxvf PyGS-0.0.1.tar.gz
cd PyGS-0.0.1
python3 setup.py build
python3 setup.py install

# Launch Python3 to see if it works
import PyGS
```
**Pip3 install will be available shortly.*

## Basic Examples
- The Grad-Shafranov (GS)-based detection (GSD)
  - Please see [instruction_gsd](https://github.com/PyGSDR/PyGS/blob/main/documentation/instruction_gsd.md) for more information.
  ```python
  import datetime
  from PyGS.FluxRopeDetection import detection
  
  rootDir = '/home/ychen/Desktop/PyGS/examples/'
  # Notice the shock list file is needed.
  # Please make sure if you have specified the correct path to this file.
  shockList = rootDir + 'IPShock_ACE_or_WIND_or_Ulysses_1996_2016_DF.p' 

  if __name__ == "__main__":
      detection(rootDir, spacecraftID='WIND',
          timeStart=datetime(2018,10,31,18,0,0), timeEnd=datetime(2018,10,31,22,0,0),
          duration=(10,30),
          includeTe=True, includeNe=False,
          Search=True, CombineRawResult=True, GetMoreInfo=True,
          LabelFluxRope=True,B_mag_threshold=5.0, shockList_DF_path=shockList, allowIntvOverlap=False)
  ```
  > <img width="600" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/detection_FR_time_series.png">
- The Grad-Shafranov (GS)-based reconstruction (GSR)
  - Please see [instruction_gsr_example](https://github.com/PyGSDR/PyGS/blob/main/documentation/instruction_gsr_examples.md) for more information.
  ```python
  import pickle
  import pandas as pd
  import datetime
  from PyGS.ReconstructionMisc import reconstruction

  # Please specify the path to where the "examples" folder is saved.
  # Parameter settings here are supported by files in the "examples" folder.
  # You may follow the instruction to start over with the initial settings.

  rootDir = '/home/ychen/Desktop/PyGS/examples/'
  inputFileName = 'selected_event.p' # The file includes the flux rope parameters
  
  SFR_detection_list = pd.read_pickle(open(rootDir + inputFileName,'rb'))
  reconstruction(rootDir, spacecraftID='WIND', FR_list=SFR_detection_list, eventNo=0,
                 timeStart=datetime(2018,8,28,0,24,0), timeEnd=datetime(2018,8,28,0,32,0), 
                 adjustAxis=False, 
                 grid_x=15, grid_y=131, 
                 get_Ab=0, pressureSwitch=0, polyOrder=3, dmid=0, dAl0=0.0, dAr0=0.0,
                 includeTe=False, includeNe=False, saveFig=False, plotJz=False, 
                 plotHodogram=False, checkHT=False, plotWalenRelation=False, 
                 plotSpacecraftTimeSeries=False, adjustInterval=False, 
                 checkPtAFitting=False, helicityTwist=False)

  ```
  > <img width="370" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_cross_section_dmid.png"> 
  > <img width="325" src="https://github.com/PyGSDR/PyGS/blob/main/example_figures/second_round_PtA.png">

- The GSR function includes some calculations, e.g., HT frame analysis, MVAB, etc., which can be run independently.
  - For HT analysis, please see [instruction_HT_analysis](https://github.com/PyGSDR/PyGS/blob/main/documentation/instruction_HT_analysis.md).
  - For MVAB frame, please see [instruction_mvab](https://github.com/PyGSDR/PyGS/blob/main/documentation/instruction_mvab.md).

## Citations and References
- If using GSD, please cite [Hu et al. 2018](https://doi.org/10.3847/1538-4365/aae57d) & [Chen and Hu 2022](https://doi.org/10.3847/1538-4357/ac3487).    
- If using GSR, please cite [Hu & Sonnuerup 2002](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001JA000293) & [Chen and Hu 2022](https://doi.org/10.3847/1538-4357/ac3487).
- Flux rope events using the original GSD are available on [flux rope database](http://www.fluxrope.info).
- Python files for the original GSD are available on [Dr. Jinlei Zheng's GitHub](https://github.com/AlexJinlei/Magnetic_Flux_Rope_Detection).

## Acknowledgements
We appreciate the help and previous work of Dr. Jinlei Zheng who created the original GSD, and acknowledge the NASA grant 80NSSC23K0256 for funding.

## Notes
@Jun 27, 2023: 
This is a beta version.     
Please reach out to Dr. Yu Chen (yc0020@uah.edu) for any bugs.
