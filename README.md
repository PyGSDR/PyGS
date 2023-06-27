# PyGS
Magnetic flux rope detection &amp; reconstruction based on the extended Grad-Shafranov equation.

## Introduction
This package consists of two key products to serve as a set of comprehensive tools for the investigation of flux ropes.
- The Grad-Shafranov (GS)-based detection (GSD):
  - Automatedly identify flux ropes and output their parameters
  - Support the purpose of the statistical analysis
  - Rely on the generalized version of the GS equation ([Teh 2018](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-018-0802-z))
  - Applicable to SFRs with a broad definition including both static and dynamic structures
  - Applicable to PSP, Solar Orbiter, Ulysses, ACE, and WIND
  - Enhance and streamline from the original automated GSD (see [Dr. Jinlei Zheng's GitHub](https://github.com/AlexJinlei/Magnetic_Flux_Rope_Detection))

- The Grad-Shafranov (GS) type reconstruction (GSR, [Hu & Sonnuerup 2002](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001JA000293)):
  - Visualize and characterize the 2D magnetic field configuration from 1D time-series data
  - Confirm the SFR detection results
  - Derive flux rope parameters, e.g., the poloidal and toroidal magnetic fluxes, the relative helicity, average twist, etc.
  - Useful for case studies

## Python & Dependencies
- Python 3.8 and later
| Numpy | SciPy | pandas | Matplotlib | ai.cdas | SpacePy

## Installations
TBC...

## Instructions and Examples
- For the GSD, please see [instruction_gsd](https://github.com/PyGSDR/PyGS/blob/main/instruction_gsd.md).    
- For the GSR, please see [instruction_gsr_example](https://github.com/PyGSDR/PyGS/blob/main/instruction_gsr_examples.md).    
- For HT analysis, please see [instruction_HT_analysis](https://github.com/PyGSDR/PyGS/blob/main/instruction_HT_analysis.md).
- For MVAB frame, please see [instruction_mvab](https://github.com/PyGSDR/PyGS/blob/main/instruction_mvab.md).

## Citations and References
- If using GSD, please cite [Hu et al. 2018](https://doi.org/10.3847/1538-4365/aae57d) & [Chen and Hu 2022](https://doi.org/10.3847/1538-4357/ac3487).    
- If using GSR, please cite [Hu & Sonnuerup 2002](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001JA000293) & & [Chen and Hu 2022](https://doi.org/10.3847/1538-4357/ac3487).
- Flux rope events using the original GSD are available on [flux rope database](http://www.fluxrope.info).

## Notes
@Jun 27, 2023: This is a test version. 
