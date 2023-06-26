# PyGS
Magnetic flux rope detection &amp; reconstruction based on the extended Grad-Shafranov equation.

## Introduction
This package consists of two key products to serve as a set of comprehensive tools for the investigation of flux ropes.
- The Grad-Shafranov (GS)-based detection (GSD):
  - Automatedly identify flux ropes and output their parameters
  - Supports the purpose of the statistical analysis
  - Enhanced from the original automated GSD (see [Dr. Jinlei Zheng's GitHub](https://github.com/AlexJinlei/Magnetic_Flux_Rope_Detection))
    - A flowchart can be found on [flux rope database](http://www.fluxrope.info/flowchart.html)
  - Improvements in the new GSD:
    - Migrate from Python 2 to Python 3 
    - Use the generalized version of the GS equation ([Teh 2018](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-018-0802-z)).
    - Applicable to SFRs with a broad definition including both static and dynamic structures
    - Applicable to more spacecraft datasets
    - Enhance and streamline functions, calculations, processes, etc.

- The Grad-Shafranov (GS) type reconstruction (GSR, [Hu & Sonnuerup 2002](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001JA000293)):
  - Visualize and characterize the 2D magnetic field configuration from 1D time-series data
  - Confirm the SFR detection results
  - Derive flux rope parameters, e.g., the poloidal and toroidal magnetic fluxes, the relative helicity, average twist, etc.
  - Useful for case studies

## Instructions and Examples
For the GSD, please see [instruction_gsd](https://github.com/PyGSDR/PyGS/blob/main/instruction_gsd.md).    
For the GSR, please see [instruction_gsr_example](https://github.com/PyGSDR/PyGS/blob/main/instruction_gsr_examples.md).

## Python & Dependencies
- Python 3.8 and later
- NumPy
- ai.cdas
- SciPy
- pandas
- Matplotlib

## Installations
TBC...

## Citations and References

## Notes
