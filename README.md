# PyGS
Magnetic flux rope detection &amp; reconstruction based on the extended Grad-Shafranov equation.

# Introduction
This package consists of two key products to serve as a set of comprehensive tools for the investigation of SFRs, i.e., detection and reconstruction. The first one is an enhancement of the original automated Grad-Shafranov (GS)-based detection (hereafter, GSD) of SFRs ([Zheng & Hu 2018](https://iopscience.iop.org/article/10.3847/2041-8213/aaa3d7), [Hu et al. 2018](https://iopscience.iop.org/article/10.3847/1538-4365/aae57d), [Dr. Jinlei Zheng's GitHub](https://github.com/AlexJinlei/Magnetic_Flux_Rope_Detection), with a flowchart available on [flux rope database](http://www.fluxrope.info/flowchart.html)), which supports the purpose of the statistical analysis. The enhancement (eGSD in short) is implemented through the generalized version of the GS equation ([Teh 2018](https://earth-planets-space.springeropen.com/articles/10.1186/s40623-018-0802-z)). It enables the original GSD to be applicable to SFRs with a broad definition including both static and dynamic structures and also to more spacecraft datasets. The second product is associated with the GS reconstruction (hereafter, GSR, [Hu & Sonnuerup 2002](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2001JA000293)), which can confirm the SFR detection results by visualizing and characterizing the two-dimensional (2D) magnetic field configuration from the one-dimensional (1D) time-series data and thus is also useful for case studies.

# Python & Dependencies
- Python 3.8 and later
- numpy
- ai.cdas
- scipy
- pandas
- matplotlib

# Installations
TBC...
