# EPFA-CLS : Deriving Controllable Local Optimal Solutions through an Environment Parameter Fixed Algorithm
This is the official repository for our recent work: [pdf](https://www.mdpi.com/2076-3417/13/12/7110)
## Overview
<p align="center"><img src="/fig/fig/ecparameter.png"  width="60%" height="60%"></p>

**EPFA-CLS**: An algorithm for finding the local optimal control parameters (*CLS*) in a deep learning regression network for a specific set of environment parameters.  
**Control Parameter**: The parameter to be controlled (optimized).  
**Environment Parameter**: The parameter that cannot be controlled (constant)  

This algorithm can be used in the following scenarios:
### Boston housing dataset
<p align="center"><img src="/fig/fig/boston.png"  width="70%" height="70%"></p>

The figure represents the distribution of optimized results obtained by setting all values from the Boston Housing dataset as initial values and introducing four random control parameters. The histograms depict the control parameters before and after optimization using each of the NOX, RM, DIS, and TAX control parameters, compared to the initial values shown in black. The distribution of Boston Housing Prices is shown in black to observe the changes in housing prices after optimization. The red color represents NOX, yellow represents RM, blue represents DIS, and green represents TAX.

### Optimal course dataset
<p align="center"><img src="/fig/fig/optimalcourse.png"  width="70%" height="70%"></p>

The 3D graph represents the relationship between the control parameters and the output of the function after fixing the environmental parameters. It starts from the initial value represented by the red dot and eventually reaches the optimized point represented by the blue dot through the optimization process. In this context, the blue dot corresponds to the derived *CLS*.

## Usage
### 0. Prepare the dataset
### 1. Training
### 2. EPFA-CLS
### Custom Inputs
## Citation
If you think this implementation is useful for your work, please cite our paper:
```
@Article{jangEPFA_CLS,
          AUTHOR = {Jang, Ohtae and Jo, Sangho and Kim, Sungho},
          TITLE = {Deriving Controllable Local Optimal Solutions through an Environment Parameter Fixed Algorithm},
          JOURNAL = {Applied Sciences},
          VOLUME = {13},
          YEAR = {2023},
          NUMBER = {12},
          ARTICLE-NUMBER = {7110},
          URL = {https://www.mdpi.com/2076-3417/13/12/7110},
          ISSN = {2076-3417},
          DOI = {10.3390/app13127110}
}
```

