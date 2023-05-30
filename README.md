# Rough Volatility Thesis
Code implementation for our thesis on Rough Volatility.


# Structure of repo
In the script impvol_spx.py the implied volatility plot and ATM skew for May 15, 2019 is produced. In data_analysis_spx.py we compute the rest of the analysis for the Motivation section.

In the script Fractional Brownian Motion.py is run, the results for the Fractional Brownian Motion section is created. 

The model objects have been placed in the rModels foder. If these are run, some of the plots will be produced. The plots will be placed in folder called Output which you will need to create before running the code.

For rBergomi, there will be created the first 10 paths for the asset price and the variance. Moreover, it will do a sanity check on the variance  process.

For rHeston, it will will do the analysis of the kernels as described in the multi-factor approximation section, and it will plot the mean asset price path and variance path with confidence intervals.

The implied volatility plots presented in the thesis can be created by running the two scripts in the ImpliedVolPlots folder.
The multi-factor comparison and the multi-factor experiment can be performed by running the scripts in the Multifactor folder.