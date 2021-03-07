# ISMIR2021_madrigal

This repository contains the scripts and data necesary to reproduce the outcomes presented in the submission to ISMIR2021 entitled "Automatic recognition of word-painting in Renaissance Music". In order to carry out the experiments follow the instructions below:

1. To extract the LLDs and functionals:
	execute extract_functionals_LLDs.py (the folder corpus should be in the same directory). 

2. To carry out the Principal Component Analysis and subsequent hypothesis testing:
	execute PCA.py (the funtionals previously extracted should be in the same directory)
	execute PCA_significance.Rmd (the PCA previously computed should be in the same directory)

3. To run the functionals-based models (SVM and MLP):
	execute run_models_functionals.py (the funtionals previously extracted should be in the same directory; models' hyperparameters should be configured for each of the setups as described in the paper)

4. To run the LLDs-based models (CNN and BLSTM-RNN):
	execute run_models_LLDs.py (the LLDs and Delta coefficients previously extracted should be in the same directory; models' hyperparameters should be configured for each of the setups as described in the paper)

5. To run the fusion (MLP + CNN):
	execute run_models_fusion.py (the functionals, LLDs, and Delta coefficients previously extracted should be in the same directory; models' hyperparameters should be configured for each of the setups as described in the paper)

If you find the content of this repository useful, you might consider giving us a citation:

E. Parada-Cabaleiro, M. Schmitt, A. Batliner, B. Schuller, & M., Schedl (2021), Automatic recognition of word-painting in Renaissance Music, in Proc. of ISMIR, Online event, to appear.
