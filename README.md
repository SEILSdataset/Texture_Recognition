# ISMIR2021_madrigal

This repository contains the scripts and data necesary to reproduce the outcomes presented in the submission to ISMIR2021 entitled "Automatic recognition of word-painting in Renaissance Music: Features evaluation and  baseline results". In order to carry out the experiments follow the instructions below:

1. To extract the NLDs (Note Level Descriptors, i.e., Low Level Descriptors over time considering the Note as frame unit) and statistical functionals:
	execute extract_functionals_NLDs.py (the folder corpus should be in the same directory). 

2. To carry out the Principal Component Analysis and subsequent hypothesis testing:
	execute PCA.py (the funtionals previously extracted should be in the same directory)
	execute PCA_significance.Rmd (the PCA previously computed should be in the same directory)

3. To run the functionals-based models (SVM and MLP):
	execute run_models_functionals.py (the statistical funtionals previously extracted should be in the same directory (models' hyperparameters are automatically optimised for each of the setups as described in the paper)

4. To run the NLDs-based models (CNN and BLSTM-RNN):
	execute run_models_NLDs.py (the NLDs, i.e., the continuous LLDs and their Delta coefficients, as well as the categorical LLD previously extracted, should be in the same directory; again, models' hyperparameters are automatically optimised for each of the setups as described in the paper)

5. To run the fusion (MLP + CNN):
	execute run_models_fusion.py (the functionals and NLDs previously extracted should be in the same directory; again, models' hyperparameters are automatically optimised for each of the setups as described in the paper)

If you find the content of this repository useful, you might consider giving us a citation:

E. Parada-Cabaleiro, M. Schmitt, A. Batliner, B. Schuller, & M., Schedl (2021), Automatic recognition of word-painting in Renaissance Music: Features evaluation and  baseline results, in Proc. of ISMIR, Online event, to appear.
