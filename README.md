# Texture_Recognition

This repository contains the code necessary to reproduce the outcomes presented in the submission to ISMIR2021 entitled "Automatic Recognition of Texture in Renaissance Music". In order to carry out the experiments, please follow the instructions below:

1. To extract the NLDs (Note Level Descriptors, i.e., Low Level Descriptors over time considering the Note as frame unit) and statistical functionals:
	execute extract_functionals_NLDs.py (the folder 'corpus' must be in the same directory). 

2. To carry out the Principal Component Analysis and subsequent hypothesis testing:
	execute PCA.py (the funtionals previously extracted must be in the same directory)
	execute PCA_significance.Rmd (the PCA previously computed must be in the same directory)

3. To run the functionals-based models (SVM and MLP):
	execute run_models_functionals.py (the statistical funtionals previously extracted must be in the same directory; models' hyperparameters are automatically optimised for each of the setups as described in the paper)

4. To run the NLDs-based models (CNN and BLSTM-RNN):
	execute run_models_NLDs.py (the NLDs, i.e., the continuous LLDs and their Delta coefficients, as well as the categorical LLD previously extracted, must be in the same directory; again, models' hyperparameters are automatically optimised for each of the setups as described in the paper)

5. To run the fusion (MLP + CNN):
	execute run_models_fusion.py (the functionals and NLDs previously extracted must be in the same directory; again, models' hyperparameters are automatically optimised for each of the setups as described in the paper)

If you find the content of this repository useful, you might consider giving us a citation:

E. Parada-Cabaleiro, M. Schmitt, A. Batliner, B. Schuller, & M. Schedl (2021), Automatic Recognition of Texture in Renaissance Music, in Proc. of ISMIR, Online event, to appear.
