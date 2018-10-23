# Predicting-Arrythmia-from-ECG-recordings
This project is done on the 2018 GDSO workshop: https://gdso.berkeley.edu/projects.html.

Abstract:
Predictive maintenance has become a billion dollar industry in recent years. It predicts when equipment failure might occur, and allows maintenance to be planned before the failure. In this project, we apply predictive maintenance to arrhythmia forecasting. Specifically, we build a machine-learning model for time series forecasting, and utilize it to predict arrhythmia beats from patients’ ECG signals. Our training and testing data comes from the MIT-BIH ECG database. With our model, we are able to predict with ~ 80% accuracy whether a patient will have an arrhythmia attack before it actually happens.
Introduction to the MIT-BIH database can be found here: https://physionet.org/physiobank/database/mitdb/


Files in the repository:
- 'data base download and visualization' contains visualization and some statistics of the MIT-BIH ECG signals.
- 'preprocess' contain denoise functions to the ECG signals.
- 'sampling' contain functions to draw time series samples from the ECG signals.
- 'main' contains the whole workflow and the prediction results.
