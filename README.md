Dependencies (also in requirements.txt):
- Python 3.11.12
- NumPy: 2.0.2
- Sklearn: 1.6.1
- Tensorflow: 2.18.0
- Pyts: 0.13.0

Original dataset can be found here: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities. We have also included this in the data folder in our repo.

To process the data, run the getNumpyData.py file to output NumPy files containing original and normalized training and testing data as well as target labels. This is also included directly in the numpy_data folder. This data can then be used for training.

The training process can be found in Training.ipynb. The files from numpy_data must be included to run the file. The notebook will create some trained models, which are also included in the trained_models folder to avoid the long process of training.

Evaluation takes place in MetricsEvaluation.ipynb. It is necessary to use the models from Training.ipynb or the models provided in trained_models for evaluation. We simply output various metrics related to multiclass classification such as accuracy, weighted f1 score, weighted precision, and weighted recall. We also analyze the training and validation accuracy during the training process.