# General imports

import numpy as np
import scipy.io
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Custom imports
from modules import RC_model

# ============ RC model configuration and hyperparameter values ============
config = {}
#config['dataset_name'] = 'JpVow'
#config['dataset_name'] = 'Gesture_datasettr.csv'

config['seed'] = 1
np.random.seed(config['seed'])

# Hyperarameters of the reservoir
config['n_internal_units'] = 100        # size of the reservoir
config['spectral_radius'] = 0.55        # largest eigenvalue of the reservoir
config['leak'] = 1.0                    # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['connectivity'] = 0.5            # percentage of nonzero connections in the reservoir
config['input_scaling'] = 0.1            # scaling of the input weights
config['noise_level'] = 0.01            # noise in the reservoir state update
config['n_drop'] = 3                    # transient states to be dropped
config['bidir'] = True                  # if True, use bidirectional reservoir
config['circ'] = False                  # use reservoir with circle topology

# Dimensionality reduction hyperparameters
config['dimred_method'] ='tenpca'       # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
config['n_dim'] = 100                  # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
config['mts_rep'] = 'reservoir'         # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
config['w_ridge_embedding'] = 3.0  # regularization parameter of the ridge regression

# Type of readout
config['readout_type'] = 'mlp'          # readout used for classification: {'lin', 'mlp', 'svm'}

# Linear readout hyperparameters
config['w_ridge'] = 5.0                 # regularization of the ridge regression readout

# SVM readout hyperparameters
config['svm_gamma'] = 0.005             # bandwith of the RBF kernel
config['svm_C'] = 5.0                   # regularization for SVM hyperplane

# MLP readout hyperparameters
config['mlp_layout'] = (5, 62)          # neurons in each MLP layer
config['num_epochs'] = 100             # number of epochs
config['w_l2'] = 0.001                  # weight of the L2 regularization
config['nonlinearity'] = 'tanh'         # type of activation function {'relu', 'tanh', 'logistic', 'identity'}


print(config)

# ============ Load dataset ============
#df = pd.read_csv("../dataset/"+config['dataset_name'])
df = pd.read_csv('../dataset/Rawdata.csv')
data = df.to_numpy()

data_set = data.reshape(90,5,62)
Xtr = data_set
print("Xtre-shape=",Xtr.ndim)
data_Xte = pd.read_csv('../dataset/Rawdata.csv')
data_Xte = data_Xte.to_numpy()
data_Xte = data_Xte.reshape(90,5,62)
Xte = data_Xte

label_Ytr = pd.read_csv('../dataset/Rawlabeltr.csv')
label_Ytr = label_Ytr['label'].to_numpy()
Ytr = label_Ytr.reshape(-1, 1)
#from sklearn.model_selection import train_test_split
#Xtr, Xte, Ytr, Yte = train_test_split(data_set, data_label, test_size=0.2, random_state=42)


label_Yte = pd.read_csv('../dataset/Rawlabeltest.csv')
label_Yte = label_Yte['label'].to_numpy()
Yte = label_Yte.reshape(-1, 1)            # Convert to 2D array

#print('Loaded ' + config['dataset_name'] + ' - Tr: ' + str(Xtr.shape)+', Te: ' + str(Xte.shape))
print(' - Tr: ' + str(Xtr.shape)+', Te: ' + str(Xte.shape))
print(' - Test: ' + str(Ytr.shape)+', Te: ' + str(Yte.shape))


# One-hot encoding for labels
onehot_encoder = OneHotEncoder(sparse=False)
ytr = onehot_encoder.fit_transform(Ytr)
yte = onehot_encoder.transform(Yte)

# ============ Initialize, train and evaluate the RC model ============
classifier = RC_model(
                        reservoir=None,     
                        n_internal_units=config['n_internal_units'],
                        spectral_radius=config['spectral_radius'],
                        leak=config['leak'],
                        connectivity=config['connectivity'],
                        input_scaling=config['input_scaling'],
                        noise_level=config['noise_level'],
                        circle=config['circ'],
                        n_drop=config['n_drop'],
                        bidir=config['bidir'],
                        dimred_method=config['dimred_method'], 
                        n_dim=config['n_dim'],
                        mts_rep=config['mts_rep'],
                        w_ridge_embedding=config['w_ridge_embedding'],
                        readout_type=config['readout_type'],            
                        w_ridge=config['w_ridge'],              
                        mlp_layout=config['mlp_layout'],
                        num_epochs=config['num_epochs'],
                        w_l2=config['w_l2'],
                        nonlinearity=config['nonlinearity'], 
                        svm_gamma=config['svm_gamma'],
                        svm_C=config['svm_C']
                        )

tr_time = classifier.train(Xtr, ytr)
print('Training time = %.2f seconds'%tr_time)

Y_output = Yte.flatten()
accuracy, f1, Y_predict = classifier.test(Xte, yte)
print('Accuracy = %.2f, F1 = %.2f \n' % (accuracy, f1))

#############################################
from sklearn.metrics import confusion_matrix, classification_report

cf_matrix = confusion_matrix(Y_output, Y_predict )
print('confusion_matrix : \n', cf_matrix)
print(classification_report(Y_output, Y_predict, digits=4))

''' 
import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.heatmap(cf_matrix, annot=True, fmt='0.1%', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()'''
