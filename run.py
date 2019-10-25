import sys
sys.path.append('/home/tlenz/afs/dnn_significance_loss') # needed for execution from laptop
import numpy as np
import os
import time
import ConfigParser
#import pandas
import keras
#import tensorflow as tf
#from keras import backend as K
#from keras.models import model_from_json
#from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_auc_score
#from sklearn.utils import shuffle
import visualization as vis
import significance_estimators as sig
import functions as fcn
import losses as loss
import architectures as arch
np.random.seed(1234) # for reproducibility
from tensorflow import set_random_seed
set_random_seed(3)


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
##### Higgs data #####
######################
# Read config
config_name = "/home/tlenz/afs/dnn_significance_loss/keras_higgs.cfg"
config = ConfigParser.ConfigParser()
config.read(config_name)
# Read and prepare (e.g. scale) input data
reload(fcn)
data, features = fcn.read_higgs_data_from_csv("/home/tlenz/afs/dnn_significance_loss/data/higgs-kaggle-challenge/training.csv")
data = fcn.add_train_weights(data)
data = fcn.add_weight_corrected_by_lumi(data, data, config)
X_train, X_test, Y_train, Y_test = fcn.prepare_df(data, features)
X_train = fcn.add_weight_corrected_by_lumi(X_train, Y_train, config)
X_test  = fcn.add_weight_corrected_by_lumi(X_test, Y_test, config)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#### Define network and train ####
##################################
#----------------------------------------------------------------------------------------------------
reload(loss)
reload(arch)
reload(fcn)
#----------------------------------------------------------------------------------------------------
# Define the architecure (!)
input_weights = None
model, input_weights = arch.model_for_weights(num_inputs = len(features), num_outputs = 2)
#model = arch.susy_2(num_inputs = len(features), num_outputs = 2)
#----------------------------------------------------------------------------------------------------
# Define callbacks
cb = fcn.define_callbacks(config)
cb_list = [cb]
#----------------------------------------------------------------------------------------------------
# Add optimizer options
keras.optimizers.Adam(lr=float(config.get('KERAS','learning_rate')))
#----------------------------------------------------------------------------------------------------
# Get the right loss function (from the config)
if config.get('KERAS','loss') != 'binary_crossentropy':
    loss_function = getattr( loss, config.get('KERAS','loss') )
    #loss_from_config = loss_function(s_exp,b_exp,float(config.get('KERAS','systematic')))
    loss_from_config = loss_function(float(config.get('PARAMETERS','s_exp')), float(config.get('PARAMETERS','b_exp')), float(config.get('KERAS','systematic')))
    print loss_from_config
else:
    loss_from_config = 'binary_crossentropy'
    
#----------------------------------------------------------------------------------------------------
# Compile the model
if input_weights is not None:
    loss_=fcn.wrapped_partial(loss_from_config,weights=input_weights)
else:
    loss_=loss_from_config
    print loss_
print loss_

#Y_train  = fcn.encode_weights(Y_train['signal'],X_train['Weight_corrected_by_lumi'])
    
model.compile(loss=loss_,
              optimizer=config.get('KERAS','optimizer'),
              metrics=[config.get('KERAS','metrics')])

d=model.summary()
#----------------------------------------------------------------------------------------------------
# Training or Reading model
if config.get('KERAS','train')=="True":
    # Train the network
    history =  fcn.train_model(model, X_train, Y_train, features, 
                               cb_list, config, sample_weights = '')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#### Predict and visualize results ####
#######################################
# Prediction
#-----------
reload(fcn)
df_pred = fcn.make_prediction(model, X_test, Y_test, features, config)
#----------------------------------------------------------------------------------------------------
# Visualization
#--------------
reload(sig)
reload(vis)
reload(sig)
# Make loss vs epochs plot
vis.plot_val_train_loss(history, plot_log = False)
# Make classification plot
vis.plot_prediction(df_pred)
# Get significance estimates
vis.plot_significances(df_pred, "Weight_corrected_by_lumi")
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# =============================================================================
# # Save the output of this test in a folder
# #----------------------------------------
# folder_name = time.strftime("%Y_%m_%d_%H_%M_%S")
# if not os.path.exists(folder_name):
#     os.makedirs(folder_name)
# os.system('cp keras.cfg ' + folder_name +'/keras.cfg')
# os.system('cp plots/classification.png ' + folder_name +'/.')
# os.system('cp plots/significance_estimates.png ' + folder_name +'/.')
# if config.get('KERAS','train')=="True":
#     os.system('cp plots/loss.png ' + folder_name +'/.')
# else:
#     os.system('cp -r models/ ' + folder_name +'/.')
# #----------------------------------------------------------------------------------------------------
# =============================================================================
