import sys
sys.path.append('/home/tlenz/afs/dnn_significance_loss') # needed for execution from laptop
import numpy as np
import os
os.environ['PYTHONHASHSEED'] = '0'
import time
import ConfigParser
import keras
import pandas
import visualization as vis
import significance_estimators as sig
import functions as fcn
import losses as loss
import architectures as arch
import random as rn
rn.seed(12345)
np.random.seed(1234) # for reproducibility
from tensorflow import set_random_seed
set_random_seed(1)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
source_data  = 'higgs' # 'susy'
#----------------------------------------------------------------------------------------------------
##### Read and prepare data #####
#################################
reload(fcn)
if source_data == 'higgs':
    config_name = "keras_higgs.cfg"
    data, features = fcn.read_higgs_data_from_csv("data/higgs-kaggle-challenge/training.csv")
elif source_data == 'susy':
    config_name = "keras_susy.cfg"
    data, features = fcn.read_susy_data_from_pkl("data/susy/combinedleonid.pkl")

config = ConfigParser.ConfigParser()
config.read(config_name)

data = fcn.add_train_weights(data)

augment_data_at_train_time = config.getboolean('KERAS','data_augmentation_train_time')
augment_data_at_test_time  = config.getboolean('KERAS','data_augmentation_test_time')
n_augmentations = int(config.get('KERAS','n_augmentations'))

print "\ntrain-time data augmentation = " + str(augment_data_at_train_time)
print "test-time data augmentation  = " + str(augment_data_at_test_time)
print "number of augmentations = " + str(n_augmentations)

X_train, X_test, Y_train, Y_test = fcn.prepare_df(data, features,
                                                  data_augmentation_train_time = augment_data_at_train_time,
                                                  data_augmentation_test_time  = augment_data_at_test_time,
                                                  n_augmentations = n_augmentations)
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
config.read(config_name)

# For reproducibility
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(12345)
np.random.seed(1234)
set_random_seed(1)

print ''
print 'batch_size = ' + str(config.get('KERAS','batch_size'))
print 'epochs     = ' + str(config.get('KERAS','epochs'))
print 'patience   = ' + str(config.get('KERAS','patience'))
print ''
#----------------------------------------------------------------------------------------------------
# Define the architecure (!)
architecture = getattr(arch, config.get('KERAS','architecture') )
model = architecture(num_inputs = len(features), num_outputs = 1)
print "\nArchitecture = " + str(architecture.__name__)
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
    loss_from_config = loss_function(float(config.get('PARAMETERS','s_exp')), float(config.get('PARAMETERS','b_exp')), float(config.get('KERAS','systematic')))
    # Set also Y to two dimensions
    Y_train = pandas.concat([Y_train['signal'],X_train["Weight_corrected_by_lumi"]], axis=1)
    Y_test  = pandas.concat([Y_test['signal'],X_test["Weight_corrected_by_lumi"]], axis=1)
    # Set sample weights correctly
    sample_weights_ = ''
else:
    loss_from_config = 'binary_crossentropy'
    sample_weights_ = 'train_weight'
    
#----------------------------------------------------------------------------------------------------
# Compile the model
loss_=loss_from_config
print "\nLoss = " + str(loss_.__name__)
    
model.compile(loss=loss_,
              optimizer=config.get('KERAS','optimizer'),
              metrics=[config.get('KERAS','metrics')])

d=model.summary()
#----------------------------------------------------------------------------------------------------
# Training or Reading model
if config.getboolean('KERAS','train'):
    # Train the network
    history =  fcn.train_model(model, X_train, Y_train, features, 
                               cb_list, config, sample_weights = sample_weights_)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#### Predict and visualize results ####
#######################################
# Prediction
#-----------
reload(fcn)
df_pred = fcn.make_prediction_higgs(model, X_test, Y_test, features, config)
#----------------------------------------------------------------------------------------------------
# Visualization
#--------------
reload(sig)
reload(vis)
reload(sig)
# Remove old plots from results folder
os.system('rm results/*')
# Make loss vs epochs plot
vis.plot_val_train_loss(history, plot_log = False)
vis.plot_val_train_loss(history, plot_log = True)
# Make classification plot
vis.plot_prediction(df_pred)
# Get significance estimates
optimal_cut_value = vis.plot_significances(df_pred, "Weight_corrected_by_lumi", history)
# Copy full framework to results folder
if not os.path.exists('results/framework'):
    os.makedirs('results/framework')

os.system('cp *.py results/framework/.')
os.system('cp *.cfg results/framework/.')
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#### Make prediction on test.csv for higgs kaggle challenge ####
################################################################
reload(fcn)
if source_data == 'higgs':
    data_test, features = fcn.read_higgs_data_from_csv("data/higgs-kaggle-challenge/test.csv")
    data_scaled         = fcn.prepare_features(data_test, features)
    df_pred_test        = fcn.make_prediction_higgs(model, data_scaled, None, features, config)
    df_csv              = fcn.make_kaggle_csv_file(df_pred_test,cut_value=optimal_cut_value, output_folder = "results")
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
