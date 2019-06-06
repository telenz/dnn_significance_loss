import numpy as np
np.random.seed(1235) # for reproducibility
import os
import argparse
import ConfigParser
import pandas
import keras
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
import time
import visualization as vis
import significance_estimators as sig
import functions as fcn
import losses as loss

#----------------------------------------------------------------------------------------------------
# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-t', dest='train', help='Train new model' , action='store_true')
args = parser.parse_args()
#----------------------------------------------------------------------------------------------------
# Read config
#------------
config = ConfigParser.ConfigParser()
config.read('keras.cfg')

s_exp = float(config.get('PARAMETERS','sig_xsec_times_eff')) * float(config.get('PARAMETERS','lumi'))
b_exp = float(config.get('PARAMETERS','bkg_xsec_times_eff')) * float(config.get('PARAMETERS','lumi'))

#----------------------------------------------------------------------------------------------------
# Make train and test dataframes
#-------------------------------
# Split dataset into test and training set
X_train, X_test, Y_train, Y_test = fcn.prepare_data()

#----------------------------------------------------------------------------------------------------
# Define the network (!)
#-----------------------
model = keras.models.Sequential()
model.add(keras.layers.Dense(23, input_shape = (X_train.shape[1],), activation='relu'))
model.add(keras.layers.Dense(2, activation='sigmoid'))

# Define callbacks
monitor_variable = 'val_loss'
if float(config.get('KERAS','validation_split')) == 0.: 
    monitor_variable = 'loss'
cb = keras.callbacks.EarlyStopping(monitor=monitor_variable, min_delta=0, patience=int(config.get('KERAS','patience')), verbose=1, mode='auto', baseline=None, restore_best_weights=True)
cb_list = [cb]

# Add optimizer options
keras.optimizers.Adam(lr=float(config.get('KERAS','learning_rate')),
                      #     beta_1=0.9,
                      #     beta_2=0.999,
                      #     epsilon=None,
                      #     decay=0.0,
                      #     amsgrad=False
                      )
#----------------------------------------------------------------------------------------------------
# Training or Reading model
#--------------------------
# Define the loss function
if config.get('KERAS','loss') != 'binary_crossentropy':
    loss_function = getattr( loss, config.get('KERAS','loss') )
    loss_from_config = loss_function(s_exp,b_exp)
else:
    loss_from_config = 'binary_crossentropy'

if args.train:

    # Train the network
    model.compile(loss=loss_from_config, optimizer=config.get('KERAS','optimizer'), metrics=[config.get('KERAS','metrics')])

    history =  model.fit(X_train.values,
                         keras.utils.to_categorical(Y_train),
                         epochs           = int(config.get('KERAS','epochs')),
                         batch_size       = int(config.get('KERAS','batch_size')),
                         validation_split = float(config.get('KERAS','validation_split')),
                         callbacks        = cb_list)

    # Save the model
    model_json = model.to_json()
    if not os.path.exists('models'):
        os.makedirs('models')
    with open("models/model.json", "w") as json_file:
        json_file.write(model_json)
        model.save_weights("models/model.h5")
        print("\n-- Saved model to disk. --\n")
else:
    # Load the model
    json_file = open('models/model.json', 'r')
    model     = model_from_json( json_file.read() )
    json_file.close()
    # Load weights into new model
    model.load_weights("models/model.h5")
    model.compile(loss=loss_from_config, optimizer=config.get('KERAS','optimizer'), metrics=[config.get('KERAS','metrics')])
    print("\n-- Loaded model from disk. -- \n")

#----------------------------------------------------------------------------------------------------
# Prediction for test data set
#-----------------------------
df_pred = fcn.make_prediction(model, X_test, Y_test)

#----------------------------------------------------------------------------------------------------
# Visualization
#--------------
vis.plot_prediction(df_pred)
if args.train: 
    vis.plot_val_train_loss(history)
vis.plot_significances(df_pred)

#----------------------------------------------------------------------------------------------------
# Save the output of this test in a folder
#----------------------------------------
folder_name = time.strftime("%Y_%m_%d_%H_%M_%S")
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.system('cp keras.cfg ' + folder_name +'/keras.cfg')
os.system('cp plots/classification.png ' + folder_name +'/.')
os.system('cp plots/significance_estimates.png ' + folder_name +'/.')
if args.train:
    os.system('cp plots/loss.png ' + folder_name +'/.')    
else:
    os.system('cp -r models/ ' + folder_name +'/.')    


#----------------------------------------------------------------------------------------------------
# Other stuff
# evaluate loaded model on test data
#score = model.evaluate(X_test[features], Y_test, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# Predict on test data
#ret = model.predict(X_test.values)
#AUC = roc_auc_score(Y_test, ret[:,1])
#print("Test Area under Cruve = {0}".format(AUC))

