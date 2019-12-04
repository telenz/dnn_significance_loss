import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ConfigParser
import keras
import math

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def read_susy_data_from_pkl(filename):

    features = [
        'HT','MET','MT','MT2W','n_jet',
        'n_bjet','sel_lep_pt0','sel_lep_eta0','sel_lep_phi0',
        'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',
        'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',
        'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2',
        ]

    # Read pickle file and drop columns that are not needed
    data = pandas.read_pickle(filename)
    data = data[features+['signal']]
    data = data.reset_index(drop=True)

    return data, features

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def read_higgs_data_from_csv(filename):

    data = pandas.read_csv( filename )

    features = list(data.columns)
    if 'Label' in features : features.remove("Label")
    if 'Weight' in features : features.remove("Weight")
    if 'EventId' in features : features.remove("EventId")

    data.rename(columns={'Label':'signal'},inplace=True)
    data.replace("s",1,inplace=True)
    data.replace("b",0,inplace=True)

    return data, features
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def prepare_features(data, features):

    # Replace all -999 values with -10 to make it closer to other values
    data.replace(-999.0,-10.0,inplace=True)

    # Pre-process your data: scale mean to 0 and variance to 1 for all input variables (scale only features!)
    ss = StandardScaler()
    data_scaled = pandas.DataFrame(ss.fit_transform(data[features]),columns = data[features].columns, index = data.index) # add index =... is very important since wo the new df would have new indices which makes a concat later impossible

    # Add non-feature elements again
    data_list=list(data.columns)
    data_scaled_list=list(data_scaled.columns)
    non_feature_elements = list(set(data_list).difference(data_scaled_list))
    print "\nList of non-feature elements = " + str(non_feature_elements)
    if len(non_feature_elements) != 0 :
        data_scaled = data_scaled.join( data[non_feature_elements], how='inner')

    return data_scaled

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def prepare_df(data, features, data_augmentation_train_time = False, data_augmentation_test_time = False, n_augmentations = 0):

    # Scale data to mean=0 and stdv=1
    data_scaled = prepare_features(data, features)

    # # Make pca (don't reduce number of variables for now)
    # data_scaled = make_pca(data_scaled, features, len(features))

    # Split dataset into test and training set -> use 30% for final testing
    data_train, data_test = train_test_split(data_scaled, test_size=0.30, random_state=1143)

    if data_augmentation_train_time:
        data_train = augment_data(data_train, n_augmentations = n_augmentations)
    if data_augmentation_test_time:
        data_test  = augment_data(data_test , n_augmentations = n_augmentations)

    # Split dataframe to dfs that either contain all variables except 'signal' or containing only 'signal'
    X_train = data_train.drop(["signal"], axis=1, inplace=False)
    Y_train = data_train[["signal"]]

    X_test = data_test.drop(["signal"], axis=1, inplace=False)
    Y_test = data_test[["signal"]]

    return X_train, X_test, Y_train, Y_test

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def augment_data(df, n_augmentations):

    # Define ouput dataframe
    df_augmented = df.copy()

    # Define phi transformations between alpha = [-1,1] with phi -> phi + alpha*pi
    spacing = 2./(n_augmentations)
    alpha_list = [-1 + x*spacing for x in range(0, n_augmentations)]
    if 0.0 in alpha_list:
        alpha_list.remove(0.0)
    print "\nFollowing phi transformations are applied: phi -> phi + alpha*pi with"
    print "alpha = " + str(["%0.2f" % i for i in alpha_list]) + "\n"

    # Non-invariant variables
    vars_for_eta_transformation = ["PRI_tau_eta", "PRI_lep_eta", "PRI_jet_leading_eta", "PRI_jet_subleading_eta"]
    vars_for_phi_transformation = ["PRI_tau_phi", "PRI_lep_phi", "PRI_jet_leading_phi", "PRI_jet_subleading_phi", "PRI_met_phi"]

    # Copy dataframe and transform all variables that are not invariant under eta and phi transformations
    df_copy = df.copy()
    for var in vars_for_eta_transformation:
        df_copy[var] = np.where(df_copy[var] != -10, -df_copy[var], df_copy[var])
    df_augmented = df_augmented.append(df_copy)

    for alpha in alpha_list:
        df_copy = df.copy()
        for var in vars_for_phi_transformation:
            df_copy[var] = np.where(df_copy[var] != -10   , df_copy[var] + alpha*math.pi, df_copy[var])
            df_copy[var] = np.where(df_copy[var] >= math.pi, df_copy[var]-2*math.pi      , df_copy[var])
            df_copy[var] = np.where(df_copy[var] < -math.pi, df_copy[var]+2*math.pi      , df_copy[var])
        df_augmented = df_augmented.append(df_copy)

    # Shuffle data and reset the index
    df_augmented = df_augmented.sample(frac=1).reset_index(drop=True)

    return df_augmented

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def make_pca(df, features, n_pca):

    print "\n.... Make PCA ....\n"

    # Get non-features for later
    columns = list(df.columns)
    non_feature_elements = list(set(columns).difference(features))
    print "\nList of non-feature elements = " + str(non_feature_elements)

    # Make principal component analysis
    pca = PCA(n_components = n_pca)
    principal_components = pca.fit_transform(df[features])
    principal_df = pandas.DataFrame(data = principal_components, columns = features)

    # Now scale the features again with the StandardScaler
    ss = StandardScaler()
    principal_df_scaled = pandas.DataFrame(ss.fit_transform(principal_df[features]), columns = principal_df[features].columns, index = principal_df.index) # add index =... is very important since wo the new df would have new indices which makes a concat later impossible

    # Add non-feature elements again
    if len(non_feature_elements) != 0 :
        principal_df_scaled = principal_df_scaled.join( df[non_feature_elements], how='inner')

    return principal_df_scaled

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def make_prediction(model, X_test, Y_test, features, config):

    s_exp = float(config.get('PARAMETERS','sig_xsec_times_eff')) * float(config.get('PARAMETERS','lumi'))
    b_exp = float(config.get('PARAMETERS','bkg_xsec_times_eff')) * float(config.get('PARAMETERS','lumi'))

    df_test_with_pred = pandas.concat([X_test,Y_test], axis=1)

    # Predict the classes for the test data
    prediction = model.predict(X_test[features], batch_size=int(config.get('KERAS','batch_size')))
    df_test_with_pred['pred_prob'] = prediction

    # Calculate weights
    df_test_with_pred['gen_weight']=1.
    s_norm = s_exp/np.sum(df_test_with_pred.loc[df_test_with_pred['signal']==1,'gen_weight'])
    b_norm = b_exp/np.sum(df_test_with_pred.loc[df_test_with_pred['signal']==0,'gen_weight'])
    df_test_with_pred['final_weight'] = df_test_with_pred['gen_weight']*s_norm
    df_test_with_pred.loc[df_test_with_pred['signal']==0,'final_weight'] = df_test_with_pred.loc[df_test_with_pred['signal']==0,'gen_weight']*b_norm

    return df_test_with_pred

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def make_prediction_higgs(model, X_test, Y_test, features, config):

    # Add the true label (not used in the prediction though)
    if Y_test is not None :
        df_test_with_pred = pandas.concat([X_test,Y_test['signal']], axis=1)
    else :
        df_test_with_pred = X_test
    # Predict the classes for the test data
    prediction = model.predict(X_test[features].values, batch_size=int(config.get('KERAS','batch_size')))
    df_test_with_pred['pred_prob'] = prediction

    return df_test_with_pred

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def make_kaggle_csv_file(df, cut_value = 0.5):

    # Write a csv file with the following entries:
    # EventId, Class (which is s or b), RankOrder (1= most bkg-like, 550000=most signal-like)
    df = df[['EventId','pred_prob']]
    df = df.sort_values(by=['pred_prob'])
    df = df.reset_index(drop=True)
    df.index = df.index+1
    df.index.name = 'RankOrder'
    df.loc[df['pred_prob']<cut_value ,'Class'] = 'b'
    df.loc[df['pred_prob']>=cut_value,'Class'] = 's'
    df = df.drop('pred_prob',axis=1)
    df = df.sort_values(by=['EventId'])
    df.to_csv('plots/submission_tlenz.csv')

    return df

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def add_weight_corrected_by_lumi(X, Y, config):

    """Adds a column with a weight that is multiplied with a global lumi event weight
    Calculated with s_exp+b_exp and the total sum of signal and background weights
    It assumes the same mixture of bkg and sig events as in the training.csv file
    Parameters:
    argument1 (panda dataframe): input dataframe
    argument2 (config file): global keras config
    Returns:
    dataframe including global event weight (global_weight)
    """

    # If Weight column does not yet exist create it
    if not 'Weight' in X.columns:
        X['Weight'] = 1

    # Get expected number of signal and background events from config
    s_exp = float(config.get('PARAMETERS','s_exp'))
    b_exp = float(config.get('PARAMETERS','b_exp'))

    # Get sum of event weights for signal and background
    weight_sum_sig = np.sum(X['Weight'][Y['signal']==1])
    weight_sum_bkg = np.sum(X['Weight'][Y['signal']==0])

    # Calculate global event weight
    X['Weight_corrected_by_lumi'] = 1.
    X.loc[Y['signal']==1,'Weight_corrected_by_lumi']=X['Weight']*s_exp/weight_sum_sig
    X.loc[Y['signal']==0,'Weight_corrected_by_lumi']=X['Weight']*b_exp/weight_sum_bkg

    return X

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def add_train_weights(data):

    """Adds a column with a weight called train_weight to weight both samples equally in the loss
    Calculated with the total sum of 'Weights' in the sample and the sum of 'Weights' of bkg and signal events
    Parameters:
    argument1 (panda dataframe): input dataframe
    Returns:
    dataframe including train_weight
    """

    # If Weight column does not yet exist create it
    if not 'Weight' in data.columns:
        data['Weight'] = 1

    # Get sum of all weights and sum of signal (background) weights
    sum_of_all_weights = data.sum(axis = 0, skipna = True)['Weight']
    sum_of_sig_weights = data[data['signal']==1].sum(axis = 0, skipna = True)['Weight']
    sum_of_bkg_weights = data[data['signal']==0].sum(axis = 0, skipna = True)['Weight']
    sig_class_weight = sum_of_all_weights/sum_of_sig_weights
    bkg_class_weight = sum_of_all_weights/sum_of_bkg_weights

    # Calculate train_weight with the sums calculated above
    data['train_weight'] = data['Weight']*(sig_class_weight*data['signal']+bkg_class_weight*(1-data['signal']))

    return data

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

from functools import partial,update_wrapper
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def define_callbacks(config):
    cb = keras.callbacks.EarlyStopping(
        monitor=config.get('KERAS','monitor_variable'),
        patience=int(config.get('KERAS','patience')),
        verbose=1,
        restore_best_weights=True)
    return cb

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def train_model(model, X_train, Y_train, features, cb_list, config, sample_weights = ''):

    if sample_weights is not '':
        weights_ =  X_train[sample_weights].values
    else:
        weights_ = None

    history =  model.fit(X_train[features].values,
                         Y_train.values,
                         epochs           = int(config.get('KERAS','epochs')),
                         batch_size       = int(config.get('KERAS','batch_size')),
                         validation_split = float(config.get('KERAS','validation_split')),
                         sample_weight    = weights_,
                         callbacks        = cb_list)

    return history

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

# def encode_weights(y_true_as_pd_series, weights_as_pd_series):

#     y_true_as_pd_series = y_true_as_pd_series.replace(0,-1)
#     y_true_as_pd_series = y_true_as_pd_series*weights_as_pd_series

#     return y_true_as_pd_series.to_frame("signal")

# # ------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------------------

# def decode_weights(y_true_as_pd_df):

#     weights = abs(y_true_as_pd_df[["signal"]])
#     y_true_as_pd_df["signal"]=y_true_as_pd_df["signal"].divide(weights["signal"])
#     y_true_as_pd_df["signal"] = (y_true_as_pd_df["signal"]+1)/2.

#     return y_true_as_pd_df, weights

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
