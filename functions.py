import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ConfigParser
import keras

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

    # Pre-process your data: scale mean to 0 and variance to 1 for all input variables (scale only features!)
    ss = StandardScaler()
    data_scaled = pandas.DataFrame(ss.fit_transform(data[features]),columns = data[features].columns, index = data.index) # add index =... is very important since wo the new df would have new indices which makes a concat later impossible

    # Add non-feature elements again
    data_list=list(data.columns)
    data_scaled_list=list(data_scaled.columns)
    non_feature_elements = list(set(data_list).difference(data_scaled_list))
    print "list of non-feature elements = " + str(non_feature_elements)
    if len(non_feature_elements) != 0 :
        data_scaled = data_scaled.join( data[non_feature_elements], how='inner')

    return data_scaled

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def prepare_df(data, features):

    # Scale data to mean=0 and stdv=1
    data_scaled = prepare_features(data, features)

    # Split dataframe to dfs that either contain all variables except 'signal' or containing only 'signal'
    X = data_scaled.drop(["signal"], axis=1, inplace=False)
    Y = data_scaled[["signal"]]

    # Split dataset into test and training set -> use 30% for final testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1143)

    return X_train, X_test, Y_train, Y_test

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
    df.to_csv('submission_tlenz.csv')

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
