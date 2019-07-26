import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ConfigParser

config = ConfigParser.ConfigParser()
config.read('keras.cfg')

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
def prepare_data():
    
    features = [
        'HT','MET','MT','MT2W','n_jet',
        'n_bjet','sel_lep_pt0','sel_lep_eta0','sel_lep_phi0',
        'selJet_phi0','selJet_pt0','selJet_eta0','selJet_m0',
        'selJet_phi1','selJet_pt1','selJet_eta1','selJet_m1',
        'selJet_phi2','selJet_pt2','selJet_eta2','selJet_m2',
        ]

    # Read pickle file and drop columns that are not needed
    data = pandas.read_pickle('/nfs/dust/cms/user/tlenz/13TeV/2018/significance_loss/dfs/combinedleonid.pkl')
    data = data[features+['signal']]
    
    # Split dataframe to dfs that either contain all variables except 'signal' or containing only 'signal'
    X = data.drop(["signal"], axis=1, inplace=False)
    Y = data[["signal"]]
    input_dim = len(features)

    # Pre-process your data: scale mean to 0 and variance to 1 for all input variables
    ss = StandardScaler()
    X_scaled = pandas.DataFrame(ss.fit_transform(X),columns = X.columns, index = X.index) # addinf index =... is very important since wo the new df would have new indices which makes a concat later impossible
    #normalized_X = preprocessing.normalize(X)

    # Split dataset into test and training set -> use 30% for final testing
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.30, random_state=1143)

    return X_train, X_test, Y_train, Y_test

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def make_prediction(model, X_test, Y_test):

    s_exp = float(config.get('PARAMETERS','sig_xsec_times_eff')) * float(config.get('PARAMETERS','lumi'))
    b_exp = float(config.get('PARAMETERS','bkg_xsec_times_eff')) * float(config.get('PARAMETERS','lumi'))

    df_test_with_pred = pandas.concat([X_test,Y_test], axis=1)

    # Predict the classes for the test data
    prediction = model.predict(X_test, batch_size=int(config.get('KERAS','batch_size')))#[:,1]
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
