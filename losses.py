from __future__ import print_function
from keras import backend as K

def significanceLoss(expectedSignal,expectedBkgd):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''
    #print_op1  = tf.print('here 1',output_stream=sys.stdout)
    def sigLoss(y_true,y_pred):

        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)
        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))
        y_pred=K.print_tensor(y_pred)
        return -(s*s)/(s+b+K.epsilon()) #Add the epsilon to avoid dividing by 0

    return sigLoss

def paperLoss(expectedSignal,expectedBkgd,systematic):

    def sigLoss(y_true,y_pred):

        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)
        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))

        return (s+b)/(s*s + K.epsilon())

    return sigLoss

def asimovLossInvert(expectedSignal,expectedBkgd,systematic):

    def sigLoss(y_true,y_pred):

        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)
        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))
        sigB=systematic*b

        return 1./(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB)))


    return sigLoss


def asimovLossInvertWithReg(expectedSignal,expectedBkgd,systematic):

    def sigLoss(y_true,y_pred):

        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)
        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))
        sigB=systematic*b
        sigma_reg = 1.8410548  # this is  68% CL from the Neyman construction for N=0 -> thus the lowest statistical uncertainty that can be achieved (see : https://twiki.cern.ch/twiki/bin/viewauth/CMS/PoissonErrorBars)
        sigB = K.sqrt(sigB*sigB + sigma_reg*sigma_reg)

        return 1./(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB)))


    return sigLoss
