from __future__ import print_function
from keras import backend as K
import tensorflow as tf


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def significanceLoss( expectedSignal=1 , expectedBkgd=1 , systematics_not_used=0):
    '''Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size'''
    #print_op1  = tf.print('here 1',output_stream=sys.stdout)
    def sigLoss(y_true,y_pred):

        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)
        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))

        return -(s*s)/(s+b)

    return sigLoss

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def paperLoss(expectedSignal,expectedBkgd,systematic):

    def sigLoss(y_true,y_pred):

        # Print y_true and y_pred
        # y_true = tf.Print(y_true,[y_true],"y_true = ",summarize=10)
        # y_pred = tf.Print(y_pred,[y_pred],"y_pred = ",summarize=10)

        # y_pred = y_pred[:,0]
        # y_true = y_true[:,0]

        # Print y_true and y_pred
        # y_true = tf.Print(y_true,[y_true],"y_true = ",summarize=10)
        # y_pred = tf.Print(y_pred,[y_pred],"y_pred = ",summarize=10)

        # Calculate s and b
        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)
        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))

        # Print s and b
        # s = K.print_tensor(s, 's=')
        # b = K.print_tensor(b, 'b=')

        return (s+b)/(s*s)

    return sigLoss

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def asimovLossInvert(expectedSignal,expectedBkgd,systematic):

    def sigLoss(y_true,y_pred):

        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)
        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))
        sigB=systematic*b

        return 1./(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB)))

    return sigLoss

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def asimovLossInvertWithReg(expectedSignal,expectedBkgd,systematic):

    # systematic = K.print_tensor(systematic, 'systematic=')

    def sigLoss(y_true,y_pred):

        # Contrain y_pred from below and above by epsilon
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # Take only first column of y_pred and y_true -> study this further
        # y_pred = y_pred[:,1]
        # y_true = y_true[:,1]


        ############################   Printing #####################################
        #print(K.int_shape(y_true))

        # y_pred = K.print_tensor(y_pred, 'y_pred 1 =')
        # y_true = K.print_tensor(y_true, 'y_true=')
        # a = y_true*y_pred
        # a = K.print_tensor(a, 'y_pred*y_true=')
        # y_true = tf.Print(y_true,[y_true],"y_true = ",summarize=10)
        # y_pred = tf.Print(y_pred,[y_pred],"y_pred = ",summarize=10)
        # a = tf.Print(a,[a],"y_pred*y_true = ",summarize=100)
        ############################   Printing #####################################

        # Calculate an overall signal and background weight
        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)

        sigWeight = 1.
        bkgWeight = 1.

        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))

        # s = K.print_tensor(s, 's=')
        # b = K.print_tensor(b, 'b=')

        sigB=systematic*b
        sigma_reg = 1.8410548  # this is  68% CL from the Neyman construction for N=0 -> thus the lowest statistical uncertainty that can be achieved (see : https://twiki.cern.ch/twiki/bin/viewauth/CMS/PoissonErrorBars)
        sigB = K.sqrt(sigB*sigB + sigma_reg*sigma_reg)

        loss = 1./(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB)))

        return loss

    return sigLoss

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def asimovLossInvertWithRegWeighted(systematic):

    def sigLoss(y_true,y_pred,weights):

        # with additional argument : https://stackoverflow.com/questions/48082655/custom-weighted-loss-function-in-keras-for-weighing-each-element
        # without additional argument : https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras

        #y_true , weights = decode_weights(y_true)
        # y_true  = y_true/weights
        # y_true  = (y_true+1)/2.

        #weights = K.print_tensor(weights, 'weights=')

        #s = K.sum(y_pred * y_true * weights     ) #* weights)
        #b = K.sum(y_pred * (1-y_true) * weights ) #* weights)
        s = K.sum(y_pred * y_true  ) #* weights)
        b = K.sum(y_pred * (1-y_true) ) #* weights)
        sigB=systematic*b
        sigma_reg = 1.8410548  # this is  68% CL from the Neyman construction for N=0 -> thus the lowest statistical uncertainty that can be achieved (see : https://twiki.cern.ch/twiki/bin/viewauth/CMS/PoissonErrorBars)
        sigB = K.sqrt(sigB*sigB + sigma_reg*sigma_reg)

        return 1./(2*((s+b)*K.log((s+b)*(b+sigB*sigB)/(b*b+(s+b)*sigB*sigB))-b*b*K.log(1+sigB*sigB*s/(b*(b+sigB*sigB)))/(sigB*sigB)))

    return sigLoss

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
