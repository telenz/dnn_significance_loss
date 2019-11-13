from __future__ import print_function
from keras import backend as K
import tensorflow as tf


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def significanceLoss( s_exp, b_exp, systematic = 0):

    def significanceLoss_(y_true,y_pred):

        # Split y_true_with_weights to y_true and weights
        # y_true, weights = tf.split(y_true_with_weights,[1,1],axis=1)

        # To normalize each batch to s_exp and b_exp calculate sum of weights for signal and bkg
        sig_weight = s_exp/K.sum( weights * y_true     )
        bkg_weight = b_exp/K.sum( weights * (1-y_true) )

        # Calculate s and b
        s = K.sum( y_pred * y_true     * weights * sig_weight )
        b = K.sum( y_pred * (1-y_true) * weights * bkg_weight )

        return -(s*s)/(s+b)

    return significanceLoss_

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def paperLoss(s_exp, b_exp, systematic):

    def paperLoss_(y_true_with_weights, y_pred):

        # Split y_true_with_weights to y_true and weights
        y_true, weights = tf.split(y_true_with_weights,[1,1],axis=1)

        # # Printing
        # y_pred  = tf.Print(y_pred,[y_pred],"y_pred = ",summarize=10)
        # weights = tf.Print(weights,[weights],"weights = ",summarize=10)
        # y_true  = tf.Print(y_true,[y_true],"y_true = ",summarize=10)

        # To normalize each batch to s_exp and b_exp calculate sum of weights for signal and bkg
        sig_weight = s_exp/K.sum( weights * y_true     )
        bkg_weight = b_exp/K.sum( weights * (1-y_true) )

        # Calculate s and b
        s = K.sum( y_pred * y_true     * weights * sig_weight )
        b = K.sum( y_pred * (1-y_true) * weights * bkg_weight )

        return (s+b)/(s*s)

    return paperLoss_

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def asimovLossInvert(s_exp, b_exp, systematic):

    def asimovLossInvert_(y_true_with_weights,y_pred):

        # Split y_true_with_weights to y_true and weights
        y_true, weights = tf.split(y_true_with_weights,[1,1],axis=1)

        # To normalize each batch to s_exp and b_exp calculate sum of weights for signal and bkg
        sig_weight = s_exp/K.sum( weights * y_true     )
        bkg_weight = b_exp/K.sum( weights * (1-y_true) )

        # Calculate s and b
        s = K.sum( y_pred * y_true     * weights * sig_weight )
        b = K.sum( y_pred * (1-y_true) * weights * bkg_weight )
        sigma_b = systematic * b

        return 1./(2*((s+b)*K.log((s+b)*(b+sigma_b*sigma_b)/(b*b+(s+b)*sigma_b*sigma_b))-b*b*K.log(1+sigma_b*sigma_b*s/(b*(b+sigma_b*sigma_b)))/(sigma_b*sigma_b)))

    return asimovLossInvert_

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def asimovLossInvertWithReg(s_exp, b_exp, systematic):

    # systematic = K.print_tensor(systematic, 'systematic=')

    def asimovLossInvertWithReg_(y_true_with_weights,y_pred):

        # Split y_true_with_weights to y_true and weights
        y_true, weights = tf.split(y_true_with_weights,[1,1],axis=1)

        # Contrain y_pred from below and above by epsilon
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())


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

        # To normalize each batch to s_exp and b_exp calculate sum of weights for signal and bkg
        sig_weight = s_exp/K.sum( weights * y_true     )
        bkg_weight = b_exp/K.sum( weights * (1-y_true) )

        # Calculate s and b
        s = K.sum( y_pred * y_true     * weights * sig_weight )
        b = K.sum( y_pred * (1-y_true) * weights * bkg_weight )

        sigma_b   = systematic * b
        sigma_reg = 1.8410548  # this is  68% CL from the Neyman construction for N=0 -> thus the lowest statistical uncertainty that can be achieved (see : https://twiki.cern.ch/twiki/bin/viewauth/CMS/PoissonErrorBars)
        sigma_b   = K.sqrt(sigma_b*sigma_b + sigma_reg*sigma_reg)

        loss_ = 1./(2*((s+b)*K.log((s+b)*(b+sigma_b*sigma_b)/(b*b+(s+b)*sigma_b*sigma_b))-b*b*K.log(1+sigma_b*sigma_b*s/(b*(b+sigma_b*sigma_b)))/(sigma_b*sigma_b)))

        return loss_

    return asimovLossInvertWithReg_

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def asimovLossInvertWithRegWeighted(s_exp, b_exp, systematic=0.0):

    def asimovLossInvertWithRegWeighted_(y_true_with_weights, y_pred):

        # with additional argument : https://stackoverflow.com/questions/48082655/custom-weighted-loss-function-in-keras-for-weighing-each-element
        # without additional argument : https://datascience.stackexchange.com/questions/25029/custom-loss-function-with-additional-parameter-in-keras

        # Split y_true_with_weights to y_true and weights
        y_true, weights = tf.split(y_true_with_weights,[1,1],axis=1)

        # To normalize each batch to s_exp and b_exp calculate sum of weights for signal and bkg
        sig_weight = s_exp/K.sum( weights * y_true     )
        bkg_weight = b_exp/K.sum( weights * (1-y_true) )

        # Calculate s and b
        s = K.sum( y_pred * y_true     * weights * sig_weight )
        b = K.sum( y_pred * (1-y_true) * weights * bkg_weight )

        sigma_b   = systematic*b
        sigma_reg = 1.8410548  # this is  68% CL from the Neyman construction for N=0 -> thus the lowest statistical uncertainty that can be achieved (see : https://twiki.cern.ch/twiki/bin/viewauth/CMS/PoissonErrorBars)
        sigma_b   = K.sqrt( sigma_b*sigma_b + sigma_reg*sigma_reg )

        return 1./(2*((s+b)*K.log((s+b)*(b+sigma_b*sigma_b)/(b*b+(s+b)*sigma_b*sigma_b))-b*b*K.log(1+sigma_b*sigma_b*s/(b*(b+sigma_b*sigma_b)))/(sigma_b*sigma_b)))

    return asimovLossInvertWithRegWeighted_

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
