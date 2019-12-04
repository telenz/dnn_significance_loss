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

        # Contrain y_pred from below and above by epsilon
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # Split y_true_with_weights to y_true and weights
        y_true, weights = tf.split(y_true_with_weights,[1,1],axis=1)

        # y_true  = tf.Print(y_true,[y_true],"y_true = ",summarize=10)
        # y_pred  = tf.Print(y_pred,[y_pred],"y_pred = ",summarize=10)
        # weights = tf.Print(weights,[weights],"weights = ",summarize=10)

        # To normalize each batch to s_exp and b_exp calculate sum of weights for signal and bkg
        sig_weight = s_exp/K.sum( weights * y_true     )
        bkg_weight = b_exp/K.sum( weights * (1-y_true) )

        # Calculate s and b
        s = K.sum( y_pred * y_true     * weights * sig_weight )
        b = K.sum( y_pred * (1-y_true) * weights * bkg_weight )
        sigma_b = systematic * b

        # s  = tf.Print(s,[s],"s = ",summarize=10)
        # b  = tf.Print(b,[b],"b = ",summarize=10)

        # The variable 'condition' defines which approximation is used
        condition = s/b * sigma_b/(b+sigma_b*sigma_b)
        loss_ = tf.cond(condition < 0.01,
                        lambda: (b+sigma_b*sigma_b)/(s*s),
                        lambda: 1./(2*((s+b)*K.log((s+b)*(b+sigma_b*sigma_b)/(b*b+(s+b)*sigma_b*sigma_b))-b*b*K.log(1+sigma_b*sigma_b*s/(b*(b+sigma_b*sigma_b)))/(sigma_b*sigma_b)))
                        )

        # # Asimov in an equivalent form
        # loss_ = 1./(2*((s+b)*K.log(1+s/b) - (s+b+b*b/(sigma_b*sigma_b))*K.log(1+sigma_b*sigma_b*s/(b*b + b*sigma_b*sigma_b))))

        # Special case for systematic = 0
        if systematic == 0:
            loss_ = 1./(2*((s+b)*K.log(1+s/b)-s))

        return loss_

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

        # Calculate uncertainty
        sigma_b   = systematic * b
        sigma_reg = 1.8410548  # this is  68% CL from the Neyman construction for N=0 -> thus the lowest statistical uncertainty that can be achieved (see : https://twiki.cern.ch/twiki/bin/viewauth/CMS/PoissonErrorBars)
        sigma_b   = K.sqrt(sigma_b*sigma_b + sigma_reg*sigma_reg)

        # The variable 'condition' defines which approximation is used
        condition = s/b * sigma_b/(b+sigma_b*sigma_b)
        loss_ = tf.cond(condition < 0.01,
                        lambda: (b+sigma_b*sigma_b)/(s*s),
                        lambda: 1./(2*((s+b)*K.log((s+b)*(b+sigma_b*sigma_b)/(b*b+(s+b)*sigma_b*sigma_b))-b*b*K.log(1+sigma_b*sigma_b*s/(b*(b+sigma_b*sigma_b)))/(sigma_b*sigma_b)))
                        )

        return loss_

    return asimovLossInvertWithReg_

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def sigmoid_significance(s_exp, b_exp, alpha):

    def sigmoid_significance_(y_true_with_weights,y_pred):

        # Define an increasing alpha called alpha_incr
        alpha_incr = tf.Variable(alpha, dtype=tf.float32)
        assign_op = tf.assign(alpha_incr, alpha_incr+0.02 )

        # Split y_true_with_weights to y_true and weights
        y_true, weights = tf.split(y_true_with_weights,[1,1],axis=1)

        # Contrain y_pred from below and above by epsilon
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # To normalize each batch to s_exp and b_exp calculate sum of weights for signal and bkg
        sig_weight = s_exp/K.sum( weights * y_true     )
        bkg_weight = b_exp/K.sum( weights * (1-y_true) )

        # Apply sigmoid to make the function differentiable (the higher alpha the steeper the sigmoid)
        with tf.control_dependencies([assign_op]):
            # alpha_incr = tf.Print(alpha_incr,[alpha_incr],"alpha_incr = ",summarize=100)
            activation_s = tf.sigmoid(alpha_incr * (y_pred - 0.5))
            activation_b = tf.sigmoid(alpha_incr * (y_pred - 0.5))

        # Calculate s and b as condition
        s = K.sum( activation_s * y_true     * weights * sig_weight )
        b = K.sum( activation_b * (1-y_true) * weights * bkg_weight )

        br = 10.0
        radicand = 2 *( (s+b+br) * K.log(1.0 + s/(b+br)) -s)
        ams = K.sqrt(radicand)

        return 1./ams

    return sigmoid_significance_

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
