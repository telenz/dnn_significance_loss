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

def paperLoss(expectedSignal,expectedBkgd):

    def sigLoss(y_true,y_pred):

        sigWeight = expectedSignal/K.sum(y_true)
        bkgWeight = expectedBkgd/K.sum(1-y_true)
        s = sigWeight*K.sum(y_pred*y_true)
        b = bkgWeight*K.sum(y_pred*(1-y_true))

        return (s+b)/(s*s + K.epsilon())

    return sigLoss
