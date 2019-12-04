import keras
from keras.regularizers import l2

def susy(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(23, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dense(num_outputs,activation='sigmoid'))
    return model

def susy_2(num_inputs, num_outputs):
    input_  = keras.layers.Input( (num_inputs,) )
    output_ = keras.layers.Dense(units=23,activation='relu')(input_)
    output_ = keras.layers.Dense(units=23,activation='relu')(output_)
    output_ = keras.layers.Dense(units=num_outputs,activation='sigmoid')(output_)
    model = keras.models.Model( inputs=input_, outputs=output_)
    return model

def susy_2_with_do(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(23, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(23, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_outputs,activation='sigmoid'))
    return model

def two_layers_with_do(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(200, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_outputs,activation='sigmoid'))
    return model

def three_layers_with_do(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(200, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(200, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(num_outputs,activation='sigmoid'))
    return model


def three_layers_with_do_and_l2(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(200, input_shape = (num_inputs,), activation='relu', kernel_regularizer=l2(1e-5)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(200, activation='relu', kernel_regularizer=l2(1e-5)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(200, activation='relu', kernel_regularizer=l2(1e-5)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_outputs,activation='sigmoid'))
    return model


def model_for_weights(num_inputs, num_outputs):
    input_        = keras.layers.Input( (num_inputs,) )
    input_weights = keras.layers.Input( (1,) )
    output_ = keras.layers.Dense(units=23,activation='relu')(input_)
    output_ = keras.layers.Dense(units=23,activation='relu')(output_)
    output_ = keras.layers.Dense(units=num_outputs,activation='sigmoid')(output_)
    model = keras.models.Model( inputs=[input_,input_weights],outputs=output_)
    return model, input_weights
