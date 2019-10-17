import keras

def susy(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(23, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dense(num_outputs,activation='softmax'))
    return model

def susy_2(num_inputs, num_outputs):
    input_  = keras.layers.Input( (num_inputs,) )
    output_ = keras.layers.Dense(units=23,activation='relu')(input_)
    output_ = keras.layers.Dense(units=23,activation='relu')(output_)
    output_ = keras.layers.Dense(units=2,activation='softmax')(output_)
    model = keras.models.Model( inputs=input_, outputs=output_)
    return model

def susy_2_with_do(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(23, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(23, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_outputs,activation='softmax'))
    return model

def model_for_weights(num_inputs, num_outputs):
    input_        = keras.layers.Input( (num_inputs,) )
    input_weights = keras.layers.Input( (1,) )
    output_ = keras.layers.Dense(units=23,activation='relu')(input_)
    output_ = keras.layers.Dense(units=23,activation='relu')(output_)
    output_ = keras.layers.Dense(units=2,activation='softmax')(output_)
    model = keras.models.Model( inputs=[input_,input_weights],outputs=output_)
    return model, input_weights
