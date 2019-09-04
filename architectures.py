import keras

def susy(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(23, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dense(num_outputs,activation='softmax'))
    return model

def susy_2(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(23, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dense(23, activation='relu'))
    model.add(keras.layers.Dense(num_outputs,activation='softmax'))
    return model

def susy_2_with_do(num_inputs, num_outputs):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(23, input_shape = (num_inputs,), activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(23, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(num_outputs,activation='softmax'))
    return model


#     model = Sequential()
#     model.add(
#         Dense(
#             300, kernel_initializer="glorot_normal", activation="tanh",
#             kernel_regularizer=l2(1e-4),
#             input_dim=num_inputs))
#     model.add(
#         Dense(
#             300, kernel_initializer="glorot_normal", activation="tanh",
#             kernel_regularizer=l2(1e-4)))
#     model.add(
#         Dense(
#             300, kernel_initializer="glorot_normal", activation="tanh",
#             kernel_regularizer=l2(1e-4)))
#     model.add(
#         Dense(
#             num_outputs, kernel_initializer="glorot_normal", activation="softmax"))
#     model.compile(loss="mean_squared_error", optimizer=Adam(), metrics=['accuracy'])
# return model
