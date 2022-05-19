from cProfile import label
from turtle import color
from sklearn import config_context
from tcn import TCN, tcn_full_summary
from keras.layers import Dense, Softmax, Input
from keras.models import Sequential
from keras.preprocessing import sequence
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from pre_process_data import data_process
import pandas as pd



'''
TCN(
    nb_filters=64,
    kernel_size=3,
    nb_stacks=1,
    dilations=(1, 2, 4, 8, 16, 32),
    padding='causal',
    use_skip_connections=True,
    dropout_rate=0.0,
    return_sequences=False,
    activation='relu',
    kernel_initializer='he_normal',
    use_batch_norm=False,
    use_layer_norm=False,
    use_weight_norm=False,
    **kwargs
)
nb_filters: Integer. The number of filters to use in the convolutional layers. Would be similar to units for LSTM. Can be a list.
kernel_size: Integer. The size of the kernel to use in each convolutional layer.
dilations: List/Tuple. A dilation list. Example is: [1, 2, 4, 8, 16, 32, 64].
nb_stacks: Integer. The number of stacks of residual blocks to use.
padding: String. The padding to use in the convolutions. 'causal' for a causal network (as in the original implementation) and 'same' for a non-causal network.
use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
activation: The activation used in the residual blocks o = activation(x + F(x)).
kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
use_batch_norm: Whether to use batch normalization in the residual layers or not.
use_layer_norm: Whether to use layer normalization in the residual layers or not.
use_weight_norm: Whether to use weight normalization in the residual layers or not.
kwargs: Any other set of arguments for configuring the parent class Layer. For example "name=str", Name of the model. Use unique names when using multiple TCN.
'''

# if time_steps > tcn_layer.receptive_field, then we should not
# be able to solve this task.
batch_size, time_steps, input_dim = None, None, 18
b, s, i = -1, 1, 18 #batch_size, time_steps, input_dim
n_splits=5
#nb, nb_test = 40, 20
data_path = "./data/my_data.csv"
checkpoint_path = "./"
model_path = "./my_model3"
logdir = "./log"

#tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

def model_def():
    tcn_layer = TCN(
        input_shape=(time_steps, input_dim),
        nb_filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32, 64],
        #input_shape=(time_steps, input_dim), 
        activation='LeakyReLU',
        )

    # The receptive field tells you how far the model can see in terms of timesteps.
    print('Receptive field size =', tcn_layer.receptive_field)

    # model definition
    model = Sequential([
        Input(shape=(time_steps, input_dim), name='input'),
        tcn_layer,
        # Dense(16, activation='sigmoid'),
        # Dense(8, activation='sigmoid'),
        Dense(1, activation='sigmoid', name='output'),
        #Softmax()
    ])

    model.build()
    #model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.metrics.BinaryAccuracy()])# categorical_crossentropy mse
    tcn_full_summary(model, expand_residual_blocks=False)
    return model

def plot_model(model, to_file='model.png'):
    tf.keras.utils.plot_model(
        model,
        to_file=to_file,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=200,
        layer_range=None,
        show_layer_activations=True,
        expand_nested=True,
    )


def train(model, epochs=10, verbose=2, data_path=data_path, n_splits=n_splits, checkpoint_path=checkpoint_path, enable_cb=True,save_best_only=True,monitor='val_binary_accuracy'):
    data_generator = data_process(data_path, n_splits=n_splits)
    # Create a callback that saves the model's weights
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=verbose,
                                                    save_best_only=save_best_only,
                                                    monitor=monitor)

    for _ in range(n_splits):
        # reshape the data for input
        x, y, x_test, y_test = next(data_generator)
        
        #print(x[:nb*b].shape, y[:nb*b].shape, x_test.shape, y_test.shape)
        print(x.shape, y.shape, x_test.shape, y_test.shape)
        #x, y, x_test, y_test = x[:nb*b].reshape(b, s, i), y[:nb*b].reshape(b, -1), x_test[:nb_test*b].reshape(b, s, i), y_test[:nb_test*b].reshape(b, -1)
        x, y, x_test, y_test = x.reshape(b, s, i), y, x_test.reshape(b, s, i), y_test
        print(x.shape, y.shape, x_test.shape, y_test.shape)

        # model.load_weights(checkpoint_path)

        result = model.fit(x, 
                  y,
                  epochs=epochs,
                  #validation_split=0.3
                  #batch_size=batch_size
                  callbacks=[cp_callback] if enable_cb else None,
                  validation_data=(x_test, y_test),
                  verbose=verbose
                  )

    return result
        
if __name__ == "__main__":
    model = model_def()
    
    
    plot_model(model, to_file='model.png')
    #DONE: serialize model, validate data, and normalize data at first. 
    #TODO:life long learning.
    
    model.load_weights(checkpoint_path)
    
    # uncomment to train the model
    train(model, 
          epochs=1, 
          verbose=2, 
          enable_cb=False, # not save the checkpoints
          data_path=data_path, checkpoint_path=checkpoint_path)

    g = data_process(data_path, n_splits=n_splits)
    x, y, x_test, y_test = next(g)
    
    # evaluate the model
    loss, acc = model.evaluate(x_test.reshape(b, s, i), y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    #save to .pb model files
    model.save(model_path)
    
    #print(model.predict(x[0, 0, :].reshape(1, 1, -1)))
    plt.plot(np.arange(np.shape(x)[0]), model.predict(x.reshape(b, s, i)), color='y', label='prediction')
    plt.scatter(np.arange(np.shape(x)[0]), y, color='r', label='real')
    plt.legend()
    plt.show()