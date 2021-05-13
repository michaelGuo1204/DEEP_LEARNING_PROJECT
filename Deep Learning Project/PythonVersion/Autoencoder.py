import tensorflow as tf

from Model import *

if __name__=='__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    if (os.path.exists('saved_model/base_model_512') or os.path.exists('./saved_model/attention_model')):
        autoencoder = tf.keras.models.load_model('saved_model/base_model_512')
        autoencoder.summary()
        autoencoder = Autoencoder_Model(load=True, loadNet=autoencoder)

    else:
        # %% Load Data
        data = load('XY.npz')
        X_onehot = data['arr_0']
        Y = data['arr_1']
        X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_train_test(X_onehot, Y, proportion=(0.8, 0.1, 0.1))

        # %% Set Model
        autoencoder = Autoencoder_Model()
        autoencoder.trainModel(X_train, X_val)

    # %% Get latent data by the encoder
    data = load('XY.npz')
    X_onehot = data['arr_0']
    Y = data['arr_1']
    # %% Make predictions by encoder
    autoencoder.encoderPredict(X_onehot, Y)
    # %% Classify the data
    autoencoder.knnClassify()
