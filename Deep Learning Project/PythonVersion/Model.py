from tensorflow.keras import Input,Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from attention import Attention
from numpy import random
import os.path

def divide_train_test(X_all, Y_all, proportion=(0.7,0.2,0.1)):
    train, val, test = proportion
    random.seed(0)  # fixed random numbers

    m, T_x, num_classes = X_all.shape
    idx = random.rand(m)
    train_idx = [i for i in range(m) if idx[i] < train]
    test_idx = [i for i in range(m) if idx[i] > 1 - test]
    val_idx = list(set(range(m)).difference(set(train_idx)).difference(set(test_idx)))

    X_train = X_all[train_idx, :]
    Y_train = Y_all[train_idx, :]
    X_val = X_all[val_idx, :]
    Y_val = Y_all[val_idx, :]
    X_test = X_all[test_idx, :]
    Y_test = Y_all[test_idx, :]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


class Autoencoder_Model:
    def __init__(self,ifAttention=False):
        self._features=5 #There are five features in DNA's one-hot model
        self._sequence_long=1444 #The adjusted sequence long is 1444 bps
        self._lstm_neurons=256 #Set lstm neurons are 256
        self._train_epoch=5000 #Set train epoch 5000
        self._batch_size=1
        if ifAttention:
            self.isAttention=True
            self._autoencoder_model,self._encoder_model = self.attentionModel()
        else:
            self._autoencoder_model,self._encoder_model = self.baseModel()

    def attentionModel(self):
        inputs = Input(shape=(self._sequence_long, self._features))
        encoded = LSTM(self._lstm_neurons,
                       return_sequences=True,
                       activation="tanh",
                       )(inputs)
        decoded = Attention()(encoded)
        decoded = RepeatVector(self._sequence_long)(decoded)
        decoded = LSTM(self._features, return_sequences=True)(decoded)
        autoencoder = Model(inputs=inputs, outputs=decoded)
        encoder = Model(inputs=inputs, outputs=encoded)
        autoencoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        autoencoder.summary()
        plot_model(autoencoder,dpi=328,show_shapes=True,show_dtype=True)
        return autoencoder, encoder
    def baseModel(self):
        inputs = Input(shape=(self._sequence_long,self._features))
        encoded = LSTM(self._lstm_neurons)(inputs)
        decoded = RepeatVector(self._sequence_long)(encoded)
        decoded = LSTM(self._features, return_sequences=True)(decoded)
        autoencoder = Model(inputs=inputs, outputs=decoded)
        encoder = Model(inputs=inputs, outputs=encoded)
        autoencoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        autoencoder.summary()
        return autoencoder,encoder
    def trainModel(self,X_train,Y_train,X_valid,Y_valid):
        checkpoint_path = "./checkpoint/Attention_checkpoint.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        #self._autoencoder_model.load_weights(checkpoint_path)
        cp_callback = ModelCheckpoint(filepath=checkpoint_dir,
                                                         save_weights_only=True,
                                                         verbose=1)
        print("=========Begin=======")
        self._autoencoder_model.fit(X_train,X_train,epochs=self._train_epoch,verbose=1,validation_data=(X_valid,X_valid),callbacks=[cp_callback])
        self._autoencoder_model.save('saved_model/attention_model')
