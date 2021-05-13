# from keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Activation
from tensorflow.keras.layers import RepeatVector, Dot, Concatenate, SimpleRNN
from tensorflow.keras.models import Model

from utils import *

# define constants, most are hyper parameters
latent_units = 256  # literature optimal value
Tx = 1444  # length of input
na = 32  # units in LSTM
ns = 64  # units in post attention RNN


# %% model2: LSTM
# copied from deep learning course
def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')


# Define shared layers as **global variables**
repeator = RepeatVector(Tx, name='repeat_hidden')
concatenator = Concatenate(axis=-1)
# the two densors are all used to train for alpha, not in the main path
densor1 = Dense(10, activation="tanh", name='dense1')
densor2 = Dense(1, activation="relu", name='dense2')
activator = Activation(softmax,
                       name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes=1)
# after attention
after_attention_layer = SimpleRNN(ns, return_state=False)  # TODO: na is another hyper parameter...
out_encoder_layer = Dense(latent_units, activation=softmax, name='output_encoder')


def one_step_attention_encoder(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # Use repeator to repeat s_prev (Tx times) to be of shape (m, Tx, n_s) so that you can concatenate it
    # with all hidden states "a"
    a = repeator(a)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])  # (m, Tx, 2*na+ns)
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate
    # energies" variable e.
    e = densor1(concat)  # (m, Tx, 10)
    # Use densor2 to propagate e through a small fully-connected neural network
    # to compute the "energies" variable energies.
    energies = densor2(e)  # (m, Tx, 1)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given
    # to the next (post-attention) LSTM-cell
    context = dotor([alphas, a])

    return context


def model2_encoder(input_shape, latent_units=256):
    # preparations
    m, Tx, input_cls = input_shape  # Tx=1440, cls=5
    X = Input(shape=(Tx, input_cls), name='input')  # we don't assign the number of samples in the input
    s0 = Input(shape=(ns,), name='post_attention_hidden')  # custom initial value (I don't know why...)
    post_attn_hidden = s0  # # the hidden state of the post attention layer
    output = []  # prepare a list for 256 outputs

    # model
    # pre-attention BiLSTM, no return seq; we just input the dimension but no output length
    encoder_lstm = Bidirectional(LSTM(na, return_sequences=False))(X)  # (m, Tx, input_cls) -> (m, Tx, 2*na)
    for i in range(latent_units):  # this range determines the output dimension
        # we share layers in the attention part
        # output all calculated attention vectors, which have been linear combined
        context = one_step_attention_encoder(encoder_lstm, post_attn_hidden)  # (m, Tx, 2*na) -> (m, 1, 2*na)
        # input the attention vectors to a simple RNN, and return sequence
        post_attn_hidden = after_attention_layer(context)  # (m, 1, 2*na) -> (m, 1, ns)
        # input the hidden state into dense layer to get an element of the latent representation
        out = out_encoder_layer(post_attn_hidden)  # (m, 1, ns) -> (m, 1, 1)
        output.append(out)
    model = Model(inputs=[X, s0], outputs=np.array(output).resize(m, latent_units))
    return model


# Define shared layers again for the decoder
repeator2 = RepeatVector(latent_units, name='repeat_hidden')
concatenator2 = Concatenate(axis=-1)
# the two densors are all used to train for alpha, not in the main path
densor1_2 = Dense(10, activation="tanh", name='dense1')
densor2_2 = Dense(1, activation="relu", name='dense2')
activator2 = Activation(softmax,
                        name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
dotor2 = Dot(axes=1)
# after attention
after_attention_layer2 = SimpleRNN(ns, return_state=True)
out_encoder_layer2 = Dense(Tx, activation=softmax, name='output_encoder')


def one_step_attention_decoder(a, s_prev):
    s_prev = repeator2(s_prev)
    concat = concatenator2([a, s_prev])
    e = densor1_2(concat)
    energies = densor2_2(e)
    alphas = activator2(energies)
    context = dotor2([alphas, a])

    return context


# decoder can have a different structure from encoder as long as it can regenerate the input sequences
def model2_decoder(input_shape, Tx):
    # it's not a good solution because I did not refer to anything but repeated LSTM and attention
    m, latent_units = input_shape  # latent = 256, Tx = 1440
    X = Input(shape=(latent_units))  # input shape=(m, latent_units)
    s0 = Input(shape=(ns,), name='post_attention_hidden')
    post_attn_hidden = s0
    output = []

    # model
    # na, ns value should change? since the input has only one value...
    decoder_lstm = LSTM(na, return_sequences=False)(X)
    for i in range(Tx):
        context = one_step_attention_decoder(decoder_lstm, post_attn_hidden)
        post_attn_hidden = after_attention_layer2(context, inputs=post_attn_hidden)
        out = out_encoder_layer2(post_attn_hidden)
        output.append(out)
    model = Model(inputs=[X, s0], outputs=np.array(output, shape=(m, Tx)))
    return model


# %% another version: sequential
# def model2(input_shape):
#     m, Tx, input_cls = input_shape
#     model = Sequential()
#     model.add(Bidirectional(LSTM(lstm_units, return_sequences=False)))  # not assigning activation
#     return model

'''
note:
1. `return_state` in LSTM means assigning x^{t} with y^{t-1}. In other words, the output at a time is going to 
be the input at the next time step.
2. `Dot` layer (two inputted tensors) seems to multiply the assigned dimension without changing others.
I tried to describe the rules:
- if we only assign one dimension, the two same dimension will be eliminated (1 by n dot n by 1)
- the two elements in the axes tuple (the outest, maybe list) refer to two tensors; len(list) could be > 2
refer to numpy.tensordot: https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html#numpy.tensordot
and: https://blog.csdn.net/wdh315172/article/details/105973194?utm_medium=distribute.pc_relevant.none-task-blog-title-7&spm=1001.2101.3001.4242
'''
