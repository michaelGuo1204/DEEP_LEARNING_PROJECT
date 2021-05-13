# from keras.initializers import glorot_uniform
from tensorflow.keras import backend as K


class model_base:
    def __init__(self):
        _latent_units = 0

    def softmax(self, input, axis=1):
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
