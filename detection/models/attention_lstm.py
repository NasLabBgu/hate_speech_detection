from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from detection.models.base_model import NeuralNetworkModel
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, Bidirectional, Concatenate, Flatten, GRU, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from detection.models.base_model import f1_m

from keras.utils.vis_utils import plot_model

import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import logging
logger = logging.getLogger(__name__)
is_tf_with_gpu = tf.test.is_built_with_cuda()
if is_tf_with_gpu:
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 0:  # gpu available
        LSTM_LAYER = CuDNNLSTM
    else:
        LSTM_LAYER = LSTM

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.initializer = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super().get_config()
        # config['supports_masking'] = self.supports_masking
        # config['initializer'] = initializers.serialize(self.initializer)
        config['W_regularizer'] = regularizers.serialize(self.W_regularizer)
        config['u_regularizer'] = regularizers.serialize(self.u_regularizer)
        config['b_regularizer'] = regularizers.serialize(self.b_regularizer)
        config['W_constraint'] = constraints.serialize(self.W_constraint)
        config['u_constraint'] = constraints.serialize(self.u_constraint)
        config['b_constraint'] = constraints.serialize(self.b_constraint)
        config['bias'] = self.bias
        return config

class AttentionLSTM(NeuralNetworkModel):
    def __init__(self, **kwargs):
        kwargs["name"] = "AttentionLSTM"
        self.kwargs = kwargs
        super().__init__(**kwargs)
        self.max_seq_len = kwargs['max_seq_len']
        self.model_api = kwargs['model_api']

        if "pretrained_embeddings" in kwargs.keys():
            self.pretrained_embeddings = kwargs['pretrained_embeddings']
            self.finetune_embeddings = kwargs['finetune_embeddings']
        else:
            self.pretrained_embeddings = None
            self.emb_size = kwargs['emb_size']
            self.vocab_size = kwargs["vocab_size"]

        self.model = self.build_model()
        self.model.summary()
    def build_model(self):
        if self.pretrained_embeddings:
            self.vocab_size, self.emb_size = self.pretrained_embeddings.shape
            emb_layer = Embedding(input_dim=self.vocab_size, output_dim=self.emb_size,
                            weights=[self.pretrained_embeddings],
                            input_length=self.max_seq_len, trainable=self.finetune_embeddings)
        else:
            emb_layer = Embedding(input_dim=self.vocab_size, output_dim=self.emb_size, input_length=self.max_seq_len)
        if self.labels_num == 2:
            output_layer = Dense(1, activation="sigmoid")
        else:
            output_layer = Dense(self.labels_num, activation="softmax")

        # Functional API
        if self.model_api == 'functional':
            INPUT = Input(shape=(self.max_seq_len,))
            EMB = emb_layer(INPUT)
            x = Bidirectional(LSTM_LAYER(128, return_sequences=True))(EMB)
            x = Bidirectional(LSTM_LAYER(64, return_sequences=True))(x)
            x = AttentionWithContext()(x)
            x = Dense(64, activation="relu")(x)
            OUTPUT = output_layer(x)
            model = Model(inputs=INPUT, outputs=OUTPUT)

        # Sequential API
        elif self.model_api == 'sequential':
            model = Sequential()
            model.add(emb_layer)
            model.add(Bidirectional(LSTM_LAYER(128, return_sequences=True)))
            model.add(Bidirectional(LSTM_LAYER(64, return_sequences=True)))
            model.add(AttentionWithContext)
            model.add(Dense(64, activation="relu"))
            model.add(output_layer)
        else:
            logger.error(f"Model-api type `{self.model_api}` is not supported. Please choose either `functional` or `sequential`")
            raise IOError(f"Model-api type `{self.model_api}` is not supported. Please choose either `functional` or `sequential`")

        if self.labels_num == 2:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy'])

        return model

    def focal_loss(self, gamma=2, alpha=0.75):
        def focal_loss_fixed(y_true, y_pred):  # with tensorflow
            eps = 1e-12
            y_pred = K.clip(y_pred, eps,
                            1. - eps)  # improve the stability of the focal loss and see issues 1 for more information
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
                (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

        return focal_loss_fixed










if __name__ == '__main__':
    al = AttentionLSTM()
