import pandas as pd
from detection.models.base_model import NeuralNetworkModel
from sklearn.feature_extraction.text import CountVectorizer
import pickle, os
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, GlobalMaxPool1D
from keras.utils.vis_utils import plot_model
import logging
logger = logging.getLogger(__name__)
is_tf_with_gpu = tf.test.is_built_with_cuda()
from detection.models.base_model import f1_m


class FeedForwardNN(NeuralNetworkModel):

    def __init__(self, **kwargs):
        kwargs["name"] = "FeedForwardNN"
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
        if self.model_api == "functional":
            INPUT = Input(shape=(self.max_seq_len,))
            x = emb_layer(INPUT)
            x = GlobalMaxPool1D()(x)
            x = Dense(10, activation='relu')(x)
            OUTPUT = output_layer(x)
            model = Model(inputs=INPUT, outputs=OUTPUT, name="feed_forward_nn")

        # Sequential API
        elif self.model_api == "sequential":
            model = Sequential()
            model.add(emb_layer)
            model.add(layers.Flatten())
            model.add(layers.Dense(10, activation='relu'))
            model.add(output_layer)

        else:
            logger.error(f"Model-api type `{self.model_api}` is not supported. Please choose either `functional` or `sequential`")
            raise IOError(f"Model-api type `{self.model_api}` is not supported. Please choose either `functional` or `sequential`")

        if self.labels_num == 2:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy'])
        # plot_model(model, to_file='cnn_lstm_model.png', show_shapes=True)
        return model


if __name__ == '__main__':
    cnn_lstm = FeedForwardNN()