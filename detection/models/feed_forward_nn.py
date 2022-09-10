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
from keras.layers import Input, Embedding, Dense, Flatten, GlobalMaxPool1D, GlobalAvgPool1D, Dropout, AvgPool1D, Average, GlobalAveragePooling1D

initializer = tf.keras.initializers.GlorotNormal(seed=42)
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
        self.dense_units = kwargs['dense_units']
        self.bs = kwargs['bs']
        self.optimizer = 'adam'
        self.model = self.build_model()
        self.model.summary()

    def build_model(self):
        if self.pretrained_embeddings:
            self.vocab_size, self.emb_size = self.pretrained_embeddings.shape
            emb_layer = Embedding(input_dim=self.vocab_size, output_dim=self.emb_size,
                                  weights=[self.pretrained_embeddings],
                                  input_length=self.max_seq_len, trainable=self.finetune_embeddings)
        else:
            emb_layer = Embedding(input_dim=self.vocab_size, output_dim=self.emb_size, input_length=self.max_seq_len, embeddings_initializer=initializer)
        if self.labels_num == 2:
            output_layer = Dense(1, activation="sigmoid")
        else:
            output_layer = Dense(self.labels_num, activation="softmax")

        # Functional API
        if self.model_api == "functional": # Gab
            INPUT = Input(shape=(self.max_seq_len,))
            x = emb_layer(INPUT)
            # x = Dropout(0.3)(x)
            x = GlobalMaxPool1D()(x)
            # x = Dropout(0.2)(x)
            x = Dense(self.dense_units, activation='relu', kernel_initializer=initializer)(x)
            # x = Dropout(0.3)(x)
            # x = Dense(self.dense_units//2, activation='tanh', kernel_initializer=initializer)(x)
            # x = Dropout(0.2)(x)
            OUTPUT = output_layer(x)
            model = Model(inputs=INPUT, outputs=OUTPUT, name="feed_forward_nn")

        # Sequential API
        elif self.model_api == "sequential":
            model = Sequential()
            model.add(emb_layer)
            model.add(layers.Flatten())
            model.add(layers.Dense(self.dense_units, activation='relu'))
            model.add(output_layer)

        else:
            logger.error(
                f"Model-api type `{self.model_api}` is not supported. Please choose either `functional` or `sequential`")
            raise IOError(
                f"Model-api type `{self.model_api}` is not supported. Please choose either `functional` or `sequential`")

        if self.labels_num == 2:
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=[f1_m, 'accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=[f1_m, 'accuracy'])
        # plot_model(model, to_file='cnn_lstm_model.png', show_shapes=True)
        return model


if __name__ == '__main__':
    cnn_lstm = FeedForwardNN()
