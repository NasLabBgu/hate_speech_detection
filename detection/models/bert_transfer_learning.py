import tensorflow
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, Bidirectional, GlobalMaxPool1D, Concatenate, Flatten, GRU, LSTM, Lambda, Layer, GlobalAveragePooling1D, BatchNormalization
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential, Model
from detection.models.base_model import f1_m, recall_m, precision_m
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
is_tf_with_gpu = tensorflow.test.is_built_with_cuda()
if is_tf_with_gpu:
    devices = tensorflow.config.list_physical_devices('GPU')
    if len(devices) > 0:  # gpu available
        # print("using cudnn lstm")
        LSTM_LAYER = CuDNNLSTM
    else:
        # print("using normal lstm")
        LSTM_LAYER = LSTM
from transformers import TFBertModel, TFRobertaModel, TFXLNetModel, TFDistilBertModel, TFBertEmbeddings
from transformers import BertConfig, RobertaConfig, XLNetConfig, DistilBertConfig
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert import BertModelLayer


from detection.models.base_model import NeuralNetworkModel
from detection.models.attention_lstm import AttentionWithContext
import logging
logger = logging.getLogger(__name__)

bert_models_mapping = {
    'bert': (BertConfig, TFBertModel),  # ['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased'],
    'roberta': (RobertaConfig, TFRobertaModel),  # ['roberta-base', 'roberta-large', 'roberta-large-mnli'],
    'distilroberta': (RobertaConfig, TFRobertaModel),  # ['distilroberta-base']
    'xlnet': (XLNetConfig, TFXLNetModel),  # ['xlnet-base-cased', 'xlnet-large-cased'],
    'distilbert': (DistilBertConfig, TFDistilBertModel),  # ['distilbert-base-uncased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-cased-distilled-squad']
}




class BertFineTuning(NeuralNetworkModel):
    def __init__(self, **kwargs):
        kwargs["name"] = kwargs['model_name']
        self.kwargs = kwargs
        super().__init__(**kwargs)
        self.bert_conf = kwargs['bert_conf']
        self.max_seq_len = kwargs['max_seq_len']
        self.model_api = kwargs['model_api']
        self.vocab_size = kwargs['vocab_size']
        self.fine_tune = kwargs['fine_tune']
        self.model = self.build_model()
        self.model.summary()
        self.config = {}

    def get_config(self):
        config = self.model.get_config().copy()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def build_model(self):
        """Creates a classification model.
        N.B. The commented out code below show how to feed a token_type_ids/segment_ids sequence (which is not needed in our case).

        """

        # pre-trained bert with config
        bert_pretrained_model_name = self.bert_conf['model_type']
        bert_config, bert_model = bert_models_mapping[bert_pretrained_model_name.split('-')[0]]
        if 'uncased' in bert_pretrained_model_name:
            # config = bert_config(dropout=0.2, attention_dropout=0.2)
            # config.output_hidden_states = False
            config = bert_config(dropout=0.2, attention_dropout=0.2)

        else:
            config = bert_config(dropout=0.2, attention_dropout=0.2)
        transformer_model = bert_model.from_pretrained(bert_pretrained_model_name, config=config)
        self.config = config
        # inputs
        idx = Input((self.max_seq_len), dtype="int32", name="input_idx")
        if self.bert_conf['use_masking']:
            masks = Input((self.max_seq_len), dtype="int32", name="input_masks")
            bert_out = transformer_model([idx, masks])[0]

        else:
            bert_out = transformer_model(idx)[0]

        ## BI-LSTM
        if self.fine_tune:

            x = Bidirectional(LSTM_LAYER(128, return_sequences=True))(bert_out)
            x = Bidirectional(LSTM_LAYER(64, return_sequences=True))(x)
            x = AttentionWithContext()(x)
            x = Dense(64, activation="relu")(x)


            # x = Bidirectional(LSTM_LAYER(50, return_sequences=True))(bert_out)
            # x = GlobalMaxPool1D()(x)
            # x = Dense(50, activation='relu')(x)
            x = Dropout(0.2)(x)
        else:
            x = GlobalAveragePooling1D()(bert_out)

        ## BATCHNORM
        # cls_token = bert_out[:, 0, :]
        # x = BatchNormalization()(cls_token)
        # x = Dense(192, activation='relu')(x)
        # x = Dropout(0.2)(x)

        # fine-tuning
        # x = GlobalAveragePooling1D()(bert_out)
        # x = Dense(256, activation="relu")(x)
        # x = Dense(128, activation="relu")(x)
        # x = Dense(64, activation="relu")(x)
        if self.labels_num == 2:
            output_layer = Dense(1, activation="sigmoid")(x)
        else:
            output_layer = Dense(self.labels_num, activation="softmax")(x)
        # compile
        if self.bert_conf['use_masking']:
            model = Model([idx, masks], output_layer)
        else:
            model = Model(idx, output_layer)

        for layer in model.layers:
            if "bert" in layer.name or "xl_net" in layer.name:
                print(f"disabling params training of layer: {layer.name}")
                logger.info(f"disabling params training of layer: {layer.name}")
                layer.trainable = False
            # else:
            #     for layer in model.layers:
            #         layer.trainable = False
        # else:
        #     # inputs
        #     idx = Input((self.max_seq_len), dtype="int32", name="input_idx")
        #     masks = Input((self.max_seq_len), dtype="int32", name="input_masks")
        #     segments = Input((self.max_seq_len), dtype="int32", name="input_segments")
        #     # pre-trained bert
        #     nlp = TFBertModel.from_pretrained("bert-base-uncased")
        #     bert_out, _ = nlp([idx, masks, segments])
        #     # fine-tuning
        #     x = GlobalAveragePooling1D()(bert_out)
        #     x = Dense(64, activation="relu")(x)
        #     if self.labels_num == 2:
        #         output_layer = Dense(1, activation="sigmoid")(x)
        #     else:
        #         output_layer = Dense(self.labels_num, activation="softmax")(x)
        #
        #     model = Model([idx, masks, segments], output_layer)
        #     # if self.training:
        #     #     for layer in model.layers[:4]:
        #     #         layer.trainable = False
        #     # else:
        #     #     for layer in model.layers:
        #     #         layer.trainable = False

        if self.labels_num == 2:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_m, 'accuracy'])

        return model