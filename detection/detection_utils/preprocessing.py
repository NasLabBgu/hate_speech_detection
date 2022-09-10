from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import functools
from detection.detection_utils.text_features_extraction import tokenize, stopwords
from detection.detection_utils.factory import create_dir_if_missing
from sklearn.model_selection import train_test_split
import transformers
from utils.constants import URL_RE, RT_RE
import logging
logger = logging.getLogger(__name__)

class PreprocessText():
    def __init__(self, max_features, max_seq_len: int=128, bert_conf: dict=None, preprocess_type: str='bert', output_path: str=None,
                 test_size: float=0.20):
        self.max_features = max_features
        self.max_seq_len = max_seq_len
        self.bert_conf = bert_conf
        self.output_path = output_path
        self.preprocess_type = preprocess_type
        self.test_size = test_size
        create_dir_if_missing(self.output_path)
        self.vocab_size = -1
        self.tokenizer = None

    def inital_preprocessing(self, X):
        X = X.apply(lambda text: URL_RE.sub('', text))  # Remove URLs
        X = X.apply(lambda text: RT_RE.sub('', text))  # Remove RT from tweets
        X = X.apply(lambda text: text.replace("…", " "))
        X = X.apply(lambda text: text.strip())
        return X

    def get_word2index(self):
        return self.tokenizer.word_index

    def fit_tokenizer(self, texts, max_features=None):
        tokenizer = Tokenizer(num_words=max_features, oov_token="UNK")
        tokenizer.fit_on_texts(texts)
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.word_index) + 1
    def transform_tokenizer(self, texts):
        """
        important: oov tokens will be considered as 1
        :param texts:
        :return:
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences

    def pad_sequences_(self, texts):
        return pad_sequences(texts, maxlen=self.max_seq_len, dtype="long", truncating="post", padding="post")

    def get_word_vocab(self, texts, normalization_type, training=True):
        if training:
            word_vectorizer = CountVectorizer(
                # vectorizer = sklearn.feature_extraction.text.CountVectorizer(
                analyzer='word',
                tokenizer=functools.partial(tokenize, normalization_type=normalization_type),
                #preprocessor=strip_hashtags,
                ngram_range=(1, 1),
                stop_words=stopwords,  # We do better when we keep stopwords
                decode_error='replace',
                max_features=self.max_features,
                min_df=2,
                max_df=0.99
            )

            counts = word_vectorizer.fit_transform(texts).toarray()
            word2index = {v: i for i, v in enumerate(word_vectorizer.get_feature_names())}
            self.count_vectorizer = word_vectorizer
            self.word2index = word2index
            self.vocab_size = len(word2index)
            # pickle.dump(word_vectorizer, open(os.path.join(self.output_path, "count_vectorizer.pkl"), "wb"))
            # pickle.dump(word2index, open(os.path.join(self.output_path, "word2index.pkl"), "wb"))

        else:
            # word_vectorizer = pickle.load(open(os.path.join(self.output_path, "count_vectorizer.pkl"), "rb"))
            counts = self.count_vectorizer.transform(texts).toarray()

        word_embedding_input = []
        for tweet in counts:
            tweet_vocab = []
            for i in range(0, len(tweet)):
                if tweet[i] != 0:
                    tweet_vocab.append(i)
            word_embedding_input.append(tweet_vocab)
        return word_embedding_input

    def bert_preprocessing(self, X, training):
        # todo: CHANGE THIS TO SUPPORT NEW VOCABULARY ?
        bert_type = self.bert_conf['model_type']
        use_masking = self.bert_conf['use_masking']
        use_token_types = False  #self.bert_conf['use_token_types']
        if training:
            do_lower_case = False
            if 'uncased' in bert_type:
                do_lower_case = True
                token_1_to_add = 'rt'
            else:
                token_1_to_add = 'RT'
            token_2_to_add = '((('
            token_3_to_add = ')))'
            token_4_to_add = '(((('
            token_5_to_add = '))))'

            bert_tokenizer = transformers.AutoTokenizer.from_pretrained(bert_type, do_lower_case=do_lower_case)
            bert_tokenizer.add_tokens([token_1_to_add, token_2_to_add, token_3_to_add, token_4_to_add, token_5_to_add])
            if hasattr(bert_tokenizer, 'vocab'):
                self.vocab_size = len(bert_tokenizer.vocab)
            self.tokenizer = bert_tokenizer

        input_dict = self.tokenizer.batch_encode_plus(X, add_special_tokens=True, max_length=self.max_seq_len,
                                   truncation_strategy='longest_first', pad_to_max_length=True,
                                   return_attention_mask=use_masking, return_token_type_ids=use_token_types)

        input_ids = input_dict['input_ids']

        if use_masking:
            attention_masks = input_dict['attention_mask']
            X = [np.asarray(input_ids, dtype='int32'), np.asarray(attention_masks, dtype='int32')]
        else:
            X = np.asarray(input_ids, dtype='int32')
        return X

    def fit_tfidf(self, texts):
        tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_df=0.95)
        tfidf_vectorizer.fit(texts.values)
        self.tfidf_vectorizer = tfidf_vectorizer

    def transform_tfidf(self, texts):
        tfidf_features = self.tfidf_vectorizer.transform(texts.values)
        return pd.DataFrame(tfidf_features.todense())

    def full_preprocessing(self, X, y, mode):
        """

        :param X: text columns (pd.Series)
        :param y: label (pd.Series
        :param mode: either: split, train or test
        :return:
        """
        logger.info(f"Preprocessing text in mode: {mode}")
        X = self.inital_preprocessing(X)
        # split to train and test
        if mode == 'split':

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=self.test_size, random_state=42)
            X_train_as_text = X_train.copy()
            X_test_as_text = X_test.copy()

            if self.preprocess_type == 'bert':
                X_train = self.bert_preprocessing(X_train.tolist(), training=True)
                X_test = self.bert_preprocessing(X_test.tolist(), training=False)

            elif self.preprocess_type == 'nn':
                self.fit_tokenizer(X_train)
                X_train = self.transform_tokenizer(X_train)
                X_test = self.transform_tokenizer(X_test)
                X_train = self.pad_sequences_(X_train)
                X_test = self.pad_sequences_(X_test)

            elif self.preprocess_type == 'tfidf':
                self.fit_tfidf(X_train)
                X_train = self.transform_tfidf(X_train)
                X_test = self.transform_tfidf(X_test)

        # only train
        elif mode == 'train':
            X_train_as_text = X.copy()
            X_test_as_text = None
            if self.preprocess_type == 'bert':
                X_train = self.bert_preprocessing(X, training=True)
                y_train = y.copy()
                X_test = None
                y_test = None

            elif self.preprocess_type == 'nn':
                self.fit_tokenizer(X)
                X_train = self.transform_tokenizer(X)
                X_train = self.pad_sequences_(X_train)
                y_train = y.copy()
                X_test = None
                y_test = None
            elif self.preprocess_type == 'tfidf':
                self.fit_tfidf(X)
                X_train = self.transform_tfidf(X)
                y_train = y.copy()
                X_test = None
                y_test = None

        # only test
        elif mode == 'test':
            X_train_as_text = None
            X_test_as_text = X.copy()
            X_train = None
            y_train = None
            y_test = None
            if self.preprocess_type == 'bert':
                X_test = self.bert_preprocessing(X.tolist(), training=False)
            elif self.preprocess_type == 'nn':
                X_test = self.transform_tokenizer(X)
                X_test = self.pad_sequences_(X_test)
            elif self.preprocess_type == 'tfidf':
                X_test = self.transform_tfidf(X)
        else:
            raise ValueError(f"mode not supported: {mode}. try `split`, `train`, or `test`")
        return X_train, X_test, y_train, y_test, X_train_as_text, X_test_as_text