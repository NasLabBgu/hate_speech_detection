from abc import ABC, abstractmethod
import os
import tensorflow as tf
import math
from detection.detection_utils.factory import create_dir_if_missing
import logging

logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from sklearn.metrics import *
from tensorflow.keras.callbacks import *
import datetime
import matplotlib.pyplot as plt

plt.style.use("ggplot")
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.labels = kwargs['labels']
        self.labels_num = len(kwargs['labels'])
        self.labels_interpretation = kwargs['labels_interpretation']
        self.paths = kwargs['paths']
        self.metric_dict = {"f1": f1_score, "accuracy": accuracy_score, "precision": precision_score,
                            "recall": recall_score, "balanced_accuracy": balanced_accuracy_score, "auc": roc_auc_score}
        self.bs = kwargs['bs']

    @abstractmethod
    def fit(self, X_train, y_train):
        logger.info(f"fitting model {self.name}")

    @abstractmethod
    def predict(self, X_test):
        logger.info(f"predicting using model {self.name}")

    @abstractmethod
    def predict_proba(self, X_test):
        logger.info(f"predicting proba using model {self.name}")


class NeuralNetworkModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_learning_rate_scheduler(self, max_learn_rate=5e-5, end_learn_rate=1e-7,
                                       warmup_epoch_count=10, total_epoch_count=90):

        def lr_scheduler(epoch):
            if epoch < warmup_epoch_count:
                res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
            else:
                res = max_learn_rate * math.exp(
                    math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (
                            total_epoch_count - warmup_epoch_count + 1))
            return float(res)

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

        return learning_rate_scheduler

    def fit(self, X_train, y_train, val=False):
        super().fit(X_train, y_train)
        train_output_path = self.paths['train_output']
        model_output_path = self.paths['model_output']

        create_dir_if_missing(train_output_path)
        create_dir_if_missing(model_output_path)

        log_dir = os.path.join(train_output_path, 'logs', datetime.datetime.now().strftime("%Y-%m-%d%--H:%M:%s"))
        tensorboard_callback = TensorBoard(log_dir=log_dir)

        epochs = self.kwargs['epochs']
        validation_split = self.kwargs['validation_split']
        validation_data = None
        if val:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_split,
                                                              random_state=0,
                                                              stratify=y_train)
            validation_data = (X_val, y_val)
            validation_split = 0.0
        model_weights_file_path = os.path.join(model_output_path, "weights_best.h5")
        model_file_path = os.path.join(model_output_path, "model_best.h5")
        # model_file_path = os.path.join(model_output_path, "model.h5")
        # full_model_file_path = os.path.join(model_output_path, "full_model.pkl")
        checkpoint = ModelCheckpoint(model_weights_file_path, monitor='val_loss', verbose=2, save_best_only=True,
                                     mode='min', save_weights_only=True)

        if self.name == 'BertFineTuning':
            reduce_lr = self.create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                            end_learn_rate=1e-7,
                                                            warmup_epoch_count=20,
                                                            total_epoch_count=epochs)
        else:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=2)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=2, mode='auto',
                                       restore_best_weights=True)
        callbacks = [reduce_lr, early_stopping, checkpoint]  # not using checkpoint since it's saving the model fully

        # class weights (handling imabalanced data)
        if len(y_train.shape) > 1:
            y_train_for_class_weight = \
                y_train[y_train == 1].stack().reset_index().drop(0, 1).set_index('level_0').rename(
                    columns={"level_1": "label"})["label"]
        else:
            y_train_for_class_weight = y_train.copy()
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train_for_class_weight), y=list(y_train_for_class_weight)
        )
        train_class_weights = dict(zip(np.unique(y_train_for_class_weight), class_weights))
        print(f"Using GPU: {tf.config.list_physical_devices('GPU')}")

        # fit the model
        # with tf.device("/device:GPU:0"):
        hist = self.model.fit(X_train, y_train, batch_size=self.bs, epochs=epochs, validation_data=validation_data,
                              validation_split=validation_split,
                              class_weight=train_class_weights, callbacks=callbacks)
        # save the model
        self.model.save_weights(model_weights_file_path)  # save model's weights
        # self.model.save(model_file_path, save_format='tf')  # save full model
        # pickle.dump(self, open(full_model_file_path, "wb"))

        # save plots of loss and accuracy during training
        hist_path = os.path.join(train_output_path, "history.pkl")
        with open(hist_path, "wb") as fout:
            pickle.dump(hist.history, fout)

        plt.figure()
        plt.title('Loss per epoch')
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='validation')
        plt.legend()
        loss_fn = os.path.join(train_output_path, "loss_graph.png")
        plt.savefig(loss_fn)
        if 'accuracy' in hist.history.keys():
            plt.figure()
            plt.title('Accuracy per epoch')
            plt.plot(hist.history['accuracy'], label='train')
            plt.plot(hist.history['val_accuracy'], label='validation')
            plt.legend()
            acc_fn = os.path.join(train_output_path, "acc_graph.png")
            plt.savefig(acc_fn)
        if 'f1_m' in hist.history.keys():
            plt.figure()
            plt.title('F1-score per epoch')
            plt.plot(hist.history['f1_m'], label='train')
            plt.plot(hist.history['val_f1_m'], label='validation')
            plt.legend()
            f1_fn = os.path.join(train_output_path, "f1_graph.png")
            plt.savefig(f1_fn)
        plt.figure()
        plt.title('Learning rate per epoch')
        plt.plot(hist.history['lr'], label='lr')
        plt.legend()
        lr_fn = os.path.join(train_output_path, "lr_graph.png")
        plt.savefig(lr_fn)

    def predict(self, X_test):
        super().predict(X_test)
        # if self.name == 'BertFineTuning':
        #     X_test = X_test["input_ids"]  # , X_test["token_type_ids"]]
        y_score = self.model.predict(X_test)

        if y_score.shape[-1] > 1:
            y_pred = y_score.argmax(axis=-1)
        else:
            y_pred = (y_score > 0.5).astype('int32')

        return pd.Series(y_pred.reshape((y_pred.shape[0],)))

    def predict_proba(self, X_test):
        super().predict_proba(X_test)
        # if self.name == 'BertFineTuning':
        #     X_test = X_test["input_ids"]  # , X_test["token_type_ids"]]
        y_score = self.model.predict(X_test)
        return y_score

    def find_optimal_threshold(self, y_true, y_pred, metric_name):
        thresholds = []
        for th in np.arange(0.1, 0.501, 0.01):
            th = np.round(th, 2)
            res = self.metric_dict[metric_name](y_true, (y_pred > th).astype(int))
            thresholds.append([th, res])
            logger.info(f"{metric_name} score at threshold {th}: {res}")

        thresholds.sort(key=lambda x: x[1], reverse=True)
        best_thresh = thresholds[0][0]
        best_score = thresholds[0][1]
        logger.info(f"Best threshold: {best_thresh} yielding {metric_name} score of {best_score}")
        return best_score, best_thresh
