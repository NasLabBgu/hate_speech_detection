import os
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, auc, roc_curve, balanced_accuracy_score
import sys
f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, "../.."))
from config.detection_config import post_level_conf, post_level_execution_config
from detection.detection_utils.preprocessing import *
from detection.detection_utils.factory import factory
from utils.my_timeit import timeit
from utils.general import init_log
logger = init_log("post_level_multiple_experiments")


@timeit
def run_multiple_plm_experiments():
    result = pd.DataFrame(columns=["model"] + [metric.__name__ for metric in
                                               [f1_score, accuracy_score, balanced_accuracy_score, recall_score, precision_score, auc]])

    dataset_name = post_level_execution_config["data"]["dataset"]
    logger.info(f"Running experiment (all models) on {dataset_name} dataset...")
    data_conf = post_level_conf[dataset_name]
    data_path = data_conf["data_path"]
    file_ending = data_path.split(".")[-1]
    if file_ending == 'csv':
        sep = ','
    elif file_ending == 'tsv':
        sep = '\t'
    else:
        raise ValueError(f"wrong ending for file {data_path}")
    # load data
    df = pd.read_csv(data_path, sep=sep)
    df = df[[data_conf["text_column"], data_conf["label_column"]]]
    df = df.dropna().reset_index(drop=True)

    X = df[data_conf["text_column"]]
    y = df[data_conf["label_column"]]

    labels = data_conf["labels"]
    labels_interpretation = data_conf["labels_interpretation"]
    post_level_execution_config["kwargs"]["labels"] = labels
    post_level_execution_config["kwargs"]["labels_interpretation"] = labels_interpretation

    # preprocess
    preprocess_output_path = post_level_execution_config["preprocessing"]["output_path"]
    max_seq_len = post_level_execution_config["kwargs"]["max_seq_len"]
    bert_conf = post_level_execution_config["bert_conf"]
    max_features = post_level_execution_config["preprocessing"]["max_features"]

    train_on_all_data = False  # always here since its a comparison!!!
    test_size = post_level_execution_config['data']['test_size']

    # here you can change the models to compare
    for model_type in ["lr", "lightgbm", "catboost", "xgboost", "feed_forward_nn", "cnn_lstm", "attenion_lstm"]:#, "bert_transfer_learning_without_finetuning", "bert_transfer_learning_with_finetuning"]:
    # for model_type in ["bert_transfer_learning_without_finetuning", "bert_transfer_learning_with_finetuning"]:
        logger.info(f"Running experiment using the model: {model_type}")
        res_row = {"model": model_type}
        if model_type in ["lr", "catboost", "lightgbm", "xgboost"]:
            preprocess_type = "tfidf"
            if model_type == "lr":
                model_class = "models.MyLogisticRegression"
            elif model_type == "catboost":
                model_class = "models.MyCatboost"
            elif model_type == "lightgbm":
                model_class = "models.MyLightGBM"
            elif model_type == "xgboost":
                model_class = "models.MyXGBoost"
        elif model_type in ["feed_forward_nn", "cnn_lstm", "attenion_lstm"]:
            preprocess_type = "nn"
            if model_type == "feed_forward_nn":
                model_class = "models.FeedForwardNN"
            elif model_type == "cnn_lstm":
                model_class = "models.CNN_LSTM"
            elif model_type == "attenion_lstm":
                model_class = "models.AttentionLSTM"
        else:
            preprocess_type = "bert"
            model_class = "models.BertFineTuning"
            if model_type == "bert_transfer_learning_without_finetuning":
                post_level_execution_config["kwargs"]["fine_tune"] = False
            elif model_type == "bert_transfer_learning_with_finetuning":
                post_level_execution_config["kwargs"]["fine_tune"] = True

        pt = PreprocessText(max_features=max_features, max_seq_len=max_seq_len,
                            bert_conf=bert_conf, preprocess_type=preprocess_type,
                            output_path=preprocess_output_path, test_size=test_size)
        if train_on_all_data:
            X_train, _, y_train, _, _, _ = pt.full_preprocessing(X, y, mode='train')
        else:
            X_train, X_test, y_train, y_test, X_train_as_text, X_test_as_text = pt.full_preprocessing(X, y, mode='split')

        # pickle.dump(pt, open(os.path.join(preprocess_output_path, "preprocess_text.pkl"), "wb"))
        post_level_execution_config["kwargs"]["vocab_size"] = pt.vocab_size

        if train_on_all_data:
            if len(labels) > 2:
                y_train = pd.get_dummies(y_train)
            y_train = y_train.astype(int)
        else:
            # original_y_test = y_test.copy()
            if len(labels) > 2:
                y_train = pd.get_dummies(y_train)
                y_test = pd.get_dummies(y_test)
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

        # train model
        post_level_execution_config["kwargs"]["bert_conf"] = post_level_execution_config["bert_conf"]
        model = factory(model_class, **post_level_execution_config["kwargs"])
        model.fit(X_train, y_train)

        # evaluate model and save results
        if not train_on_all_data and X_test is not None:
            # y_true_dummy = y_test.copy()
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)

            fp, tp, th = roc_curve(y_true=y_test, y_score=y_score[:, -1])
            res_row[auc.__name__] = auc(fp, tp)
            for metric in [f1_score, accuracy_score, recall_score, precision_score]:
                # print(f"{metric.__name__}: {metric(y_test, y_pred):.2f}")
                res_row[metric.__name__] = metric(y_test, y_pred)


            for acc_metric in [accuracy_score, balanced_accuracy_score]:
                res_row[acc_metric.__name__] = acc_metric(y_test, y_pred)
            for metric in [f1_score, recall_score, precision_score]:
                if len(labels) == 2:
                    res_row[metric.__name__] = metric(y_test, y_pred)
                else:
                    for averaging_technique in ["micro", "macro", "weighted"]:
                        res_row[f"{metric.__name__}__{averaging_technique}"] = metric(y_test, y_pred,
                                                                                             average=averaging_technique)
            result = result.append(res_row, ignore_index=True)
            result.to_csv(os.path.join(f"detection/outputs/{dataset_name}", f"post_level_results__{int(test_size * 100)}_test_size.tsv"),
                  sep='\t', index=False)


if __name__ == '__main__':
    run_multiple_plm_experiments()
