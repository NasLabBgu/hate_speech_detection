import os
import sys
f = os.path.dirname(__file__)
# import sys
sys.path.append(os.path.join(f, "../.."))
import json
import pickle
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, auc, roc_curve, balanced_accuracy_score

from config.detection_config import post_level_conf, post_level_execution_config
from detection.detection_utils.preprocessing import *
from detection.detection_utils.factory import factory
from utils.my_timeit import timeit
from utils.general import init_log
logger = init_log("post_level_single_experiments")

@timeit
def run_single_plm_experiment():
    dataset_name = post_level_execution_config["data"]["dataset"]
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
    df = pd.read_csv(data_path, sep=sep, engine='python')
    df = df[[data_conf["text_column"], data_conf["label_column"]]]
    df = df.dropna().reset_index(drop=True)
    text_col = data_conf["text_column"]
    logger.info(f"Data size: {len(df)}")
    if not post_level_execution_config["keep_all_data"]:
        if post_level_execution_config['omit_echo']:
            df = df[df[text_col].apply(lambda t: '(((' not in t and ')))' not in t)].reset_index(drop=True)
            logger.info(f"Data size after omitting posts with echo symbol: {len(df)}")
        else:
            df = df[df[text_col].apply(lambda t: '(((' in t and ')))' in t)].reset_index(drop=True)
            logger.info(f"Data size after keeping only posts with echo symbol: {len(df)}")

    X = df[text_col]
    y = df[data_conf["label_column"]]


    labels = data_conf["labels"]
    labels_interpretation = data_conf["labels_interpretation"]
    post_level_execution_config["kwargs"]["labels"] = labels
    post_level_execution_config["kwargs"]["labels_interpretation"] = labels_interpretation

    # preprocess configs
    preprocess_output_path = post_level_execution_config["preprocessing"]["output_path"]
    max_seq_len = post_level_execution_config["kwargs"]["max_seq_len"]
    bert_conf = post_level_execution_config["preprocessing"]["bert_conf"]
    max_features = post_level_execution_config["preprocessing"]["max_features"]
    preprocess_type = post_level_execution_config['preprocessing']['type']
    test_size = post_level_execution_config['data']['test_size']

    model_type = post_level_execution_config['model']

    pt = PreprocessText(max_features=max_features, max_seq_len=max_seq_len,
                        bert_conf=bert_conf, preprocess_type=preprocess_type,
                        output_path=preprocess_output_path, test_size=test_size)
    train_on_all_data = post_level_execution_config["train_on_all_data"]
    if train_on_all_data:
        X_train, _, y_train, _, _, _ = pt.full_preprocessing(X, y, mode='train')
    else:
        X_train, X_test, y_train, y_test, X_train_as_text, X_test_as_text = pt.full_preprocessing(X, y, mode='split')

    pickle.dump(pt, open(os.path.join(preprocess_output_path, "preprocess_text.pkl"), "wb"))
    post_level_execution_config["kwargs"]["vocab_size"] = pt.vocab_size

    if train_on_all_data:
        if len(labels) > 2:
            y_train = pd.get_dummies(y_train)
        y_train = y_train.astype(int)
    else:
        original_y_test = y_test.copy()
        if len(labels) > 2:
            y_train = pd.get_dummies(y_train)
            y_test = pd.get_dummies(y_test)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    # instantiate the model
    model = factory(model_type, **post_level_execution_config["kwargs"])
    # train model
    model.fit(X_train, y_train)

    # evaluate model and save results
    if not train_on_all_data and X_test is not None:
        y_true_dummy = y_test.copy()
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)

        y_true_p = original_y_test.to_frame(name="y_true").reset_index(drop=True)
        y_score_p = pd.DataFrame(y_score, columns=[f"y_score_{x}" for x in labels] if len(labels) > 2 else ['y_score']).reset_index(drop=True)
        y_pred_p = pd.DataFrame(y_pred, columns=['y_pred']).reset_index(drop=True)
        X_test_as_text = X_test_as_text.reset_index(drop=True)
        simple_metrics = {}

        for acc_metric in [accuracy_score, balanced_accuracy_score]:
            simple_metrics[acc_metric.__name__] = acc_metric(y_true_p, y_pred_p)
        for metric in [f1_score, recall_score, precision_score]:
            if len(labels) == 2:
                simple_metrics[metric.__name__] = metric(y_true_p, y_pred_p)
            else:
                for averaging_technique in ["micro", "macro", "weighted"]:
                    simple_metrics[f"{metric.__name__}__{averaging_technique}"] = metric(y_true_p, y_pred_p, average=averaging_technique)
        # fp, tp, th = roc_curve(y_true=y_true_p, y_score=y_score_p)
        # simple_metrics[auc.__name__] = auc(fp, tp)
        predictions = pd.concat([X_test_as_text, y_true_p, y_score_p, y_pred_p], axis=1, ignore_index=True)
        if len(labels) == 2:
            predictions.columns = ["text", "y_true", "y_score", "y_pred"]
        else:
            predictions.columns = ["text", "y_true", "y_score_0", "y_score_1", "y_score_2", "y_pred"]
        evaluation_output_path = post_level_execution_config["evaluation"]["output_path"]
        create_dir_if_missing(evaluation_output_path)
        predictions.to_csv(os.path.join(evaluation_output_path, "predictions.tsv"), sep='\t', index=False)

        with open(os.path.join(evaluation_output_path, "simple_metrics.txt"), "w") as fout:
            fout.write(json.dumps(simple_metrics))

        labels_mapping = {label_id: label for label_id, label in zip(labels, labels_interpretation)}
        evaluation_kwargs = {"labels": labels_interpretation, "labels_mapping": labels_mapping,
                             "output_path": post_level_execution_config["evaluation"]["output_path"]}
        for metric in post_level_execution_config["evaluation"]["metrics"]:
            evaluator = factory(metric, None, **evaluation_kwargs)
            evaluator.calc(y_true=original_y_test, y_true_dummy=y_true_dummy, y_pred=y_pred, y_score=y_score)

if __name__ == '__main__':
    run_single_plm_experiment()
