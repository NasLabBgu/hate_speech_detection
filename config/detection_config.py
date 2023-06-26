import os

# post level execution config
post_level_execution_config = {
    "multiple_experiments": False,
    # set multiple_experiments to False when running post_level__experiment.py file.
    # set multiple_experiments to True when running the file post_level__multiple_experiments.py.
    "data": {
        "dataset": "echo_2",  # possible values: ["echo_2", "gab, "parler"]
        'test_size': 0.2
    },
    "train_on_all_data": False,
    "keep_all_data": True,
    "omit_echo": False,
    # relevant only if keep_al_data is set to false. if omit_echo is true - then we keep only posts without echo, and vice versa.
    "model": "models.BertFineTuning",  # choose the model to run
    # possible model values: ["AttentionLSTM", "CNN_LSTM", "BertFineTuning", "FeedForwardNN", "MyLogisticRegression",
    #                         "MyCatboost", "MyLightGBM", "MyXGBoost"]
    "bert_conf": {  # relevant only for "BertFineTuning" model
        # possible values:
        # 'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased'
        # 'roberta-base', 'roberta-large', 'roberta-large-mnli'
        # 'xlnet-base-cased', 'xlnet-large-cased'
        # 'distilroberta-base', 'distilbert-base-uncased', 'distilbert-base-uncased-distilled-squad', 'distilbert-base-cased', 'distilbert-base-cased-distilled-squad'

        # best bert transformer are 'bert-base-uncased', 'distilbert-base-uncased'
        'model_type': "distilbert-base-uncased",
        'use_masking': True,
        'use_token_types': False
    },
    "preprocessing": {
        "type": 'bert',  # one of 'nn', 'bert', 'tfidf'
        "output_path": "detection/outputs",
        "max_features": 10000  # applicable only for non-bert models
    },
    "kwargs": {
        "model_name": "BertFineTuning",
        "max_seq_len": 128,  # For echo_2 256
        "emb_size": 300,  # 64 Gab # 300 # Gab dataset has emb_size of 300
        "epochs": 20,
        "fine_tune": True,
        "validation_split": 0.2,
        "model_api": "functional",
        'dense_units': 16,  # 4, Gab
        'bs': 8,
        "paths": {
            "train_output": "detection/outputs/",
            "model_output": "detection/outputs/"
        },
        "weights_only": True
    },
    "evaluation": {
        "metrics": [
            "evaluation_metrics.ConfusionMatrix",
            "evaluation_metrics.ROC",
            "evaluation_metrics.PrecisionRecallCurve"
        ],
        "output_path": "detection/outputs/"
    }
}

# user level execution config
user_level_execution_config = {
    "trained_data": "echo_2",  # possible values: ["echo_2", "gab, "parler"]
    "inference_data": "echo_2",  # possible values: ["echo_2", "gab, "parler"]
}

# configs specific to posts with/wo the echo sign
echo_data_conf = {
    "with_rt": True,
    "only_en": False  # if false -> with_rt must be true
}

if echo_data_conf["only_en"] == True:
    echo_suffix = "en_"
else:
    echo_suffix = "all_lang_"
if echo_data_conf["with_rt"] == True:
    echo_suffix += "with_rt"
else:
    echo_suffix += "no_rt"

# things to add to the execution config
if post_level_execution_config["multiple_experiments"] == True:
    post_level_execution_config["kwargs"]["model_name"] = "multiple_experiments"
else:
    post_level_execution_config["kwargs"]["model_name"] = post_level_execution_config["model"].split(".")[-1]

# additional paths config (corresponding to the output directory that is set according to the model_name param)
post_level_execution_config["preprocessing"]["output_path"] = os.path.join(
    post_level_execution_config["preprocessing"]["output_path"],
    post_level_execution_config["data"]["dataset"],
    post_level_execution_config["kwargs"]["model_name"], "preprocessing")
post_level_execution_config["kwargs"]["paths"]["model_output"] = os.path.join(
    post_level_execution_config["kwargs"]["paths"]["model_output"],
    post_level_execution_config["data"]["dataset"],
    post_level_execution_config["kwargs"]["model_name"], "saved_model")
post_level_execution_config["kwargs"]["paths"]["train_output"] = os.path.join(
    post_level_execution_config["kwargs"]["paths"]["train_output"],
    post_level_execution_config["data"]["dataset"],
    post_level_execution_config["kwargs"]["model_name"], "training")
post_level_execution_config["evaluation"]["output_path"] = os.path.join(
    post_level_execution_config["evaluation"]["output_path"],
    post_level_execution_config["data"]["dataset"],
    post_level_execution_config["kwargs"]["model_name"], "evaluation")

if 'bert' in post_level_execution_config['model'].lower():
    post_level_execution_config["preprocessing"]["bert_conf"] = post_level_execution_config["bert_conf"]
    post_level_execution_config["kwargs"]["bert_conf"] = post_level_execution_config["bert_conf"]
else:
    post_level_execution_config["preprocessing"]["bert_conf"] = None
    post_level_execution_config["kwargs"]["bert_conf"] = None

# Data path configs
post_level_conf = {
    "davidson_2": {
        "data_path": "data/twitter/hate-speech-and-offensive-language/davidson_2_labels_no_offensive.tsv",
        # davidson_2_labels.tsv
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["neither", "hate-speech/offensive-language"]

    },
    "davidson_3": {
        "data_path": "data/twitter/hate-speech-and-offensive-language/davidson_3_labels.tsv",
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1, 2],
        "labels_interpretation": ["neither", "offensive-language", "hate-speech"]
    },
    "waseem_2": {
        "data_path": "data/twitter/hate_speech_naacl/mkr_posts_annotations_2_label.tsv",
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["none", "sexism/racism"]
    },
    "waseem_3": {
        "data_path": "data/twitter/hate_speech_naacl/mkr_posts_annotations_3_label.tsv",
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1, 2],
        "labels_interpretation": ["none", "sexism", "racism"]
    },
    "echo_2": {
        "data_path": "data/post_level/echo_posts_2_labels.tsv",
        # all_annotations_2_labels.tsv  echo_tweets_2_labels.tsv",
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["neutral-responsive", "hate speech"]
    },
    "echo_3": {
        "data_path": "data/post_level/echo_posts_3_labels.tsv",
        "unique_column": "tweet_id",
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1, 2],
        "labels_interpretation": ["neutral", "hate speech", "responsive"]
    },
    "gab": {
        "data_path": "data/post_level/gab_posts_2_labels.tsv",
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["Not-HS", "HS"]
    },
    "combined": {
        "data_path": "data/post_level/combined_post_data_2_labels_no_offensive_davidson.tsv",
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["Not-HS", "HS"]
    },
    "parler": {
        "data_path": "data/post_level/parler_posts_2_labels.tsv",
        # "data/post_level/parler_posts_2_labels.tsv",
        "text_column": "text",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["Not-HS", "HS"]
    },
}

user_level_conf = {
    "echo_2": {
        "data_path": "data/user_level/echo_users_2_labels.tsv",
        "following_fn": "data_users_mention_edges_df.tsv",
        "user_unique_column": "user_id",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["neutral-responsive", "hate speech"],
        "posts_per_user_path": f"hate_networks/outputs/echo_networks/pickled_data/corpora_list_per_user.pkl"
    },
    "echo_3": {
        "data_path": "data/twitter/echo/echo_users_3_labels.tsv",
        "user_unique_column": "user_id",
        "label_column": "label",
        "labels": [0, 1, 2],
        "labels_interpretation": ["neutral", "hate speech", "responsive"]
    },
    "gab": {
        "data_path": "data/user_level/gab_users_2_labels.tsv",
        "following_fn": "labeled_users_followers.tsv",
        "user_unique_column": "user_id",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["Not-HM", "HM"],
        "posts_per_user_path": "hate_networks/outputs/gab_networks/pickled_data/corpora_list_per_user.pkl"
    },
    "parler": {
        "data_path": "data/user_level/parler_users_2_labels.tsv",
        "following_fn": "labeled_users_followers.tsv",
        "user_unique_column": "user_id",
        "label_column": "label",
        "labels": [0, 1],
        "labels_interpretation": ["Not-HM", "HM"],
        "posts_per_user_path": "/sise/Yalla_work/data/parler/gnn/users_posts_list.p"
    }
}

# user_level_conf["echo_2"]["posts_per_user_path"] = f"hate_networks/echo_networks/pickled_data/corpora_list_per_user_{echo_suffix}.pkl"
