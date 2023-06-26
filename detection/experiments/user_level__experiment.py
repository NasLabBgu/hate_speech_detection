import os
import sys

f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, "../.."))
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, auc, roc_curve
from config.detection_config import post_level_conf, user_level_conf, post_level_execution_config, \
    user_level_execution_config
from detection.detection_utils.factory import create_dir_if_missing
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Concatenate, Input, BatchNormalization, Dropout, Conv2D, MaxPooling2D, \
    Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.utils.class_weight import compute_class_weight
from detection.models.attention_lstm import AttentionWithContext
from tensorflow.keras.initializers import glorot_uniform
from detection.detection_utils.factory import factory
from detection.models.base_model import f1_m
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt

plt.style.use("ggplot")

from utils.my_timeit import timeit
from utils.general import init_log

logger = init_log("user_level_experiment")


def load_post_model(**post_level_kwargs):
    """
    A function that loads the preprocess object and the PLM
    :param post_level_kwargs:
    :return:
    """
    logger.info(f'loading preprocessing object from {post_level_kwargs["preprocessing"]["output_path"]}')

    pt = pickle.load(open(os.path.join(post_level_kwargs["preprocessing"]["output_path"], 'preprocess_text.pkl'), "rb"))

    post_level_kwargs["kwargs"]["vocab_size"] = pt.vocab_size

    model = factory(post_level_kwargs["model"], **post_level_kwargs["kwargs"])

    if post_level_kwargs['kwargs']['weights_only']:
        # model = factory(post_level_kwargs["model"], **post_level_kwargs["kwargs"])
        # logger.info(f'loading post model from {os.path.join(model.kwargs["paths"]["model_output"])}')

        logger.info(f'loading post model from {os.path.join(model.kwargs["paths"]["model_output"])}')
        model_weights_file_path = os.path.join(model.kwargs["paths"]["model_output"], "weights_best.h5")
        model.model.load_weights(model_weights_file_path)

    else:
        model_weights_file_path = os.path.join(post_level_kwargs["kwargs"]["paths"]["model_output"], "model_best.h5")

        model.model = load_model(model_weights_file_path,
                                 custom_objects={'f1_m': f1_m, 'AttentionWithContext': AttentionWithContext})
    # model = load_model(os.path.join(model_path, "saved_model/model.h5"),
    #                custom_objects={'AttentionWithContext': AttentionWithContext, 'GlorotUniform': glorot_uniform()})
    print(model.model.summary())
    return pt, model


def predict_df(trained_dataset_name, inference_dataset_name, df):
    logger.info(
        f"predicting all users for dataset: {inference_dataset_name} using models train on {trained_dataset_name} data")
    logger.info(f"predicting all users for dataset: {inference_dataset_name}")
    post_level_data_conf = post_level_conf[trained_dataset_name]
    labels = post_level_data_conf["labels"]
    labels_interpretation = post_level_data_conf["labels_interpretation"]
    post_level_execution_config["kwargs"]["labels"] = labels
    post_level_execution_config["kwargs"]["labels_interpretation"] = labels_interpretation

    pt, post_model = load_post_model(**post_level_execution_config)

    _, X_test, _, _, _, _ = pt.full_preprocessing(df['text'], None, mode='test')
    # logger.info(f"predicting users tweets; indices: {user_range} to {user_range + chunk_size - 1}...")
    y_proba = post_model.predict_proba(X_test)
    # y_proba = post_model.predict(X_test)

    return y_proba


def predict_all_users(trained_dataset_name, inference_dataset_name, user_level_path):
    logger.info(
        f"predicting all users for dataset: {inference_dataset_name} using models train on {trained_dataset_name} data")
    logger.info(f"predicting all users for dataset: {inference_dataset_name}")
    post_level_data_conf = post_level_conf[trained_dataset_name]
    labels = post_level_data_conf["labels"]
    labels_interpretation = post_level_data_conf["labels_interpretation"]
    post_level_execution_config["kwargs"]["labels"] = labels
    post_level_execution_config["kwargs"]["labels_interpretation"] = labels_interpretation

    # load preprocessor and model
    pt, post_model = load_post_model(**post_level_execution_config)
    # get posts per user
    # if os.path.exists(os.path.join(user_level_path, "X_test.pkl")):
    #     X_test = pickle.load(open(os.path.join(user_level_path, "X_test.pkl"), "rb"))
    # else:
    if not os.path.exists(os.path.join(user_level_path, "ego_network_users_tweets.parquet")):
        logger.info(f"reading all posts of {inference_dataset_name} dataset...")
        posts_per_user_dict = pickle.load(
            open(user_level_conf[inference_dataset_name]["posts_per_user_path"], "rb"))
        for k, v in posts_per_user_dict.items():
            posts_per_user_dict[k] = [p.strip() for p in v if p.strip() != '']  # omit empty posts
        logger.info(f"finished reading all posts of {inference_dataset_name} dataset")

        # i = 0
        if len(posts_per_user_dict[list(posts_per_user_dict.keys())[0]][0]) == 2:
            full_df = pd.DataFrame(columns=['user_id', 'post_id', 'text'])
            for user_id, user_posts in tqdm(posts_per_user_dict.items()):
                user_id = str(user_id)
                current_user_df = pd.DataFrame({'user_id': [user_id for _ in range(len(user_posts))],
                                                'post_id': [user_post_tup[0] for user_post_tup in user_posts],
                                                'text': [user_post_tup[1].strip() for user_post_tup in user_posts if
                                                         user_post_tup[1].strip() != ''],
                                                })
                full_df = full_df.append(current_user_df, ignore_index=True)
        else:
            full_df = pd.DataFrame(columns=['user_id', 'text'])
            for user_id, user_posts in tqdm(posts_per_user_dict.items()):
                user_id = str(user_id)
                user_posts = [p.strip() for p in user_posts if p.strip() != '']
                current_user_df = pd.DataFrame({'user_id': [user_id for _ in range(len(user_posts))],
                                                'text': user_posts,
                                                })
                full_df = full_df.append(current_user_df, ignore_index=True)
        full_df.to_parquet(os.path.join(user_level_path, "all_users_tweets.parquet"), index=False)
    else:  # tweets per user df already exists
        logger.info("reading all_users_tweets.parquet file...")

        full_df = pd.read_parquet(os.path.join(user_level_path, "ego_network_users_tweets.parquet"))
        logger.info(f"full_df shape: {full_df.shape}")
        full_df = full_df[full_df["text"].apply(lambda t: t.strip() != "")].reset_index(drop=True)
        # first_10000_users = list(full_df["user_id"].unique())[:10000]
        # full_df = full_df[full_df["user_id"].isin(first_10000_users)]
        logger.info(f"full_df shape after removing empty posts: {full_df.shape}")
        logger.info("file read.")

    # SPLITTING TO CHUNKS BY TWEETS
    chunk_size = np.minimum(10000000, len(full_df))
    logger.info(f"preprocessing in chunks of {chunk_size}...")
    logger.info(f"Length of full_df: {len(full_df)}")
    for user_range in tqdm(range(0, len(full_df), chunk_size)):
        current_full_df = full_df.loc[user_range:user_range + chunk_size - 1]
        current_X = current_full_df["text"]

        _, X_test, _, _, _, _ = pt.full_preprocessing(current_X, None, mode='test')
        logger.info(f"predicting users tweets; indices: {user_range} to {user_range + chunk_size - 1}...")
        y_proba = post_model.predict_proba(X_test)
        # y_proba = post_model.predict(X_test)
        current_full_df.loc[:, 'predictions'] = y_proba

        create_dir_if_missing(os.path.join(user_level_path, "split_by_posts"))
        create_dir_if_missing(os.path.join(user_level_path, "split_by_posts", "no_text"))
        create_dir_if_missing(os.path.join(user_level_path, "split_by_posts", "with_text"))
        logger.info(f"saving predictions to {user_level_path}")

        current_full_df[['user_id', 'predictions']].to_parquet(
            os.path.join(user_level_path, "split_by_posts", "no_text",
                         f"user2pred_min_idx_{user_range}_max_idx_{user_range + chunk_size - 1}.parquet"),
            index=False)

        current_full_df.to_parquet(os.path.join(user_level_path, "split_by_posts", "with_text",
                                                f"user2pred_with_text_min_idx_{user_range}_max_idx_{user_range + chunk_size - 1}.parquet"),
                                   index=False)


def build_user_model(max_user_tweets: int, max_followings_num: int, max_followers_num: int, network_feautres_num: int,
                     relevant_inputs) -> Model:
    """
    Function that builds the NN for the user-level model
    :param max_user_tweets:
    :param max_followings_num:
    :param max_followers_num:
    :param network_feautres_num:
    :param relevant_inputs:
    :return:
    """
    self_input = Input(shape=(max_user_tweets,), name="self_input", dtype="float32")
    followings_input = Input(shape=(max_followings_num,), name="followings_input", dtype="float32")
    followers_input = Input(shape=(max_followers_num,), name="followers_input", dtype="float32")
    network_features_input = Input(shape=(network_feautres_num,), name="network_features_input", dtype="float32")
    input_mapping = {0: self_input, 1: followings_input, 2: followers_input, 3: network_features_input}
    relevant_inputs = [input for idx, input in input_mapping.items() if idx in relevant_inputs]
    if len(relevant_inputs) > 1:
        all_inputs = Concatenate(axis=-1)(relevant_inputs)
    else:
        all_inputs = relevant_inputs[0]
    x = Dense(512, activation='relu')(all_inputs)
    x = Dropout(0.2)(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=relevant_inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m])

    return model


def prepare_data_for_modeling(X, y, relevant_features_idx, normalize_features, test_size, output_path,
                              only_inference=False):
    for i in range(len(X)):
        X[i][np.isnan(X[i])] = 0.0

    if test_size is not None:
        _, _, y_train, y_test = train_test_split(pd.DataFrame(X[0]), pd.Series(y), test_size=test_size, random_state=0)
        train_idx = y_train.index
        test_idx = y_test.index

        X_train = []
        X_test = []

        for i, feature_idx in enumerate(relevant_features_idx):
            X_train.append(np.array(pd.DataFrame(X[feature_idx]).loc[train_idx]))
            X_test.append(np.array(pd.DataFrame(X[feature_idx]).loc[test_idx]))
            if normalize_features:
                # if feature_idx == 3:
                scaler = StandardScaler()
                # scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                # scaler = MinMaxScaler(feature_range=(0,1))
                X_train[i] = scaler.fit_transform(X_train[i])
                X_test[i] = scaler.transform(X_test[i])
    else:
        X_train = []
        X_test = None
        if y is not None:
            y_train = y.copy()
        else:
            y_train = None
        y_test = None
        for i, feature_idx in enumerate(relevant_features_idx):
            X_train.append(np.array(pd.DataFrame(X[feature_idx])))
            if normalize_features:
                if only_inference:
                    scaler = pickle.load(open(os.path.join(output_path, "scaler.pkl"), "rb"))
                    X_train[i] = scaler.transform(X_train[i])
                else:
                    scaler = PowerTransformer(method="yeo-johnson", standardize=True)
                    X_train[i] = scaler.fit_transform(X_train[i])
                    pickle.dump(scaler, open(os.path.join(output_path, "scaler.pkl"), "wb"))
    return X_train, X_test, y_train, y_test


def run_user_model(X, y, features_to_use, output_path, model_type="nn", normalize_features=True, test_size=0.2):
    input_features_mapping = {"self": 0, "followings": 1, "followers": 2, "network": 3}
    relevant_features_idx = [v for k, v in input_features_mapping.items() if k in features_to_use]
    res_row = {f: False for f in input_features_mapping.keys()}
    for f in features_to_use:
        res_row[f] = True
    res_row["model"] = model_type
    if test_size is None:
        ## train with all data for best performing configuration.
        X_train, _, y_train, _ = prepare_data_for_modeling(X, y, relevant_features_idx, normalize_features, test_size,
                                                           output_path)
    else:
        X_train, X_test, y_train, y_test = prepare_data_for_modeling(X, y, relevant_features_idx, normalize_features,
                                                                     test_size, output_path)
    if model_type == "nn":
        user_model = build_user_model(X[0].shape[1], X[1].shape[1], X[2].shape[1], X[3].shape[1], relevant_features_idx)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto',
                                       restore_best_weights=True)
        class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
        hist = user_model.fit(x=X_train, y=y_train, batch_size=128, epochs=60, validation_split=0.2, verbose=0,
                              callbacks=[], class_weight={i: class_weight[i] for i in range(2)})
        if test_size is None:
            user_model.save(os.path.join(output_path, "best_user_model.model"), save_format='tf')
    else:
        X_train = np.hstack(tuple([input for input in X_train]))
        if model_type == "catboost":  # requires concatenation in advance
            user_model = CatBoostClassifier()
            user_model.fit(X_train, y_train, verbose=False)
        elif model_type == "lightgbm":
            user_model = lgb.LGBMClassifier()
            user_model.fit(X_train, y_train, verbose=False)
        elif model_type == "lr":
            user_model = LogisticRegressionCV(cv=5, max_iter=10000)
            user_model.fit(X_train, y_train)
        elif model_type == "xgboost":
            user_model = XGBClassifier()
            user_model.fit(X_train, y_train, verbose=False)
        if test_size is None:
            pickle.dump(user_model, open(os.path.join(output_path, "best_user_model.model"), "wb"))
    # evaluation phase
    if test_size is not None:
        if model_type == "nn":
            y_score = user_model.predict(X_test)
            fp, tp, th = roc_curve(y_true=y_test, y_score=y_score[:, -1])
            res_row[auc.__name__] = auc(fp, tp)

            y_pred = (y_score > 0.5).astype('int32')
            for metric in [f1_score, accuracy_score, recall_score, precision_score]:
                # logger.info(f"{metric.__name__}: {metric(y_test, y_pred):.2f}")
                res_row[metric.__name__] = metric(y_test, y_pred)

            # test_loss, test_acc, test_f1, test_precision, test_recall = user_model.evaluate(X_test, y_test)
            test_loss, test_f1 = user_model.evaluate(X_test, y_test)
            logger.info(f"test loss: {test_loss}")
            # logger.info(f"test acc: {test_acc}")
            logger.info(f"test f1: {test_f1}")
            # logger.info(f"test precision: {test_precision}")
            # logger.info(f"test recall: {test_recall}")
            ## save the model
            # user_model.save_weights(model_weights_file_path)  # save model's weights
            # self.model.save(model_file_path, save_format='tf')  # save full model
            # pickle.dump(self, open(full_model_file_path, "wb"))

            ## save plots of loss and accuracy during training
            training_output_path = os.path.join(output_path, "training")
            create_dir_if_missing(training_output_path)
            plt.figure()
            plt.title('Loss')
            plt.plot(hist.history['loss'], label='train')
            plt.plot(hist.history['val_loss'], label='validation')
            plt.legend()
            loss_fn = os.path.join(training_output_path, "loss_graph.png")
            plt.savefig(loss_fn)
            # plt.figure()
            # plt.title('Accuracy')
            # plt.plot(hist.history['accuracy'], label='train')
            # plt.plot(hist.history['val_accuracy'], label='validation')
            # plt.legend()
            # acc_fn = os.path.join(training_output_path, "acc_graph.png")
            # plt.savefig(acc_fn)
        else:
            X_test = np.hstack(tuple([input for input in X_test]))
            y_score = user_model.predict_proba(X_test)[:, -1]
            fp, tp, th = roc_curve(y_true=y_test, y_score=y_score)
            res_row[auc.__name__] = auc(fp, tp)
            y_pred = user_model.predict(X_test)
            for metric in [f1_score, accuracy_score, recall_score, precision_score]:
                # logger.info(f"{metric.__name__}: {metric(y_test, y_pred):.2f}")
                res_row[metric.__name__] = metric(y_test, y_pred)
    return res_row


def predict_all_users_labels(X, features_to_use, model_type, model_path, normalize_features, output_path):
    if model_type == "nn":
        user_model = load_model(model_path, custom_objects={'f1': f1_m})
    else:
        user_model = pickle.load(open(model_path, "rb"))
    input_features_mapping = {"self": 0, "followings": 1, "followers": 2, "network": 3}
    relevant_features_idx = [v for k, v in input_features_mapping.items() if k in features_to_use]
    X = prepare_data_for_modeling(X, None, relevant_features_idx, normalize_features, None, output_path,
                                  only_inference=True)
    y_pred = user_model.predict(X)
    user_df = pd.read_csv(os.path.join(output_path, "io", "all_users_df.tsv"), sep='\t')
    user_df.to_csv(os.path.join(output_path, "all_pred.tsv"), sep='\t', index=False)

    user2pred = pd.concat([user_df, y_pred], axis=1)
    user2pred.to_csv(os.path.join(output_path, "all_users2pred.tsv"), sep='\t', index=False)


def get_followers_followees_dicts(network_dir, following_fn, min_threshold=3):
    edges_dir = os.path.join(network_dir, "edges")
    mentions_df = pd.read_csv(os.path.join(edges_dir, following_fn), sep='\t')  # "data_users_mention_edges_df.tsv"
    for col in ['source', 'dest']:
        mentions_df[col] = mentions_df[col].astype(str)
    # keep only mentions above the minimal threshold
    if 'mention' in following_fn:
        mentions_df = mentions_df[mentions_df["weight"] >= min_threshold].reset_index(drop=True)
    mentions_dict = {}  # users mentioned by the observed user
    mentioned_by_dict = {}  # users mentioning the observed user
    for idx, row in mentions_df.iterrows():
        src = row['source']
        dest = row['dest']
        if src not in mentions_dict.keys():
            mentions_dict[src] = []
        if dest not in mentioned_by_dict.keys():
            mentioned_by_dict[dest] = []
        mentions_dict[src].append(dest)
        mentioned_by_dict[dest].append(src)
    return mentions_dict, mentioned_by_dict


def prepare_inputs_outputs(dataset_name, all_posts_probs_df_path, output_path, only_inference=False):
    """
    predict whether a user is a hatemonger or not by his tweets and his friends' (mentioned by the user and mentioning
     the user) tweets.
    :return:
    """
    logger.info(f"Preparing inputs outputs for {dataset_name} dataset...")
    # user level config
    data_conf = user_level_conf[dataset_name]
    if "data_path" in data_conf.keys():
        data_path = data_conf["data_path"]
        user_column = data_conf["user_unique_column"]
        label_column = data_conf["label_column"]
        labels = data_conf["labels"]
        labels_interpretation = data_conf["labels_interpretation"]
        if only_inference:  # get all users in the network for inference
            user_df = pd.read_csv(f"hate_networks/{dataset_name.split('_')[0]}_networks/tsv_data/users.tsv", sep="\t")
            if dataset_name == 'gab':
                user_df = user_df.sample(n=10000)  # too many users in gab net... sample 10K
            user_df[user_column] = user_df[user_column].astype(str)
        else:
            file_ending = data_path.split(".")[-1]
            if file_ending == 'csv':
                sep = ','
            elif file_ending == 'tsv':
                sep = '\t'
            else:
                raise ValueError(f"wrong ending for file {data_path}")
            user_df = pd.read_csv(data_path, sep=sep)
            user_df[user_column] = user_df[user_column].astype(str)

    network_dir = f"hate_networks/outputs/{dataset_name.split('_')[0]}_networks/network_data/"
    # load centralities features for each user
    centralities_df = pd.read_csv(
        os.path.join(network_dir, "features", "centralities_mention_edges_filtered_singletons_filtered.tsv"),
        sep='\t')  # centralities_mention_all_edges_all_nodes.tsv
    centralities_df[user_column] = centralities_df[user_column].astype(str)
    centrality_measurements = [col for col in centralities_df.columns if col != 'user_id']
    min_mention_threshold = 3
    following_fn = data_conf["following_fn"]
    mentions_dict, mentioned_by_dict = get_followers_followees_dicts(network_dir, following_fn, min_mention_threshold)

    # take mean of lengths as maximum length of feature array
    # consider average of lengths only from labeled users!
    labeled_mentions_dict = {key: val for key, val in mentions_dict.items() if key in user_df.user_id.tolist()}
    labeled_mentioned_by_dict = {key: val for key, val in mentioned_by_dict.items() if key in user_df.user_id.tolist()}

    max_followings_num = int(np.mean([len(x) for x in labeled_mentions_dict.values()]))
    max_followers_num = int(np.mean([len(x) for x in labeled_mentioned_by_dict.values()]))
    if all_posts_probs_df_path.endswith("parquet") or os.path.isdir(all_posts_probs_df_path):
        all_users_probs = pd.read_parquet(all_posts_probs_df_path)
    else:
        all_users_probs = pd.read_csv(all_posts_probs_df_path, sep='\t', engine='python')
    labeled_users_predictions = all_users_probs[all_users_probs['user_id'].isin(user_df.user_id.tolist())]
    max_user_tweets = int(labeled_users_predictions.groupby('user_id').size().reset_index()[0].mean())
    self_input = []
    followings_input = []
    followers_input = []
    network_features_input = []
    outputs = []
    all_users_list = list(all_users_probs["user_id"].unique())

    for idx, row in tqdm(user_df.iterrows()):
        user_id = str(row[user_column])

        if user_id not in centralities_df["user_id"].tolist():
            current_network_features = np.zeros(len(centrality_measurements))
        else:
            current_network_features = []
            for cm in centrality_measurements:
                current_network_features.append(centralities_df.loc[centralities_df["user_id"] == user_id, cm].iloc[0])
        network_features_input.append(np.array(current_network_features))
        if not only_inference:
            outputs.append(row[label_column])
        avg_followings_predictions = []
        avg_followers_predictions = []
        if user_id in mentions_dict.keys():
            followings = mentions_dict[user_id]  # users mentioned by the observed user
            followings = [followee for followee in followings if followee in all_users_list]
            if len(followings) > 0:
                followings_predictions = all_users_probs.loc[all_users_probs["user_id"].isin(followings)]
                avg_followings_predictions = followings_predictions.groupby('user_id').agg({'predictions': 'mean'})[
                    'predictions']

        if user_id in mentioned_by_dict.keys():
            followers = mentioned_by_dict[user_id]  # users mentioning the observed user
            followers = [follower for follower in followers if follower in all_users_list]
            if len(followers) > 0:
                followers_predictions = all_users_probs.loc[all_users_probs["user_id"].isin(followers)]
                avg_followers_predictions = followers_predictions.groupby('user_id').agg({'predictions': 'mean'})[
                    'predictions']

        self_predictions = all_users_probs.loc[all_users_probs["user_id"] == user_id, "predictions"]

        # handle followings/followers predictions (average them)
        self_predictions = list(self_predictions)
        avg_followings_predictions = list(avg_followings_predictions)
        avg_followers_predictions = list(avg_followers_predictions)

        # enough when max is really max (not average)
        if len(self_predictions) < max_user_tweets:
            self_predictions.extend([0.0] * (max_user_tweets - len(self_predictions)))
        if len(avg_followings_predictions) < max_followings_num:
            avg_followings_predictions.extend([0.0] * (max_followings_num - len(avg_followings_predictions)))
        if len(avg_followers_predictions) < max_followers_num:
            avg_followers_predictions.extend([0.0] * (max_followers_num - len(avg_followers_predictions)))
        # when max is an average, we need to also remove some of the predictions for users with more than the avg
        if len(self_predictions) > max_user_tweets:
            self_predictions = self_predictions[:max_user_tweets]
        if len(avg_followings_predictions) > max_followings_num:
            avg_followings_predictions = avg_followings_predictions[:max_followings_num]
        if len(avg_followers_predictions) > max_followers_num:
            avg_followers_predictions = avg_followers_predictions[:max_followers_num]

        self_input.append(np.array(self_predictions))
        followings_input.append(np.array(avg_followings_predictions))
        followers_input.append(np.array(avg_followers_predictions))

    self_input = np.array(self_input, dtype='float32')
    followings_input = np.array(followings_input, dtype='float32')
    followers_input = np.array(followers_input, dtype='float32')
    network_features_input = np.array(network_features_input, dtype='float32')
    # user_model = build_user_model(max_user_tweets, max_followings_num, max_followers_num, len(centrality_measurements))
    all_inputs = [self_input, followings_input, followers_input, network_features_input]

    io_path = os.path.join(output_path, "io")
    create_dir_if_missing(io_path)
    if not only_inference:
        outputs = np.array(outputs)
        pickle.dump(all_inputs, open(os.path.join(io_path, "inputs.pkl"), "wb"))
        pickle.dump(outputs, open(os.path.join(io_path, "outputs.pkl"), "wb"))
        user_df.to_csv(os.path.join(io_path, "labeled_users_df.tsv"), sep='\t', index=False)
    else:
        pickle.dump(all_inputs, open(os.path.join(io_path, "only_inputs.pkl"), "wb"))
        user_df.to_csv(os.path.join(io_path, "all_users_df.tsv"), sep='\t', index=False)
    return all_inputs, outputs


@timeit
def run_ulm_experiment():
    """
    Main function to run the ULM
    :return:
    """
    trained_data = user_level_execution_config["trained_data"]
    inference_data = user_level_execution_config["inference_data"]
    model = post_level_execution_config["kwargs"]["model_name"]
    model_path = f"detection/outputs/{trained_data}/{model}/"
    user_level_path = os.path.join(model_path, "user_level")
    create_dir_if_missing(user_level_path)
    if trained_data != inference_data:
        inference_path = os.path.join(user_level_path, inference_data)
    else:
        inference_path = user_level_path
    create_dir_if_missing(inference_path)
    if 'split_by_posts' in os.listdir(inference_path):
        all_posts_probs_df_path = os.path.join(inference_path, "split_by_posts", "no_text")
    else:
        all_posts_probs_df_path = os.path.join(inference_path, "user2pred.parquet")

    # if not os.path.exists(
    #         all_posts_probs_df_path):  # for the first time running this data - predict all posts for all users.
    predict_all_users(trained_data, inference_data, inference_path)

    # posts = [
    #     '"Muslim terrorists did this and obama brought them here to run our government. Gotta get rid of Obammy, Omar and all these other fucking Muslims"',
    #     '"is there a more debased creature in our modern pop-cultural cesspool than (((Amy Schumer)))? Probably not."',
    #     '"We must secure the existence of our people and a future for white children!"',
    #     '"If white people are going to continue  existing - we must support our race"',
    #     '"When the youngest white generation becomes adult they will be 5\% of the world population, down from 1/3 in 1900"',
    #     '"Did you know that most dog breeds were bred by whites? We’re awesome and they’re awesome!"',
    #     '"White lives matter!"',
    #     '"Blue lives matter!"'
    # ]
    # df = pd.DataFrame(posts, columns=['text'])
    #
    # y_proba = predict_df(trained_data, inference_data, df)
    # print(y_proba)
    # return

    io_path = os.path.join(inference_path, "io")
    create_dir_if_missing(io_path)

    data_conf = user_level_conf[inference_data]
    data_path = data_conf["data_path"]
    user_column = data_conf["user_unique_column"]
    file_ending = data_path.split(".")[-1]
    if file_ending == 'csv':
        sep = ','
    elif file_ending == 'tsv':
        sep = '\t'
    else:
        raise ValueError(f"wrong ending for file {data_path}")
    user_df = pd.read_csv(data_path, sep=sep)
    user_df[user_column] = user_df[user_column].astype(str)

    user_df.to_csv(os.path.join(io_path, "labeled_users_df.tsv"), sep='\t', index=False)
    if os.path.exists(os.path.join(io_path, "inputs.pkl")):
        logger.info(f"Inputs-outputs already exists for dataset {inference_data}. Loading them from {io_path}...")
        inputs = pickle.load(open(os.path.join(io_path, "inputs.pkl"), "rb"))
        output = pickle.load(open(os.path.join(io_path, "outputs.pkl"), "rb"))
    else:
        inputs, output = prepare_inputs_outputs(inference_data, all_posts_probs_df_path, output_path=user_level_path,
                                                only_inference=False)

    features_to_use = ["self", "followings", "followers", "network"]  # "self", "followings", "followers", "network"
    test_size = 0.2
    normalize_features = True
    PREDICT_ALL_DATA_USERS = False
    if not PREDICT_ALL_DATA_USERS:
        result = pd.DataFrame(columns=features_to_use + ["model"] + [metric.__name__ for metric in
                                                                     [f1_score, accuracy_score, recall_score,
                                                                      precision_score, auc]])
        # logger.info(run_user_model(inputs, output, features_to_use=["self", "followings"], output_path=user_level_path,
        #                model_type="nn", normalize_features=True))

        for model_type in ["nn"]:  # ,"lr", "catboost", "lightgbm", "xgboost",
            logger.info(f"Executing {model_type} model...")
            logger.info(model_type)
            for r in range(1, len(features_to_use) + 1):
                for fc in combinations(features_to_use, r):
                    res_row = run_user_model(inputs, output, features_to_use=fc, output_path=user_level_path,
                                             model_type=model_type, normalize_features=normalize_features,
                                             test_size=test_size)
                    result = result.append(res_row, ignore_index=True)

        result.to_csv(
            os.path.join(user_level_path, f"user_level_results__{int(test_size * 100)}_test.tsv"),
            sep='\t', index=False)
    else:
        if not os.path.exists(
                os.path.join(user_level_path, "best_user_model.model")):  # best model doesn't exist - search for it

            result = pd.DataFrame(columns=features_to_use + ["model"] + [metric.__name__ for metric in
                                                                         [f1_score, accuracy_score, recall_score,
                                                                          precision_score, auc]])
            # logger.info(run_user_model(inputs, output, features_to_use=["self", "followings"], output_path=user_level_path,
            #                model_type="nn", normalize_features=True))

            for model_type in ["lr", "catboost", "lightgbm", "xgboost", "nn"]:  # ,
                logger.info(f"Executing {model_type} model...")
                logger.info(model_type)
                for r in range(1, len(features_to_use) + 1):
                    for fc in combinations(features_to_use, r):
                        res_row = run_user_model(inputs, output, features_to_use=fc, output_path=user_level_path,
                                                 model_type=model_type, normalize_features=normalize_features,
                                                 test_size=test_size)
                        result = result.append(res_row, ignore_index=True)

            result.to_csv(
                os.path.join(user_level_path, f"user_level_results__{int(test_size * 100)}_test_power_transformed.tsv"),
                sep='\t', index=False)
            sorted_f1_results = result.sort_values('f1_score', ascending=False).reset_index(drop=True)
            best_row = sorted_f1_results.iloc[0]
            best_model = best_row["model"]
            best_features = features_to_use.copy()
            if not best_row["self"]:
                best_features.remove('self')
            if not best_row["followings"]:
                best_features.remove('followings')
            if not best_row["followers"]:
                best_features.remove('followers')
            if not best_row["network"]:
                best_features.remove('network')
            # fit best performing model on all data and save it.
            run_user_model(inputs, output, features_to_use=best_features, output_path=user_level_path,
                           model_type=best_model, normalize_features=normalize_features, test_size=None)
        else:
            models_results = pd.read_csv(
                os.path.join(user_level_path, f"user_level_results__{int(test_size * 100)}_test_power_transformed.tsv"),
                sep='\t')
            sorted_f1_results = models_results.sort_values('f1_score', ascending=False).reset_index(drop=True)
            best_row = sorted_f1_results.iloc[0]
            best_model = best_row["model"]
            best_features = features_to_use.copy()
            if not best_row["self"]:
                best_features.remove('self')
            if not best_row["followings"]:
                best_features.remove('followings')
            if not best_row["followers"]:
                best_features.remove('followers')
            if not best_row["network"]:
                best_features.remove('network')
        if os.path.exists(os.path.join(io_path, "only_inputs.pkl")):
            logger.info(f"only_inputs already exists for dataset {inference_data}. Loading them from {io_path}...")
            only_inputs = pickle.load(open(os.path.join(io_path, "only_inputs.pkl"), "rb"))
        else:
            only_inputs, _ = prepare_inputs_outputs(inference_data, all_posts_probs_df_path,
                                                    output_path=user_level_path, only_inference=True)

        predict_all_users_labels(only_inputs, best_features, model_type=best_model,
                                 model_path=os.path.join(user_level_path, "best_user_model.model"),
                                 normalize_features=normalize_features, output_path=user_level_path)


if __name__ == '__main__':
    """
    load data of tweets per user
    classify user's tweets. if 𝛿 ≥ 5 hate tweets => user id hatemonger.

    lets define a neighbor as one of the following: 

    - users that the observed user is retweeting.
    - users that the observed user is replying to (might be too sparse).
    - users that the observed user is mentioning.
    - users that mention the observed user.

    we will then consider tweets of neighbors in addition to self tweets of the observed user. A self tweet will have a weight of 1, and neighbor's tweets will have a lesser weight, e.g., 0.5.
    HM = 2 (obligatory) self hate tweets + 3 hate tweets of neighbors.
    """
    run_ulm_experiment()
