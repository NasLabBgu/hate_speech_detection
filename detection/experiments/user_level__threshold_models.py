import pandas as pd
import numpy as np
import os, sys
import warnings
import random
import json

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

f = os.path.dirname(__file__)
sys.path.append(os.path.join(os.getcwd(), "../.."))
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, auc, roc_curve, \
    balanced_accuracy_score, precision_recall_curve, confusion_matrix, roc_auc_score, RocCurveDisplay, \
    PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.utils import compute_class_weight
from sklearn.linear_model import LogisticRegression
from detection.detection_utils.factory import create_dir_if_missing
from config.detection_config import user_level_execution_config, user_level_conf, post_level_execution_config

sns.set(rc={'figure.figsize': (10, 10)}, font_scale=1.4)
from scipy.optimize import minimize
from utils.my_timeit import timeit
from utils.general import init_log

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.TPESampler(seed=0)

logger = init_log("user_level_simple_models")


def expect_f1(y_true, y_prob, thres):
    idxs = np.where(y_prob >= thres)[0]
    tp = y_prob[idxs].sum()
    fp = len(idxs) - tp
    idxs = np.where(y_prob < thres)[0]
    fn = y_prob[idxs].sum()
    return 2 * tp / (2 * tp + fp + fn)


def optimal_threshold(y_true, y_prob):
    y_prob = np.sort(y_prob)[::-1]
    f1s = [expect_f1(y_true, y_prob, p) for p in y_prob]
    thres = y_prob[np.argmax(f1s)]
    return thres  # , f1s


def get_hs_count(current_preds, threshold=0.5):
    return len(current_preds[current_preds > threshold])


def calc_metrics(y_true, y_pred):
    return {f.__name__: f(y_true, y_pred) for f in
            [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]}


f1_scores_lst = ['f1',
                 'f1_macro',
                 'f1_micro',
                 'f1_samples',
                 'f1_weighted']


def fixed_threshold_method(X: pd.DataFrame, y: pd.Series, post_threshold=0.5, test_ratio=0.2, random_state=None,
                           min_post_th=1, max_post_th=300, cv=None):
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=random_state, stratify=y)
    X_train = X[X['user_id'].isin(y_train.index)]
    X_test = X[X['user_id'].isin(y_test.index)]

    X_train.sort_index(inplace=True)
    y_train.sort_index(inplace=True)
    X_test.sort_index(inplace=True)
    y_test.sort_index(inplace=True)

    args = [post_threshold]
    train_hs_count_df = X_train.groupby('user_id').predictions.agg(get_hs_count, *args)
    min_num_of_posts_thresholds = range(max(min_post_th, train_hs_count_df.min()),
                                        min(max_post_th, train_hs_count_df.max()) + 1)

    train_preds = np.expand_dims(train_hs_count_df, axis=1) >= min_num_of_posts_thresholds
    train_f1_scores = [f1_score(y_train, p) for p in train_preds.T]
    best_f1_train, best_th = np.max(train_f1_scores), min_num_of_posts_thresholds[np.argmax(train_f1_scores)]

    test_hs_count_df = X_test.groupby('user_id').predictions.agg(get_hs_count, *args)
    test_preds = test_hs_count_df >= best_th
    train_preds = train_hs_count_df >= best_th

    metrics_dict = {'train': calc_metrics(y_train, train_preds), 'test': calc_metrics(y_test, test_preds)}

    return {'threshold': best_th}, metrics_dict


def fixed_threshold_method_cv(X: pd.DataFrame, y: pd.Series, post_threshold=0.5, test_ratio=0.2, random_state=None,
                              min_post_th=1, max_post_th=300, cv=None):
    if cv:
        args = [post_threshold]
        hs_count_df = X.groupby('user_id').predictions.agg(get_hs_count, *args)
        min_num_of_posts_thresholds = range(max(min_post_th, hs_count_df.min()),
                                            min(max_post_th, hs_count_df.max()) + 1)
        skf = StratifiedKFold(cv, random_state=random_state, shuffle=True)
        train_results_cv = []
        test_results_cv = []

        for train_idx, test_idx in skf.split(y, y):
            y_train = y[train_idx]
            y_test = y[test_idx]
            X_train = X[X['user_id'].isin(y_train.index)]
            X_test = X[X['user_id'].isin(y_test.index)]
            # train_cv_list.append([X_train, y_train])
            # test_cv_list.append([X_test, y_test])
            train_hs_count_df = X_train.groupby('user_id').predictions.agg(get_hs_count, *args)
            train_preds = np.expand_dims(train_hs_count_df, axis=1) >= min_num_of_posts_thresholds
            train_f1_scores = [f1_score(y_train, p) for p in train_preds.T]
            train_res = np.column_stack((min_num_of_posts_thresholds, train_f1_scores))
            train_results_cv.append(train_res)
            test_hs_count_df = X_test.groupby('user_id').predictions.agg(get_hs_count, *args)
            test_preds = np.expand_dims(test_hs_count_df, axis=1) >= min_num_of_posts_thresholds
            test_f1_scores = [f1_score(y_test, p) for p in test_preds.T]
            test_res = np.column_stack((min_num_of_posts_thresholds, test_f1_scores))
            test_results_cv.append(test_res)

        train_results_cv = np.array(train_results_cv)
        test_results_cv = np.array(test_results_cv)

        mean_train_res = train_results_cv.mean(axis=0)
        best_th_idx = mean_train_res.argmax(axis=1)
        best_th = int(mean_train_res[best_th_idx][0][0])
        train_mean_f1_score = mean_train_res[best_th_idx][0][1]

        mean_test_res = test_results_cv.mean(axis=0)
        test_mean_f1_score = mean_test_res[best_th_idx][0][1]

        return best_th, train_mean_f1_score, test_mean_f1_score

    else:
        return fixed_threshold_method(X, y, post_threshold, test_ratio, random_state, min_post_th, max_post_th)


def fixed_threshold_method_no_cv(X: pd.DataFrame, y: pd.Series, post_threshold=0.5, test_ratio=0.2, random_state=None,
                                 min_post_th=1, max_post_th=300):
    y_train, y_test = train_test_split(y, test_size=0.2, random_state=random_state, stratify=y)
    X_train = X[X['user_id'].isin(y_train.index)]
    X_test = X[X['user_id'].isin(y_test.index)]

    args = [post_threshold]
    train_hs_count_df = X_train.groupby('user_id').predictions.agg(get_hs_count, *args)
    min_num_of_posts_thresholds = range(max(min_post_th, train_hs_count_df.min()),
                                        min(max_post_th, train_hs_count_df.max()) + 1)

    train_preds = np.expand_dims(train_hs_count_df, axis=1) >= min_num_of_posts_thresholds
    train_f1_scores = [f1_score(y_train, p) for p in train_preds.T]
    best_f1_train, best_th = np.max(train_f1_scores), min_num_of_posts_thresholds[np.argmax(train_f1_scores)]

    test_hs_count_df = X_test.groupby('user_id').predictions.agg(get_hs_count, *args)
    test_preds = test_hs_count_df >= best_th
    train_preds = train_hs_count_df >= best_th
    test_f1_score = f1_score(y_test, test_preds)

    return best_th, best_f1_train, test_f1_score


def fixed_threshold_method(X: pd.DataFrame, y: pd.Series, post_threshold=0.5, test_ratio=0.2, random_state=None,
                           min_post_th=1, max_post_th=300, cv=None):
    if cv:
        args = [post_threshold]
        hs_count_df = X.groupby('user_id').predictions.agg(get_hs_count, *args)
        min_num_of_posts_thresholds = range(max(min_post_th, hs_count_df.min()),
                                            min(max_post_th, hs_count_df.max()) + 1)
        skf = StratifiedKFold(cv, random_state=random_state, shuffle=True)
        train_results_cv = []
        test_results_cv = []

        X.sort_index(inplace=True)
        y.sort_index(inplace=True)

        for train_idx, test_idx in skf.split(y, y):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            X_train = X[X['user_id'].isin(y_train.index)]
            X_test = X[X['user_id'].isin(y_test.index)]

            #             X_train.sort_index(inplace=True)
            #             y_train.sort_index(inplace=True)
            #             X_test.sort_index(inplace=True)
            #             y_test.sort_index(inplace=True)

            train_hs_count_df = X_train.groupby('user_id').predictions.agg(get_hs_count, *args)
            train_preds = np.expand_dims(train_hs_count_df, axis=1) >= min_num_of_posts_thresholds
            train_f1_scores = [f1_score(y_train, p) for p in train_preds.T]
            train_results_cv.append(train_f1_scores)
            test_hs_count_df = X_test.groupby('user_id').predictions.agg(get_hs_count, *args)
            test_preds = np.expand_dims(test_hs_count_df, axis=1) >= min_num_of_posts_thresholds
            test_f1_scores = [f1_score(y_test, p) for p in test_preds.T]
            test_results_cv.append(test_f1_scores)

        train_results_cv = np.array(train_results_cv)
        test_results_cv = np.array(test_results_cv)

        mean_train_f1_score = train_results_cv.mean(axis=0)
        best_train_f1_score = mean_train_f1_score.max()
        best_th_idx = mean_train_f1_score.argmax()
        best_th = min_num_of_posts_thresholds[best_th_idx]

        mean_test_f1_score = test_results_cv.mean(axis=0)
        test_f1_score = mean_test_f1_score[best_th_idx]

        return best_th, best_train_f1_score, test_f1_score

    else:

        y_train, y_test = train_test_split(y, test_size=0.2, random_state=random_state, stratify=y)
        X_train = X[X['user_id'].isin(y_train.index)]
        X_test = X[X['user_id'].isin(y_test.index)]

        X_train.sort_index(inplace=True)
        y_train.sort_index(inplace=True)
        X_test.sort_index(inplace=True)
        y_test.sort_index(inplace=True)

        args = [post_threshold]
        train_hs_count_df = X_train.groupby('user_id').predictions.agg(get_hs_count, *args)
        min_num_of_posts_thresholds = range(max(min_post_th, train_hs_count_df.min()),
                                            min(max_post_th, train_hs_count_df.max()) + 1)

        train_preds = np.expand_dims(train_hs_count_df, axis=1) >= min_num_of_posts_thresholds
        train_f1_scores = [f1_score(y_train, p) for p in train_preds.T]
        best_f1_train, best_th = np.max(train_f1_scores), min_num_of_posts_thresholds[np.argmax(train_f1_scores)]

        test_hs_count_df = X_test.groupby('user_id').predictions.agg(get_hs_count, *args)
        test_preds = test_hs_count_df >= best_th
        train_preds = train_hs_count_df >= best_th
        test_f1_score = f1_score(y_test, test_preds)

        return best_th, best_f1_train, test_f1_score


def extract_relational_features(X, y, min_mentions_threshold=1, min_retweets_threshold=1, post_threshold=0.5):
    user_hs_count = X.set_index('user_id').eval('`predictions`>@post_threshold').groupby('user_id').sum().rename(
        'hs_count')
    user_mean_preds = X.groupby('user_id').predictions.mean().rename('mean_preds')

    filtered_mentions_df = mentions_df.query('`weight`>=@min_mentions_threshold').drop(columns=['weight'])
    filtered_retweets_df = retweets_df.query('`weight`>=@min_retweets_threshold').drop(columns=['weight'])

    mentiones_feats_df = filtered_mentions_df.query('`source` in @y.index')
    mentiones_feats_df['mentioned_hs_count'] = mentiones_feats_df['dest'].map(user_hs_count)
    mentiones_feats_df['mentioned_mean_preds'] = mentiones_feats_df['dest'].map(user_mean_preds)
    mentiones_feats_df.fillna(0, inplace=True)

    mentioned_by_feats_df = filtered_mentions_df.query('`dest` in @y.index')
    mentioned_by_feats_df['mentioned_by_hs_count'] = mentioned_by_feats_df['source'].map(user_hs_count)
    mentioned_by_feats_df['mentioned_by_mean_preds'] = mentioned_by_feats_df['source'].map(user_mean_preds)
    mentioned_by_feats_df.fillna(0, inplace=True)

    retweeted_feats_df = filtered_retweets_df.query('`source` in @y.index')
    retweeted_feats_df['retweeted_hs_count'] = retweeted_feats_df['dest'].map(user_hs_count)
    retweeted_feats_df['retweeted_mean_preds'] = retweeted_feats_df['dest'].map(user_mean_preds)
    retweeted_feats_df.fillna(0, inplace=True)

    retweeted_by_feats_df = filtered_retweets_df.query('`dest` in @y.index')
    retweeted_by_feats_df['retweeted_by_hs_count'] = retweeted_by_feats_df['source'].map(user_hs_count)
    retweeted_by_feats_df['retweeted_by_mean_preds'] = retweeted_by_feats_df['source'].map(user_mean_preds)
    retweeted_by_feats_df.fillna(0, inplace=True)

    features_df = pd.concat([user_hs_count.loc[y.index],
                             mentiones_feats_df.groupby('source').mean(),
                             mentioned_by_feats_df.groupby('dest').mean(),
                             retweeted_feats_df.groupby('source').mean(),
                             retweeted_by_feats_df.groupby('dest').mean()
                             ], axis=1).fillna(0)  # .drop(columns=['dest','source'])
    return features_df


def relational_threshold_method_gs(X: pd.DataFrame, y: pd.DataFrame, param_grid, post_threshold=0.5, thresholds=None,
                                   feats_to_use=None, min_mentions_threshold=1, min_retweets_threshold=1,
                                   test_ratio=0.2, random_state=None):
    user_hs_count = X.set_index('user_id').eval('`predictions`>@post_threshold').groupby('user_id').sum().rename(
        'hs_count')
    user_mean_preds = X.groupby('user_id').predictions.mean().rename('mean_preds')

    y_train, y_test = train_test_split(y, test_size=0.2, random_state=random_state, stratify=y)
    y_train.sort_index(inplace=True)
    y_test.sort_index(inplace=True)

    #     filtered_mentions_df = mentions_df.query('`weight`>=@min_mentions_threshold') #.drop(columns=['weight'])
    #     filtered_retweets_df = retweets_df.query('`weight`>=@min_retweets_threshold')

    #     mentiones_feats_df = filtered_mentions_df.query('`source` in @y.index or `dest` in @y.index')
    #     mentiones_feats_df['mentioned_hs_count'] = mentiones_feats_df['dest'].map(user_hs_count)
    #     mentiones_feats_df['mentioned_by_hs_count'] = mentiones_feats_df['source'].map(user_hs_count)
    #     mentiones_feats_df['mentioned_mean_preds'] = mentiones_feats_df['dest'].map(user_mean_preds)
    #     mentiones_feats_df['mentioned_by_mean_preds'] = mentiones_feats_df['source'].map(user_mean_preds)
    #     mentiones_feats_df[['weighted_mentioned_hs_count', 'weighted_mentioned_by_hs_count', 'weighted_mentioned_mean_preds', 'weighted_mentioned_by_mean_preds']] = mentiones_feats_df[['mentioned_hs_count', 'mentioned_by_hs_count', 'mentioned_mean_preds', 'mentioned_by_mean_preds']].multiply(mentiones_feats_df["weight"], axis="index")
    #     mentiones_feats_df.fillna(0, inplace=True)

    #     retweets_feats_df = filtered_retweets_df.query('`source` in @y.index or `dest` in @y.index')
    #     retweets_feats_df['retweeted_hs_count'] = retweets_feats_df['dest'].map(user_hs_count)
    #     retweets_feats_df['retweeted_by_hs_count'] = retweets_feats_df['source'].map(user_hs_count)
    #     retweets_feats_df['retweeted_mean_preds'] = retweets_feats_df['dest'].map(user_mean_preds)
    #     retweets_feats_df['retweeted_by_mean_preds'] = retweets_feats_df['source'].map(user_mean_preds)
    #     retweets_feats_df[['weighted_retweeted_hs_count', 'weighted_retweeted_by_hs_count', 'weighted_retweeted_mean_preds', 'weighted_retweeted_by_mean_preds']] = retweets_feats_df[['retweeted_hs_count', 'retweeted_by_hs_count', 'retweeted_mean_preds', 'retweeted_by_mean_preds']].multiply(retweets_feats_df["weight"], axis="index")
    #     retweets_feats_df.fillna(0, inplace=True)

    features_df = extract_relational_features(X, y, post_threshold)

    feats_to_use = ['hs_count', 'mentioned_hs_count', 'mentioned_by_hs_count'] if feats_to_use is None else feats_to_use

    scaler = StandardScaler()

    X_train = pd.DataFrame(scaler.fit_transform(features_df.loc[y_train.index][feats_to_use]), columns=feats_to_use)
    X_test = pd.DataFrame(scaler.transform(features_df.loc[y_test.index][feats_to_use]), columns=feats_to_use)

    features_df = pd.concat([X_train, X_test])

    if thresholds is None:
        thresholds = np.linspace(0, 1, num=21)

    y_train.sort_index(inplace=True)
    X_train.sort_index(inplace=True)
    y_test.sort_index(inplace=True)
    X_test.sort_index(inplace=True)

    train_hs_scores = np.dot(X_train, param_grid.T)

    train_preds = np.expand_dims(train_hs_scores, axis=-1) >= thresholds

    kwargs = {'y_true': y_train}
    res = np.apply_along_axis(func1d=lambda y_pred, y_true: f1_score(y_true, y_pred), axis=0, arr=train_preds, **kwargs)

    idx = np.unravel_index(res.argmax(), res.shape)

    best_params, best_th, best_f1_train = param_grid[idx[0]], thresholds[idx[1]], res[idx]

    test_hs_scores = np.dot(X_test, param_grid.T)

    test_preds = np.dot(X_test, best_params) >= best_th

    train_preds = np.dot(X_train, best_params) >= best_th

    metrics_dict = {'train': calc_metrics(y_train, train_preds), 'test': calc_metrics(y_test, test_preds)}

    best_params = {k: v for k, v in zip(feats_to_use, best_params)}

    return features_df, best_params, metrics_dict


# ## Optuna

# In[157]:


def relational_threshold_method_optuna(X: pd.DataFrame, y: pd.DataFrame, dataset, min_mentions_threshold=1,
                                       min_retweets_threshold=1, test_ratio=0.2, post_threshold=0.5, random_state=None,
                                       min_th=0, max_th=1, n_trials=1000):
    """
    Here we consider the assumption that relation to followers/followees effect the users' behaviour.
    For each user - get his average HS score, and the average HS scores of his followers and followees.
    then search for the optimal relational threshold to yield the best f1-score.
    This threshold will be combined from a self-TH + followers-TH + followees-TH.

    :param X:
    :param y:
    :return:
    """

    network_dir = f"hate_networks/outputs/{dataset.split('_')[0]}_networks/network_data/"
    edges_dir = os.path.join(network_dir, "edges")
    mentions_df = pd.read_csv(os.path.join(edges_dir, "data_users_mention_edges_df.tsv"), sep='\t')
    retweets_df = pd.read_csv(os.path.join(edges_dir, "data_users_retweet_edges_df.tsv"), sep='\t')

    y_train, y_test = train_test_split(y, test_size=test_ratio, random_state=random_state, stratify=y)
    filtered_mentions_df = mentions_df.query('`weight`>=@min_mentions_threshold')
    filtered_retweets_df = retweets_df.query('`weight`>=@min_retweets_threshold')

    args = [post_threshold]
    user_hs_count = X.query('`user_id` in @y.index').groupby('user_id').predictions.mean().rename('hs_count')
    mentions_hs_count = X.query('`user_id` in @filtered_mentions_df.source').groupby(
        'user_id').predictions.mean().rename('hs_count').reset_index()
    mentioned_by_hs_count = X.query('`user_id` in @filtered_mentions_df.dest').groupby(
        'user_id').predictions.mean().rename('hs_count').reset_index()

    filtered_mentions_hs_count_df = pd.merge(
        pd.merge(filtered_mentions_df, mentions_hs_count, left_on='source', right_on='user_id', how='left'),
        mentioned_by_hs_count, left_on='dest', right_on='user_id', how='left', suffixes=('_source', '_dest')
    ).fillna(0).drop(columns=['source', 'dest', 'weight'])

    following_hs_df = filtered_mentions_hs_count_df.groupby('user_id_source').agg(
        {'hs_count_dest': ['mean', 'count', 'median']})
    following_hs_df.columns = [f'following_hs_{x[1]}' for x in following_hs_df.columns.to_flat_index()]
    followers_hs_df = filtered_mentions_hs_count_df.groupby('user_id_dest').agg(
        {'hs_count_source': ['mean', 'count', 'median']})
    followers_hs_df.columns = [f'followers_hs_{x[1]}' for x in followers_hs_df.columns.to_flat_index()]

    followees_mean_hs_count_df = filtered_mentions_hs_count_df.groupby('user_id_source')['hs_count_dest'].mean().rename(
        'following_mean_hs_count')
    followers_mean_hs_count_df = filtered_mentions_hs_count_df.groupby('user_id_dest')['hs_count_source'].mean().rename(
        'followers_mean_hs_count')

    cols = ['hs_count', 'following_hs_mean', 'followers_hs_mean']

    user_hs_count_followees_followers_mean_hs_count = pd.merge(
        pd.merge(
            user_hs_count.rename('hs_count'), following_hs_df, left_index=True, right_index=True, how='left'
        ), followers_hs_df, left_index=True, right_index=True, how='left'
    ).fillna(0)[cols]

    X_train = user_hs_count_followees_followers_mean_hs_count.loc[y_train.index]
    X_test = user_hs_count_followees_followers_mean_hs_count.loc[y_test.index]

    y_train.sort_index(inplace=True)
    X_train.sort_index(inplace=True)
    y_test.sort_index(inplace=True)
    X_test.sort_index(inplace=True)

    def objective(trial, X, y):
        self_weight = trial.suggest_float('self_weight', 0, 1)
        followers_weight = trial.suggest_float('followers_weight', 0, 1)
        following_weight = trial.suggest_float('following_weight', 0, 1)
        hs_score = np.dot(X, [self_weight, followers_weight, following_weight])
        threshold = trial.suggest_float('threshold', min_th, max_th)
        preds = hs_score >= threshold
        return f1_score(y, preds)

    study = optuna.create_study(direction="maximize", sampler=sampler)  # Create a new study.
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, show_progress_bar=True)

    train_preds = np.dot(X_train, [study.best_params['self_weight'], study.best_params['followers_weight'],
                                   study.best_params['following_weight']]) >= study.best_params['threshold']
    test_preds = np.dot(X_test, [study.best_params['self_weight'], study.best_params['followers_weight'],
                                 study.best_params['following_weight']]) >= study.best_params['threshold']

    metrics_dict = {'train': calc_metrics(y_train, train_preds), 'test': calc_metrics(y_test, test_preds)}

    return user_hs_count_followees_followers_mean_hs_count, study.best_params, metrics_dict


def dynamic_threshold_method(X: pd.DataFrame, y: pd.DataFrame, test_ratio=0.2, random_state=None, post_threshold=0.5,
                             n_trials=5000):
    hs_count_and_avg_score_per_user = X.groupby('user_id').agg(
        avg_hs_score=("predictions", "mean"),
        hs_count=("predictions", lambda p: get_hs_count(p, post_threshold)))

    y_train, y_test = train_test_split(y, test_size=test_ratio, random_state=random_state, stratify=y)
    X_train = hs_count_and_avg_score_per_user.loc[y_train.index]
    X_test = hs_count_and_avg_score_per_user.loc[y_test.index]

    y_train.sort_index(inplace=True)
    X_train.sort_index(inplace=True)
    y_test.sort_index(inplace=True)
    X_test.sort_index(inplace=True)

    def calc_soft_threshold(arr, lower_bound, higher_bound, low_th, medium_th, high_th):
        return arr[:, 1] >= np.where(arr[:, 0] < lower_bound, high_th,
                                     np.where(arr[:, 0] < higher_bound, medium_th, low_th))

    def objective(trial):
        lower_bound = trial.suggest_float("lower_bound", 0.01, 0.5)
        higher_bound = trial.suggest_float("higher_bound", lower_bound + 0.01, 1)

        low_th = trial.suggest_int("low_th", 1, np.percentile(X_train['hs_count'].values, 90).astype(int) - 1)
        medium_th = trial.suggest_int("medium_th", low_th + 1,
                                      np.percentile(X_train['hs_count'].values, 90).astype(int))
        high_th = trial.suggest_int("high_th", medium_th + 1,
                                    np.percentile(X_train['hs_count'].values, 90).astype(int) + 1)

        y_pred = calc_soft_threshold(X_train.values, lower_bound, higher_bound, low_th, medium_th, high_th)

        f1 = f1_score(y_train, y_pred)
        return f1

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner()
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_f1 = study.best_value

    test_preds = calc_soft_threshold(X_test.values, **study.best_params)
    train_preds = calc_soft_threshold(X_train.values, **study.best_params)
    metrics_dict = {'train': calc_metrics(y_train, train_preds), 'test': calc_metrics(y_test, test_preds)}
    study.best_params['post_threshold'] = post_threshold
    return study.best_params, metrics_dict

@timeit
def run_simple_ulm_experiments():
    dataset = user_level_execution_config["inference_data"]
    logger.info(f"executing dataset {dataset}...")
    model_name = post_level_execution_config["kwargs"][
        "model_name"]  # 'BertFineTuning' # post_level_execution_config["kwargs"][args.model] # new_bert_fine_tuning
    user2pred = pd.read_parquet(f"detection/outputs/{dataset}/{model_name}/user_level/split_by_posts/no_text/")
    # user2pred['user_id'] = user2pred['user_id'].astype(int)
    user2label_path = user_level_conf[dataset]["data_path"]
    sep = ","
    if user2label_path.endswith("tsv"):
        sep = "\t"
    y = pd.read_csv(user2label_path, sep=sep, index_col=[0]).squeeze()
    user2pred['user_id'] = user2pred['user_id'].astype(y.index.dtype)
    # user2pred = user2pred[user2pred['user_id'].isin(labeled_users.index)]
    X = user2pred.query('`user_id` in @y.index')

    predictions_output_path = os.path.join(post_level_execution_config["evaluation"]["output_path"], 'predictions.tsv')
    predictions_df = pd.read_csv(predictions_output_path, sep='\t')
    y_true = predictions_df['y_true']
    y_prob = predictions_df['y_score']
    y_pred = predictions_df['y_pred']

    print(f'Percent HS Users: {y.mean()}')

    test_size = 0.2
    post_threshold = optimal_threshold(y_true, y_prob)
    post_threshold=0.5

    seed = 2927694937  # random.randrange(2 ** 32)
    print("Seed is:", seed)
    output_path = f"detection/outputs/{dataset}/{model_name}/user_level/experiments/{seed}"
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists.")
    else:
        os.makedirs(output_path)
        print(f"Output path {output_path} created.")

    user_hs_count = user2pred.set_index('user_id').eval('`predictions`>@post_threshold').groupby('user_id').sum().rename(
        'hs_count')
    user_mean_preds = user2pred.groupby('user_id').predictions.mean().rename('mean_preds')

    # Run Fixed Threshold
    best_params, metrics_dict = fixed_threshold_method(X, y, post_threshold=post_threshold, test_ratio=0.2,
                                                       random_state=seed)
    best_params['post_threshold'] = post_threshold
    print(best_params)
    results_df = pd.DataFrame.from_dict(metrics_dict)

    with open(os.path.join(output_path, 'fixed_threshold_params.txt'), 'w') as f:
        f.write(json.dumps(best_params))
    results_df.to_csv(os.path.join(output_path, 'fixed_threshold_results.csv'))

    # Run Relational Threshold


    min_mentions_threshold = 1
    min_retweets_threshold = 1

    #### Grid Search - Brute Force

    feats_to_use = ['hs_count', 'mentioned_hs_count',
                    'mentioned_by_hs_count']  # , 'retweeted_hs_count', 'retweeted_by_hs_count']
    a = np.linspace(0, 1, 11)
    param_grid = np.array(np.meshgrid(*[a] * len(feats_to_use))).T.reshape(-1, len(feats_to_use))
    param_grid = param_grid[param_grid.sum(axis=1) == 1]

    features, best_params, metrics_dict = relational_threshold_method_gs(user2pred, y, param_grid, post_threshold=post_threshold, min_mentions_threshold=min_mentions_threshold, min_retweets_threshold=min_retweets_threshold, test_ratio=test_size, random_state=seed, feats_to_use=feats_to_use)
    best_params['min_mentions_threshold'] = min_mentions_threshold
    best_params['min_retweets_threshold'] = min_retweets_threshold
    best_params['post_threshold'] = post_threshold

    print(best_params)
    results_df = pd.DataFrame.from_dict(metrics_dict)

    with open(os.path.join(output_path, 'relational_threshold_gs_params.txt'), 'w') as f:
        f.write(json.dumps(best_params))
    results_df.to_csv(os.path.join(output_path, 'relational_threshold_gs_results.csv'))

    # post_threshold = optimal_threshold(y_true, y_prob)
    min_mentions_threshold = 1
    post_threshold = 0.5
    features, best_params, metrics_dict = relational_threshold_method_optuna(user2pred, y, dataset, post_threshold=post_threshold,
                                                                             min_mentions_threshold=min_mentions_threshold,
                                                                             n_trials=100, random_state=seed)
    print(best_params)
    best_params['min_mentions_threshold'] = min_mentions_threshold
    best_params['post_threshold'] = post_threshold
    results_df = pd.DataFrame.from_dict(metrics_dict)

    with open(os.path.join(output_path, 'relational_threshold_optuna_params.txt'), 'w') as f:
        f.write(json.dumps(best_params))
    results_df.to_csv(os.path.join(output_path, 'relational_threshold_optuna_results.csv'))

    # ## Run Dynamic Threshold
    best_params, results = dynamic_threshold_method(X, y, random_state=seed, post_threshold=post_threshold, n_trials=100)
    best_params['post_threshold'] = post_threshold
    print(best_params)
    results_df = pd.DataFrame.from_dict(results)

    with open(os.path.join(output_path, 'dynamic_threshold_optuna_params.txt'), 'w') as f:
        f.write(json.dumps(best_params))  # use `json.loads` to do the reverse
    results_df.to_csv(os.path.join(output_path, 'dynamic_threshold_optuna_results.csv'))

if __name__ == '__main__':
    run_simple_ulm_experiments()