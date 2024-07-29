from config.detection_config import user_level_execution_config, user_level_conf, post_level_execution_config
from utils.general import init_log
from detection.aggregative_methods.FixedAggregation import get_fixed_threshold_pipeline
from detection.aggregative_methods.RelationalAggregation import get_relational_aggregation_pipeline
from detection.aggregative_methods.DynamicAggregation import get_dynamic_aggregation_pipeline
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score, \
    roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
import os
import sys
import json
from functools import reduce
from tabulate import tabulate
import warnings
pd.options.mode.chained_assignment = None # default='warn'


warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

logger = init_log("user_level_aggregate_methods_experiment")
scoring_list = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
scoring_names = ['_'.join(f.__name__.split('_')[:-1]) for f in scoring_list]
scoring_dict = {n: f for n, f in zip(scoring_names, scoring_list)}


# def expect_precision(y_true, y_prob, thres):
#     idxs = np.where(y_prob >= thres)[0]
#     tp = y_prob[idxs].sum()
#     fp = len(idxs) - tp
#     # idxs = np.where(y_prob < thres)[0]
#     # fn = y_prob[idxs].sum()
#     return tp / (tp + fp)


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


# def get_hs_count(current_preds, threshold=0.5):
#     return len(current_preds[current_preds > threshold])
#
#
# def calc_metrics(y_true, y_pred):
#     return {f.__name__: f(y_true, y_pred) for f in scoring_list}


def write_method_params_and_results(method_name, output_path, results_dict, overwrite=False):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for name, df in results_dict.items():
        path = os.path.join(output_path, f'{method_name}_{name}_results')
        # print(results_path)
        if not os.path.exists(f'{path}.csv') or overwrite:
            df.to_csv(f'{path}.csv')
        else:
            print(f'{name}.csv file already exists')
        if name != 'params':
            if (not os.path.exists(f'{path}.txt') or overwrite):
                np.savetxt(f'{path}.txt', df.values.T, fmt='%.3f', delimiter=' & ')
            else:
                print(f'{name}.txt file already exists')


def read_method_params_and_results(method_name, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    params_path = os.path.join(output_path, f'{method_name}_best_params.txt')
    train_results_path = os.path.join(output_path, f'{method_name}_train_results.csv')
    test_results_path = os.path.join(output_path, f'{method_name}_test_results.csv')
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    train_results = pd.read_csv(train_results_path)
    test_results = pd.read_csv(test_results_path)
    metrics_dict = {'train': train_results, 'test': test_results}
    return best_params, metrics_dict


def get_best_results_cv(results_cv, scoring_name=None, best_score_idx=None):
    scores_mean = results_cv.mean(axis=0)
    scores_std = results_cv.std(axis=0)

    if scoring_name is not None and best_score_idx is None:
        scoring_idx = get_scoring_idx_by_name(scoring_name)
        scoring_mean = scores_mean[scoring_idx]
        scoring_std = scores_std[scoring_idx]
        best_score_idx = scoring_mean.argmax() if not best_score_idx else best_score_idx

        best_scores_mean = scores_mean[:, best_score_idx]
        best_scores_std = scores_std[:, best_score_idx]
        results_df = pd.DataFrame([best_scores_mean, best_scores_std], columns=scoring_names, index=['mean', 'std']).T

        return results_df, best_score_idx

    best_scores_mean = scores_mean[:, best_score_idx]
    best_scores_std = scores_std[:, best_score_idx]
    results_df = pd.DataFrame([best_scores_mean, best_scores_std], columns=scoring_names, index=['mean', 'std']).T

    return results_df


def get_scoring_idx_by_name(name):
    return [idx for idx, e in enumerate(scoring_list) if name in e.__name__][0]


# def write_results_latex(df, name):
#     with open(os.path.join(method_output_path, f'{method_name}_{name}_results_latex.txt'), 'w') as f:
#         res = ' & '.join([f'${m:.3f} \pm {s:.3f}$' for m, s in df.values.reshape(-1, 2)]) + '\\\\'
#         f.write(res)
#
#
def get_best_results_from_gs(gs):
    train_results = pd.DataFrame(pd.DataFrame.from_dict(gs.cv_results_).loc[
                                     gs.best_index_, [c for c in gs.cv_results_ if
                                                      'mean_train' in c or 'std_train' in c]].values.reshape(-1, 2),
                                 columns=['mean', 'std'], index=scoring_names[:-1])
    test_results = pd.DataFrame(pd.DataFrame.from_dict(gs.cv_results_).loc[gs.best_index_, [c for c in gs.cv_results_ if
                                                                                            'mean_test' in c or 'std_test' in c]].values.reshape(
        -1, 2), columns=['mean', 'std'], index=scoring_names[:-1])

    best_hyper_params = pd.Series(
        reduce(lambda a, b: a | b, [e[1].get_params() for e in gs.best_estimator_.get_params(deep=False)['steps']]))
    # .to_csv(os.path.join(method_output_path, 'best_hyperparams.csv'))
    return best_hyper_params, {'train': train_results, 'test': test_results}


#
#
# def write_best_results_and_params_from_gs(gs):
#     results_dict = get_best_results_from_gs(gs)
#     pd.Series(gs.best_estimator_.get_params()).rename('value').to_csv(
#         os.path.join(method_output_path, 'best_hyperparams.csv'))
#     pd.Series(gs.best_estimator_.get_learned_params()).rename('value').to_csv(
#         os.path.join(method_output_path, 'best_params.csv'))
#
#     for name, df in results_dict.items():
#         write_results_latex(df, name=name)
#         df.to_csv(os.path.join(method_output_path, f'best_results_{name}.csv'))
#
#     return results_dict


def run_aggregation_methods():
    dataset = user_level_execution_config["inference_data"]
    logger.info(f"loading dataset {dataset}...")
    model_name = post_level_execution_config["kwargs"]["model_name"]
    predictions_output_path = os.path.join(post_level_execution_config["evaluation"]["output_path"], 'predictions.tsv')
    users_posts_predictions = pd.read_parquet(
        f"detection/outputs/{dataset}/{model_name}/user_level/split_by_posts/no_text/")
    user2label_path = user_level_conf[dataset]["data_path"]
    user_key = user_level_conf[dataset]['user_unique_column']
    network_dir = f"hate_networks/outputs/{dataset.split('_')[0]}_networks/network_data/"

    sep = "\t" if user2label_path.endswith("tsv") else ","
    y = pd.read_csv(user2label_path, sep=sep, index_col=[0]).squeeze()
    users_posts_predictions['user_id'] = users_posts_predictions['user_id'].astype(y.index.dtype)
    logger.info(f'Percent HS Users: {y.mean()}')

    predictions_df = pd.read_csv(predictions_output_path, sep='\t')
    y_true = predictions_df['y_true']
    y_prob = predictions_df['y_score']

    users_df = users_posts_predictions.groupby(user_key)['predictions'].apply(np.array).reset_index().set_index(user_key)
    X_not_flat = users_posts_predictions.query('user_id in @y.index')
    X = users_df.loc[y.index]

    logger.info("Generating seed...")
    seed = 555  # random.randrange(2 ** 32)
    logger.info(f"Seed is: {seed}")
    output_path = f"detection/outputs/{dataset}/{model_name}/user_level/experiments/{seed}"
    if os.path.exists(output_path):
        logger.info(f"Output path {output_path} already exists.")
    else:
        os.makedirs(output_path)
        logger.info(f"Output path {output_path} created.")

    logger.info("Fixed Threshold Classifier")
    run_fixed_threshold(X, y, y_true, y_prob, users_posts_predictions, random_state=seed)

    logger.info("Relational Aggregation Method")
    run_relational_aggregation(X, y, y_true, y_prob, users_df, network_dir, random_state=seed)

    logger.info("Dynamic Aggregation Method")
    run_dynamic_aggregation(X, y, y_true, y_prob, users_df, random_state=seed)


def run_fixed_threshold(X, y, y_true, y_prob, users_df, random_state=None):
    # out = pd.qcut(users_df['predictions'], q=3)
    # thresholds = np.array([x[1] for x in out.cat.categories.to_tuples()])

    param_grid = {
        'hs_count__post_threshold': [0.1, 0.25, 0.5, 0.75, 0.9] + [optimal_threshold(y_true, y_prob)]
    }

    estimator = get_fixed_threshold_pipeline()
    best_hyper_params, results_dict = run_method(estimator, X, y, param_grid, random_state)
    print(best_hyper_params.to_dict())
    print(tabulate(results_dict['test'], headers='keys', tablefmt='psql'))
    return estimator, best_hyper_params, results_dict


def run_relational_aggregation(X, y, y_true, y_prob, users_df_flat, network_dir, random_state=None):
    edges_dir = os.path.join(network_dir, "edges")
    mentions_df = pd.read_csv(os.path.join(edges_dir, "data_users_mention_edges_df.tsv"), sep='\t')
    retweets_df = pd.read_csv(os.path.join(edges_dir, "data_users_retweet_edges_df.tsv"), sep='\t')

    weights = np.array(np.meshgrid(*[np.linspace(0, 1, 11)] * 3)).T.reshape(-1, 3)
    weights = weights[weights.sum(axis=1) == 1]

    # param_grid = [{
    #     'user_hs_count_and_mean__hs_count__post_threshold': [0.1, 0.25, 0.5, 0.75, 0.9] + [
    #         optimal_threshold(y_true, y_prob)],
    #     'clf__self_weight': self_weight,
    #     'clf__followers_weight': followers_weight,
    #     'clf__following_weight': following_weight,
    #     'clf__user_threshold': np.arange(1, 300, 10)
    # } for self_weight, followers_weight, following_weight in weights]

    param_grid = {
        'user_hs_count_and_mean__hs_count__post_threshold': [0.5, 0.75, 0.9, optimal_threshold(y_true, y_prob)],
        'clf__C': np.logspace(-2, 0, 7),
        'clf__penalty': ['l1', 'l2']
    }

    estimator = get_relational_aggregation_pipeline(users_df_flat, mentions_df, retweets_df, random_state=random_state)
    best_hyper_params, results_dict = run_method(estimator, X, y, param_grid, random_state)
    return estimator, best_hyper_params, results_dict


def run_dynamic_aggregation(X, y, y_true, y_prob, users_df, random_state=None):
    estimator = get_dynamic_aggregation_pipeline()

    param_grid = {
        'user_hs_count_and_mean__hs_count__post_threshold': [0.1, 0.25, 0.5, 0.75, 0.9] + [optimal_threshold(y_true, y_prob)],
        'clf__lower_bound': [0.1, 0.25, 0.5, 0.75, 0.9],
        'clf__upper_bound': [0.1, 0.25, 0.5, 0.75, 0.9],
        'clf__low_th': []
    }

    best_hyper_params, results_dict = run_method(estimator, X, y, param_grid, random_state)
    return estimator, best_hyper_params, results_dict


def run_method(estimator, X, y, param_grid, random_state=None):
    gs = GridSearchCV(estimator, param_grid=param_grid, scoring=scoring_names[:-1],
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
                      return_train_score=True, refit='f1')
    gs.fit(X, y)
    best_hyper_params, results_dict = get_best_results_from_gs(gs)

    print(best_hyper_params.to_dict())
    print('train results')
    print(tabulate(results_dict['train'], headers='keys', tablefmt='psql'))
    print('test results')
    print(tabulate(results_dict['test'], headers='keys', tablefmt='psql'))
    return best_hyper_params, results_dict


if __name__ == "__main__":
    run_aggregation_methods()
