# from botorch.settings import suppress_botorch_warnings
# from botorch.settings import validate_input_scaling
import pandas as pd
import numpy as np
import os, sys
import warnings
import random
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, "../.."))
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, auc, roc_curve, \
    balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config.detection_config import user_level_execution_config, user_level_conf, post_level_execution_config
from detection.detection_utils.factory import create_dir_if_missing
from collections import defaultdict

sns.set(rc={'figure.figsize': (10, 10)}, font_scale=1.4)
from scipy.optimize import minimize
from utils.my_timeit import timeit
from utils.general import init_log

logger = init_log("user_level_simple_models")


def find_optimal_threshold(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


# array([[175,  98],
#        [ 46,  81]])
def get_hs_count(current_preds, threshold=0.5):
    return len(current_preds[current_preds >= threshold])


def fixed_threshold_num_of_posts(user2pred: pd.DataFrame, labeled_users: pd.DataFrame, output_path: str,
                                 dataset_name: str,
                                 test_ratio: float, random_state=42):
    """
    Hard threshold of number of HS predictions per user. Threshold is an integer and above 1.
    :param user2pred:
    :param labeled_users:
    :return:

    Args:
        output_path:
        random_state:
        test_ratio:
        dataset_name:
    """
    logger.info("Executing fixed threshold...")
    output_path = os.path.join(output_path, "hard_threshold")
    create_dir_if_missing(output_path)
    user2pred["user_id"] = user2pred["user_id"].astype(str)
    labeled_users["user_id"] = labeled_users["user_id"].astype(str)
    # train_idx = labeled_users.sample(frac=(1 - test_ratio)).index
    train_labeled_users, test_labeled_users = train_test_split(labeled_users, test_size=test_ratio,
                                                               stratify=labeled_users["label"],
                                                               random_state=random_state)  # stratify by label

    print(f'Train Percent HS Users: {train_labeled_users["label"].mean()}')
    print(f'Test Percent HS Users: {test_labeled_users["label"].mean()}')

    train_user2pred = user2pred[user2pred["user_id"].isin(list(train_labeled_users["user_id"]))].reset_index(drop=True)
    test_user2pred = user2pred[user2pred["user_id"].isin(list(test_labeled_users["user_id"]))].reset_index(drop=True)

    train_g_df = train_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
        columns={"predictions": "hs_count"})
    test_g_df = test_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
        columns={"predictions": "hs_count"})

    to_plot = {"thresholds": [], "f-scores": [], "precisions": [], "recalls": [], "accuracies": []}

    def objective(trial):
        threshold = trial.suggest_int('threshold', min_th, max_th)
        train_g_df["y_pred"] = train_g_df["hs_count"].apply(lambda h_count: 1 if h_count >= threshold else 0)
        true_pred = pd.merge(train_labeled_users, train_g_df, on='user_id')
        y_true = true_pred["label"]
        y_pred = true_pred["y_pred"]
        current_f1_score = f1_score(y_true, y_pred)
        to_plot["thresholds"].append(threshold)
        to_plot["f-scores"].append(current_f1_score)
        to_plot["precisions"].append(precision_score(y_true, y_pred))
        to_plot["recalls"].append(recall_score(y_true, y_pred))
        to_plot["accuracies"].append(accuracy_score(y_true, y_pred))
        return current_f1_score

    min_th, max_th = 0, 300 #train_g_df.hs_count.max() + 1
    sampler = optuna.samplers.TPESampler(**optuna.samplers.TPESampler.hyperopt_parameters())
    study = optuna.create_study(direction="maximize", sampler=sampler)  # Create a new study.
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    # for threshold in tqdm(range(0, max_threshold)):
    #     to_plot["thresholds"].append(threshold)
    #     train_g_df["y_pred"] = train_g_df["hs_count"].apply(lambda h_count: 1 if h_count >= threshold else 0)
    #
    #     true_pred = pd.merge(train_labeled_users, train_g_df, on='user_id')
    #     y_true = true_pred["label"]
    #     y_pred = true_pred["y_pred"]
    #     current_f1_score = f1_score(y_true, y_pred)
    #     if max_f1 < current_f1_score:
    #         max_f1 = current_f1_score
    #         best_th = threshold
    #     to_plot["f-scores"].append(current_f1_score)
    #     to_plot["precisions"].append(precision_score(y_true, y_pred))
    #     to_plot["recalls"].append(recall_score(y_true, y_pred))
    #     to_plot["accuracies"].append(accuracy_score(y_true, y_pred))
    plt.figure(figsize=(8, 6))
    # plt.tight_layout()
    plt.xticks(range(min(to_plot["thresholds"]), max(to_plot["thresholds"]), 50 if max_th >= 200 else 2))
    sns.set(rc={'figure.figsize': (8, 6)}, font_scale=1.7)

    for score_ in ["f-score", "precision", "recall", "accuracy"]:
        current_score_name = "accuracies" if score_.endswith("y") else f"{score_}s"
        if score_ != "recall":
            sns.lineplot(to_plot["thresholds"], to_plot[current_score_name],
                         label=f"{score_}" if score_ != 'f-score' else f"{score_} (max: {max(to_plot['f-scores']):.3f})")
        else:
            sns.lineplot(to_plot["thresholds"], to_plot[current_score_name], label=f"{score_}")
    dataset_name = 'Echo' if 'echo' in dataset_name else dataset_name
    plt.title(f"Fixed threshold - {dataset_name.capitalize()}")
    plt.xlabel('Threshold')
    plt.ylabel('Measurement score')
    plt.savefig(os.path.join(output_path, "hard_threshold_plot.png"))
    pd.DataFrame(to_plot).to_csv(os.path.join(output_path, "hard_threshold.csv"), index=False)
    logger.info(f"Max f1-score: {study.best_value:.3f}")

    best_th = study.best_params["threshold"]
    logger.info(f"Best threshold: {best_th}")
    # evaluate on test
    test_g_df["y_pred"] = test_g_df["hs_count"].apply(lambda h_count: 1 if h_count >= best_th else 0)
    true_pred = pd.merge(test_labeled_users, test_g_df, on='user_id')
    y_true = true_pred["label"]
    y_pred = true_pred["y_pred"]
    with open(os.path.join(output_path, "fixed_threshold_evaluation.txt"), "w") as fout:
        fout.write(f"Threshold: {best_th}\n")
        fout.write(f"F1-score: {f1_score(y_true, y_pred):.3f}\n")
        fout.write(f"Precision: {precision_score(y_true, y_pred):.3f}\n")
        fout.write(f"Recall: {recall_score(y_true, y_pred):.3f}\n")
        fout.write(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
        fout.write(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")


def relational_threshold(user2pred: pd.DataFrame, labeled_users: pd.DataFrame, output_path: str, dataset_name: str,
                         test_ratio: float, random_state=42, brute_force=False):
    """
    Here we consider the assumption that relation to followers/followees effect the users' behaviour.
    For each user - get his average HS score, and the average HS scores of his followers and followees.
    then search for the optimal relational threshold to yield the best f1-score.
    This threshold will be combined from a self-TH + followers-TH + followees-TH.

    :param user2pred:
    :param labeled_users:
    :return:
    """
    logger.info("Executing relational threshold...")
    output_path = os.path.join(output_path, "relational_threshold")
    create_dir_if_missing(output_path)
    user2pred["user_id"] = user2pred["user_id"].astype(str)
    labeled_users["user_id"] = labeled_users["user_id"].astype(str)
    print(user2pred.shape)
    print(labeled_users.shape)
    min_mention_threshold = 3  # minimum number of mentions to consider a user as a relevant user

    # avg_hs_score_per_user = train_user2pred.groupby('user_id').agg({"predictions": "mean"}).reset_index() \
    #     .rename(columns={"predictions": "avg_hs_score"})
    # hs_count_per_user = user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
    #     columns={"predictions": "hs_count"})

    train_users, test_users = train_test_split(labeled_users, test_size=test_ratio,
                                               stratify=labeled_users["label"],
                                               random_state=random_state)  # stratify by label    # test_users = labeled_users.drop(train_users.index, axis=0)

    train_user2pred = user2pred[user2pred["user_id"].isin(list(train_users["user_id"]))].reset_index(drop=True)
    test_user2pred = user2pred[user2pred["user_id"].isin(list(test_users["user_id"]))].reset_index(drop=True)

    args = [1 - train_user2pred.predictions.mean()]

    train_g_df = train_user2pred.groupby('user_id').predictions.agg(get_hs_count, *args).reset_index().rename(
        columns={"predictions": "hs_count"})
    train_g_df = pd.merge(train_g_df, train_users, on='user_id')
    test_g_df = test_user2pred.groupby('user_id').predictions.agg(get_hs_count, *args).reset_index().rename(
        columns={"predictions": "hs_count"})
    test_g_df = pd.merge(test_g_df, test_users, on='user_id')
    print(f'Train Percent HS Users: {train_users["label"].mean()}')
    print(f'Test Percent HS Users: {test_users["label"].mean()}')

    # train_user2pred = user2pred[user2pred["user_id"].isin(list(train_users["user_id"]))].reset_index(drop=True)
    # test_user2pred = user2pred[user2pred["user_id"].isin(list(test_users["user_id"]))].reset_index(drop=True)
    #
    # train_g_df = train_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
    #     columns={"predictions": "hs_count"})
    # test_g_df = test_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
    #     columns={"predictions": "hs_count"})
    # # print(hs_count_per_user.shape)

    # get followers/followees
    network_dir = f"hate_networks/outputs/{dataset_name.split('_')[0]}_networks/network_data/"
    edges_dir = os.path.join(network_dir, "edges")
    mentions_df = pd.read_csv(os.path.join(edges_dir, "data_users_mention_edges_df.tsv"), sep='\t')
    retweets_df = pd.read_csv(os.path.join(edges_dir, "data_users_retweet_edges_df.tsv"), sep='\t')
    for col in ['source', 'dest']:
        mentions_df[col] = mentions_df[col].astype(str)
        retweets_df[col] = retweets_df[col].astype(str)
    # keep only mentions above the minimal threshold
    filtered_mentions_df = mentions_df[mentions_df["weight"] >= min_mention_threshold].reset_index(drop=True)
    mentions_dict = defaultdict(list)  # users mentioned by the observed user
    mentioned_by_dict = defaultdict(list)  # users mentioning the observed user
    for idx, row in filtered_mentions_df.iterrows():
        src = row['source']
        dest = row['dest']
        if src not in mentions_dict.keys():
            mentions_dict[src] = []
        if dest not in mentioned_by_dict.keys():
            mentioned_by_dict[dest] = []
        mentions_dict[src].append(dest)
        mentioned_by_dict[dest].append(src)

    def calc_preds(df, self_weight, followers_weight, following_weight, threshold):
        """
        Calculate the relational score for a user.
        :param user_id:
        :return:
        """
        preds = []
        for user_id, hs_count in df[["user_id", "hs_count"]].values:
            # get the average HS score of the user and its followers and followees
            followers = mentioned_by_dict[user_id]
            followees = mentions_dict[user_id]
            avg_hs_score_followers = np.nan_to_num(df[df["user_id"].isin(followers)]["hs_count"].mean())
            avg_hs_score_following = np.nan_to_num(df[df["user_id"].isin(followees)]["hs_count"].mean())
            relational_score = self_weight * hs_count + followers_weight * avg_hs_score_followers + following_weight * avg_hs_score_following
            pred = int(relational_score >= threshold)
            preds.append(pred)
        # return relational_score
        return preds

    # def calc_preds(row, df, self_weight, followers_weight, following_weight, threshold):
    #     """
    #     Calculate the relational score for a user.
    #     :param user_id:
    #     :return:
    #     """
    #     user_id = row["user_id"]
    #     hs_count = row["hs_count"]
    #     followers = mentioned_by_dict[user_id]
    #     followees = mentions_dict[user_id]
    #     avg_hs_score_followers = np.nan_to_num(df[df["user_id"].isin(followers)]["hs_count"].mean())
    #     avg_hs_score_following = np.nan_to_num(df[df["user_id"].isin(followees)]["hs_count"].mean())
    #     relational_score = self_weight * hs_count + followers_weight * avg_hs_score_followers + following_weight * avg_hs_score_following
    #     return int(relational_score >= threshold)
    # return relational_score

    to_plot = defaultdict(lambda: defaultdict(
        list))  # = {"thresholds": [], "f-scores": [], "precisions": [], "recalls": [], "accuracies": []}

    # result = pd.DataFrame()

    def objective(trial):
        # self_weight = trial.suggest_discrete_uniform('self_weight', 0, 1, 0.05)
        self_weight = trial.suggest_float('self_weight', 0, 1)
        # followers_weight = trial.suggest_discrete_uniform('followers_weight', 0, 1, 0.05)
        followers_weight = trial.suggest_float('followers_weight', 0, 1)
        following_weight = trial.suggest_float('following_weight', 0, 1)
        total = self_weight + followers_weight + following_weight
        self_weight = self_weight / total
        followers_weight = followers_weight / total
        following_weight = following_weight / total
        # type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        # following_weight = 1 - self_weight - followers_weight
        y_true = train_g_df["label"].astype(int).values
        # if brute_force:
        #     best_f1 = 0
        #     for th in range(min_th, max_th + 1, th_step):
        #         preds = calc_preds(train_g_df, self_weight, followers_weight, following_weight, th)
        #         f1 = f1_score(y_true, preds)
        #         precision = precision_score(y_true, preds)
        #         recall = recall_score(y_true, preds)
        #         accuracy = accuracy_score(y_true, preds)
        #         weights_str = f'{self_weight}_{followers_weight}_{following_weight}'
        #         to_plot[weights_str]["thresholds"].append(th)
        #         to_plot[weights_str]["f1"].append(f1)
        #         to_plot[weights_str]["precision"].append(precision)
        #         to_plot[weights_str]["recall"].append(recall)
        #         to_plot[weights_str]["accuracy"].append(accuracy)
        #         if f1 > best_f1:
        #             best_f1 = f1
        #             trial.set_user_attr("best_threshold", th)
        #     current_f1_score = best_f1
        # else:
        threshold = trial.suggest_int('threshold', min_th, max_th, step=th_step)
        preds = calc_preds(train_g_df, self_weight, followers_weight, following_weight, threshold)
        current_f1_score = f1_score(y_true, preds)
        precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)
        accuracy = accuracy_score(y_true, preds)
        weights_str = f'{self_weight}_{followers_weight}_{following_weight}'
        to_plot[weights_str]["thresholds"].append(threshold)
        to_plot[weights_str]["f1"].append(current_f1_score)
        to_plot[weights_str]["precision"].append(precision)
        to_plot[weights_str]["recall"].append(recall)
        to_plot[weights_str]["accuracy"].append(accuracy)
        # def threshold_objective(t):
        #     threshold = t.suggest_int('threshold', min_th, max_th, step=th_step)
        #     preds = calc_preds(train_g_df, self_weight, followers_weight, following_weight, threshold)
        #     f1 = f1_score(y_true, preds)
        #     precision = precision_score(y_true, preds)
        #     recall = recall_score(y_true, preds)
        #     accuracy = accuracy_score(y_true, preds)
        #     weights_str = f'{self_weight}_{followers_weight}_{following_weight}'
        #     to_plot[weights_str]["thresholds"].append(threshold)
        #     to_plot[weights_str]["f1"].append(f1)
        #     to_plot[weights_str]["precision"].append(precision)
        #     to_plot[weights_str]["recall"].append(recall)
        #     to_plot[weights_str]["accuracy"].append(accuracy)
        #     return f1
        #
        # th_sampler = optuna.samplers.TPESampler(**optuna.samplers.TPESampler.hyperopt_parameters())
        # th_study = optuna.create_study(direction="maximize", sampler=th_sampler)  # Create a new study.
        # th_study.optimize(threshold_objective, n_trials=20)
        # trial.set_user_attr("best_threshold", th_study.best_params["threshold"])
        # current_f1_score = th_study.best_value
        # result = pd.concat(
        #     [result, pd.DataFrame([{"self_weight": self_weight, "followers_weight": followers_weight,
        #                          "followees_weight": following_weight, "th": threshold,
        #                          "f1_score": current_f1_score, "precision": current_precision_score,
        #                          "recall": current_recall_score, "accuracy": current_accuracy_score}])],
        #     ignore_index=True)
        return current_f1_score

    min_th, max_th = 1, 300 #round(np.percentile(train_g_df['hs_count'], 95))  # train_g_df.hs_count.max() + 1
    th_step = 1
    sampler = optuna.samplers.TPESampler(**optuna.samplers.TPESampler.hyperopt_parameters())
    study = optuna.create_study(direction="maximize", sampler=sampler)  # Create a new study.
    study.optimize(objective, n_trials=1000, show_progress_bar=True)

    self_weight = study.best_params['self_weight']
    followers_weight = study.best_params['followers_weight']
    following_weight = study.best_params['following_weight']
    threshold = study.best_trial.user_attrs["best_threshold"] if brute_force else study.best_trial.params['threshold']

    total = self_weight + followers_weight + following_weight
    self_weight = self_weight / total
    followers_weight = followers_weight / total
    following_weight = following_weight / total

    best_f1 = study.best_value
    logger.info(f"Max f1-score: {best_f1}")
    logger.info(f"Best threshold: {threshold}")
    logger.info(f"Best self_weight: {self_weight}")
    logger.info(f"Best followers_weight: {followers_weight}")
    logger.info(f"Best following_weight: {following_weight}")
    # if brute_force:
    weights_str = f'{self_weight}_{followers_weight}_{following_weight}'
    dataset_name = 'Echo' if 'echo' in dataset_name else dataset_name
    plt.figure(figsize=(8, 6))
    plt.tight_layout()
    plt.xticks(range(min(to_plot[weights_str]["thresholds"]),
                     max(to_plot[weights_str]["thresholds"]),
                     max(1, (max(to_plot[weights_str]["thresholds"]) - min(to_plot[weights_str]["thresholds"])) // 10)))
    sns.set(rc={'figure.figsize': (8, 6)}, font_scale=1.5)
    for score_ in ["f1", "precision", "recall", "accuracy"]:
        if score_ != "recall":
            sns.lineplot(to_plot[weights_str]["thresholds"], to_plot[weights_str][score_],
                         label=f"{score_}" if score_ != 'f1' else f"{score_} (max: {max(to_plot[weights_str]['f1']):.3f})")
        else:
            sns.lineplot(to_plot[weights_str]["thresholds"], to_plot[weights_str][score_],
                         label=f"{score_}")
    plt.title(f"Releational threshold - {dataset_name.capitalize()}")
    plt.xlabel('Threshold')
    plt.ylabel('Measurement score')
    plt.savefig(os.path.join(output_path,
                             f"relational_threshold_plot.png"),
                # bbox_inches="tight"
                )

    y_true = test_g_df["label"].values
    y_pred = calc_preds(test_g_df, self_weight, followers_weight, following_weight, threshold)

    with open(os.path.join(output_path, "relational_threshold_evaluation.txt"), "w") as fout:
        fout.write(f"self_weight: {round(self_weight, 2)}\n")
        fout.write(f"followers_weight: {round(followers_weight, 2)}\n")
        fout.write(f"following_weight: {round(following_weight, 2)}\n")
        fout.write(f"Train best_f1: {best_f1}\n")
        fout.write(f"best_th: {threshold}\n")
        fout.write("Test set:\n")
        fout.write(f"F1-score: {f1_score(y_true, y_pred):.3f}\n")
        fout.write(f"Precision: {precision_score(y_true, y_pred):.3f}\n")
        fout.write(f"Recall: {recall_score(y_true, y_pred):.3f}\n")
        fout.write(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
        fout.write(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")

    # res.to_csv(os.path.join(output_path, "relational_threshold.csv"), index=False)


def dynamic_threshold_hs_score(user2pred: pd.DataFrame, labeled_users: pd.DataFrame, output_path: str,
                               test_ratio: float, random_state=42):
    """

    :param user2pred:
    :param labeled_users:
    :param output_path:
    :return:
    """
    logger.info("Executing dynamic/adjusted threshold...")
    output_path = os.path.join(output_path, "soft_threshold")
    create_dir_if_missing(output_path)
    user2pred["user_id"] = user2pred["user_id"].astype(str)
    labeled_users["user_id"] = labeled_users["user_id"].astype(str)
    user2pred = user2pred[user2pred["user_id"].isin(list(labeled_users["user_id"]))].reset_index(drop=True)
    hs_count_and_avg_score_per_user = user2pred.groupby('user_id', as_index=False).agg(
        avg_hs_score=("predictions", "mean"),
        hs_count=("predictions", get_hs_count))

    train_users, test_users = train_test_split(labeled_users, test_size=test_ratio,
                                               stratify=labeled_users["label"],
                                               random_state=random_state)  # stratify by label

    train_g_df = pd.merge(hs_count_and_avg_score_per_user, train_users, on='user_id')
    test_g_df = pd.merge(hs_count_and_avg_score_per_user, test_users, on='user_id')

    print(f'Train Percent HS Users: {train_users["label"].mean()}')
    print(f'Test Percent HS Users: {test_users["label"].mean()}')

    # hs_count_per_user = user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
    #     columns={"predictions": "hs_count"})

    # res = pd.DataFrame(
    #     columns=["lower_bound", "higher_bound", "low_th", "medium_th", "high_th", "f1_score", "precision_score",
    #              "recall_score", "accuracy_score"])
    min_low_th, min_mid_th, min_high_th = 1, 3, 10
    max_low_th, max_mid_th, max_high_th = 5, 10, 60  # np.percentile(train_g_df['hs_count'], 95)

    def calc_soft_threshold(df, lower_bound, higher_bound, low_th, medium_th, high_th):
        preds = []
        for user_id, hs_score, hs_count in df[["user_id", "avg_hs_score", "hs_count"]].values:
            if hs_score < lower_bound:
                th = high_th
            elif lower_bound <= hs_score < higher_bound:
                th = medium_th
            else:
                th = low_th
            preds.append(int(hs_count >= th))
        return preds

    def objective(trial):
        # lower_bound = trial.suggest_float("lower_bound", 0.01, np.percentile(train_g_df['avg_hs_score'], 50))
        lower_bound = trial.suggest_float("lower_bound", 0.1, 0.4, step=0.1)
        # higher_bound = trial.suggest_float("higher_bound", lower_bound + 0.01, train_g_df['avg_hs_score'].max())
        higher_bound = trial.suggest_float("higher_bound", lower_bound + 0.1, 0.6, step=0.1)
        # bounds = sorted([lower_bound, higher_bound])
        low_th = trial.suggest_int("low_th", 1, 10)
        medium_th = trial.suggest_int("medium_th", low_th + 1, 50)
        high_th = trial.suggest_int("high_th", medium_th + 1, 200)
        # thresholds = sorted([low_th, medium_th, high_th])

        c0 = float(0.01 + lower_bound - higher_bound)
        c1 = float(1 + medium_th - high_th)
        c2 = float(1 + low_th - medium_th)

        # Store the constraints as user attributes so that they can be restored after optimization.
        trial.set_user_attr("constraint", (c0, c1, c2))

        # if high_th <= medium_th or medium_th <= low_th or lower_bound >= higher_bound:
        #     raise optuna.exceptions.TrialPruned()

        y_pred = calc_soft_threshold(train_g_df, lower_bound, higher_bound, low_th, medium_th, high_th)
        y_true = train_g_df["label"].values

        f1 = f1_score(y_true, y_pred)
        return f1

    def constraints(trial):
        return trial.user_attrs["constraint"]

    # sampler = optuna.integration.BoTorchSampler(
    #     constraints_func=constraints,
    #     n_startup_trials=10,
    # )
    sampler = optuna.samplers.NSGAIISampler(
        constraints_func=constraints
    )
    # sampler = optuna.samplers.TPESampler(**optuna.samplers.TPESampler.hyperopt_parameters())

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
    )
    study.optimize(objective, n_trials=5000, show_progress_bar=True)

    lower_bound = study.best_params["lower_bound"]
    higher_bound = study.best_params["higher_bound"]
    low_th = study.best_params["low_th"]
    medium_th = study.best_params["medium_th"]
    high_th = study.best_params["high_th"]
    # threshold = study.best_trial.params['threshold']

    best_f1 = study.best_value
    logger.info(f"Max f1-score: {best_f1}")
    # logger.info(f"Best threshold: {threshold}")

    y_true = test_g_df["label"].values
    y_pred = calc_soft_threshold(test_g_df, lower_bound, higher_bound, low_th, medium_th, high_th)

    with open(os.path.join(output_path, "dynamic_threshold_evaluation.txt"), "w") as fout:
        fout.write(f"Lower bound: {lower_bound}\n")
        fout.write(f"Higher bound: {higher_bound}\n")
        fout.write(f"Low threshold: {low_th}\n")
        fout.write(f"Medium threshold: {medium_th}\n")
        fout.write(f"High threshold: {high_th}\n")
        fout.write(f"Train best_f1: {best_f1}\n")
        # fout.write(f"best_th: {threshold}\n")
        fout.write("Test set:\n")
        fout.write(f"F1-score: {f1_score(y_true, y_pred):.3f}\n")
        fout.write(f"Precision: {precision_score(y_true, y_pred):.3f}\n")
        fout.write(f"Recall: {recall_score(y_true, y_pred):.3f}\n")
        fout.write(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
        fout.write(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")

    # res.to_csv(os.path.join(output_path, "soft_threshold.csv"), index=False)
    # return res


@timeit
def run_simple_ulm_experiments():
    # take the dataset to predict from config
    dataset = user_level_execution_config["inference_data"]

    logger.info(f"executing dataset {dataset}...")
    model_name = post_level_execution_config["kwargs"]["model_name"]  # new_bert_fine_tuning
    user2pred = pd.read_parquet(f"detection/outputs/{dataset}/{model_name}/user_level/split_by_posts/no_text/")
    user2label_path = user_level_conf[dataset]["data_path"]
    sep = ","
    if user2label_path.endswith("tsv"):
        sep = "\t"
    labeled_users = pd.read_csv(user2label_path, sep=sep)

    seed = random.randrange(2 ** 32)
    # seed = 579566482 #3327495705 # 1297126752 # 2198782776 # 1297126752  # 2382227748  # 2334642105 # 1472770416 # 42  # 338761188
    print("Seed is:", seed)
    output_path = f"detection/outputs/{dataset}/{model_name}/user_level/optuna/{seed}"
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists.")
    else:
        os.makedirs(output_path)
        print(f"Output path {output_path} created.")
    # rng = random.Random(seed)
    # dataset = 'Echo' if 'echo' in dataset else dataset
    fixed_threshold_num_of_posts(user2pred, labeled_users, output_path, dataset, test_ratio=0.2, random_state=seed)
    relational_threshold(user2pred, labeled_users, output_path, dataset, test_ratio=0.2, random_state=seed, brute_force=False)
    dynamic_threshold_hs_score(user2pred, labeled_users, output_path, test_ratio=0.2, random_state=seed)


if __name__ == '__main__':
    run_simple_ulm_experiments()
