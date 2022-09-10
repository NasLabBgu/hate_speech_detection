import pandas as pd
import numpy as np
import os, sys
import warnings
import random

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

sns.set(rc={'figure.figsize': (10, 10)}, font_scale=1.4)
from scipy.optimize import minimize
from utils.my_timeit import timeit
from utils.general import init_log

logger = init_log("user_level_simple_models")


def get_hs_count(current_preds):
    return len(current_preds[current_preds > 0.5])


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
    max_f1 = 0.0
    best_th = 0
    min_th, max_th = 0, min(300, train_g_df.hs_count.max()) + 1
    for threshold in tqdm(range(min_th, max_th)):
        to_plot["thresholds"].append(threshold)
        train_g_df["y_pred"] = train_g_df["hs_count"].apply(lambda h_count: 1 if h_count >= threshold else 0)

        true_pred = pd.merge(train_labeled_users, train_g_df, on='user_id')
        y_true = true_pred["label"]
        y_pred = true_pred["y_pred"]
        current_f1_score = f1_score(y_true, y_pred)
        if max_f1 < current_f1_score:
            max_f1 = current_f1_score
            best_th = threshold
        to_plot["f-scores"].append(current_f1_score)
        to_plot["precisions"].append(precision_score(y_true, y_pred))
        to_plot["recalls"].append(recall_score(y_true, y_pred))
        to_plot["accuracies"].append(accuracy_score(y_true, y_pred))
    plt.figure(figsize=(8, 6))
    # plt.tight_layout()
    plt.xticks(range(0, max_th, 50 if max_th >= 200 else 2))
    sns.set(rc={'figure.figsize': (8, 6)}, font_scale=1.5)

    for score_ in ["f-score", "precision", "recall", "accuracy"]:
        current_score_name = "accuracies" if score_.endswith("y") else f"{score_}s"
        if score_ != "recall":
            sns.lineplot(to_plot["thresholds"], to_plot[current_score_name],
                         label=f"{score_}" if score_ != 'f-score' else f"{score_} (max: {max(to_plot['f-scores']):.3f})")
        else:
            sns.lineplot(to_plot["thresholds"], to_plot[current_score_name], label=f"{score_}")
    plt.title(f"Fixed threshold - {dataset_name.capitalize()}")
    plt.xlabel('Threshold')
    plt.ylabel('Measurement score')
    plt.savefig(os.path.join(output_path, "hard_threshold_plot.png"))
    pd.DataFrame(to_plot).to_csv(os.path.join(output_path, "hard_threshold.csv"), index=False)
    logger.info(f"Max f1-score: {max_f1}")
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
                         test_ratio: float, random_state=42):
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
    hs_count_per_user = user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
        columns={"predictions": "hs_count"})

    train_labeled_users, test_labeled_users = train_test_split(labeled_users, test_size=test_ratio,
                                                               stratify=labeled_users["label"],
                                                               random_state=random_state)  # stratify by label    # test_labeled_users = labeled_users.drop(train_labeled_users.index, axis=0)
    print(f'Train Percent HS Users: {train_labeled_users["label"].mean()}')
    print(f'Test Percent HS Users: {test_labeled_users["label"].mean()}')

    train_user2pred = user2pred[user2pred["user_id"].isin(list(train_labeled_users["user_id"]))].reset_index(drop=True)
    test_user2pred = user2pred[user2pred["user_id"].isin(list(test_labeled_users["user_id"]))].reset_index(drop=True)

    train_g_df = train_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
        columns={"predictions": "hs_count"})

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
    mentions_dict = {}  # users mentioned by the observed user
    mentioned_by_dict = {}  # users mentioning the observed user
    for idx, row in filtered_mentions_df.iterrows():
        src = row['source']
        dest = row['dest']
        if src not in mentions_dict.keys():
            mentions_dict[src] = []
        if dest not in mentioned_by_dict.keys():
            mentioned_by_dict[dest] = []
        mentions_dict[src].append(dest)
        mentioned_by_dict[dest].append(src)
    res = pd.DataFrame()
    # SELF_WEIGHT = 0.5
    # FOLLOWERS_WEIGHT = 0.25
    # FOLLOWEES_WEIGHT = 0.25
    best_f1 = 0.0
    for SELF_WEIGHT in tqdm(np.linspace(0, 1, num=11)):
        for FOLLOWERS_WEIGHT in np.linspace(0, 1, num=11):
            if SELF_WEIGHT + FOLLOWERS_WEIGHT >= 1:
                # if FOLLOWEES_WEIGHT + FOLLOWERS_WEIGHT >= 1:
                # tmp = SELF_WEIGHT + FOLLOWERS_WEIGHT
                # SELF_WEIGHT = SELF_WEIGHT / tmp
                # FOLLOWERS_WEIGHT = FOLLOWERS_WEIGHT / tmp
                continue
            # else:
            FOLLOWEES_WEIGHT = 1.0 - SELF_WEIGHT - FOLLOWERS_WEIGHT
            # SELF_WEIGHT = 1.0 - FOLLOWEES_WEIGHT - FOLLOWERS_WEIGHT

            # logger.info(f"self-weight: {SELF_WEIGHT:.2f}, followers-weight: {FOLLOWERS_WEIGHT:.2f}, followees-weight: {FOLLOWEES_WEIGHT:.2f}")
            user_ids = []
            relational_scores = []
            type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for user_id in labeled_users["user_id"].tolist():
                if user_id not in hs_count_per_user.user_id.tolist():
                    continue
                user_ids.append(user_id)
                has_followees = True
                has_followers = True
                if user_id in mentions_dict.keys():
                    current_followees = mentions_dict[user_id]
                    followees_df = hs_count_per_user.loc[
                        hs_count_per_user["user_id"].isin(current_followees), "hs_count"]
                    if len(followees_df) == 0:
                        has_followees = False
                    else:
                        followees_hs_counts = followees_df.mean()
                else:
                    has_followees = False
                if user_id in mentioned_by_dict.keys():
                    current_followers = mentioned_by_dict[user_id]
                    followers_df = hs_count_per_user.loc[
                        hs_count_per_user["user_id"].isin(current_followers), "hs_count"]
                    if len(followers_df) == 0:
                        has_followers = False
                    else:
                        followers_hs_counts = followers_df.mean()
                else:
                    has_followers = False

                user_hs_count = int(
                    hs_count_per_user.loc[hs_count_per_user["user_id"] == user_id, "hs_count"].iloc[0])
                if has_followers and has_followees:
                    type_counts[1] += 1
                    current_score = SELF_WEIGHT * user_hs_count + FOLLOWEES_WEIGHT * followees_hs_counts + FOLLOWERS_WEIGHT * followers_hs_counts
                elif has_followees and not has_followers:
                    type_counts[2] += 1
                    current_score = SELF_WEIGHT * user_hs_count + FOLLOWEES_WEIGHT * followees_hs_counts
                elif not has_followees and has_followers:
                    type_counts[3] += 1
                    current_score = SELF_WEIGHT * user_hs_count + FOLLOWERS_WEIGHT * followers_hs_counts
                else:
                    type_counts[4] += 1
                    current_score = SELF_WEIGHT * user_hs_count

                relational_scores.append(current_score)
            # logger.info(type_counts)
            user2relational_score = pd.DataFrame({"user_id": user_ids, "relational_score": relational_scores})

            train_user2relational_score = pd.merge(train_labeled_users, user2relational_score, on="user_id",
                                                   how="left")
            test_user2relational_score = pd.merge(test_labeled_users, user2relational_score, on="user_id",
                                                  how="left")

            to_plot = {"thresholds": [], "f-scores": [], "precisions": [], "recalls": [], "accuracies": []}
            max_f1 = 0.0
            best_th = 0
            min_th, max_th = 0, min(300, train_g_df.hs_count.max()) + 1
            for threshold in range(min_th, max_th):
                # to_plot["thresholds"].append(threshold)
                train_user2relational_score["y_pred"] = train_user2relational_score["relational_score"].apply(
                    lambda rs: 1 if rs >= threshold else 0)
                y_true = train_user2relational_score["label"]
                y_pred = train_user2relational_score["y_pred"]
                current_f1_score = f1_score(y_true, y_pred)
                current_precision_score = precision_score(y_true, y_pred)
                current_recall_score = recall_score(y_true, y_pred)
                current_accuracy_score = accuracy_score(y_true, y_pred)
                if max_f1 < current_f1_score:
                    max_f1 = current_f1_score
                    best_th = threshold
                to_plot["thresholds"].append(threshold)
                to_plot["f-scores"].append(current_f1_score)
                to_plot["precisions"].append(current_precision_score)
                to_plot["recalls"].append(current_recall_score)
                to_plot["accuracies"].append(current_accuracy_score)
                res = pd.concat(
                    [res, pd.DataFrame([{"self_weight": SELF_WEIGHT, "followers_weight": FOLLOWERS_WEIGHT,
                                         "followees_weight": FOLLOWEES_WEIGHT, 'best_f1': max_f1,
                                         'best_th': best_th,
                                         "th": threshold,
                                         "f1_score": current_f1_score, "precision": current_precision_score,
                                         "recall": current_recall_score, "accuracy": current_accuracy_score}])],
                    ignore_index=True)
            plt.figure(figsize=(8, 6))
            plt.tight_layout()
            plt.xticks(range(min_th, max_th, 50 if max_th >= 200 else 2))
            sns.set(rc={'figure.figsize': (8, 6)}, font_scale=1.5)

            dataset_name = 'Echo' if 'echo' in dataset_name else dataset_name
            if max_f1 > best_f1:
                logger.info(f"Max f1-score: {max_f1}")
                logger.info(f"Best threshold: {best_th}")
                best_f1 = max_f1
                for score_ in ["f-score", "precision", "recall", "accuracy"]:
                    current_score_name = "accuracies" if score_.endswith("y") else f"{score_}s"
                    if score_ != "recall":
                        sns.lineplot(to_plot["thresholds"], to_plot[current_score_name],
                                     label=f"{score_}" if score_ != 'f-score' else f"{score_} (max: {max(to_plot['f-scores']):.3f})")
                    else:
                        sns.lineplot(to_plot["thresholds"], to_plot[current_score_name], label=f"{score_}")
                plt.title(f"Releational threshold - {dataset_name.capitalize()}")
                plt.xlabel('Threshold')
                plt.ylabel('Measurement score')
                plt.savefig(os.path.join(output_path,
                                         f"relational_threshold_{round(SELF_WEIGHT, 2)}_{round(FOLLOWERS_WEIGHT, 2)}_{round(FOLLOWEES_WEIGHT, 2)}_plot.png"),
                            # bbox_inches="tight"
                            )

                test_user2relational_score["y_pred"] = test_user2relational_score["relational_score"].apply(
                    lambda rs: 1 if rs >= best_th else 0)
                y_true = test_user2relational_score["label"]
                y_pred = test_user2relational_score["y_pred"]
                with open(os.path.join(output_path, "relational_threshold_evaluation.txt"), "w") as fout:
                    fout.write(f"self_weight: {round(SELF_WEIGHT, 2)}\n")
                    fout.write(f"followers_weight: {round(FOLLOWERS_WEIGHT, 2)}\n")
                    fout.write(f"followees_weight: {round(FOLLOWEES_WEIGHT, 2)}\n")
                    fout.write(f"best_f1: {max_f1}\n")
                    fout.write(f"best_th: {best_th}\n")
                    fout.write("Test set:\n")
                    fout.write(f"F1-score: {f1_score(y_true, y_pred):.3f}\n")
                    fout.write(f"Precision: {precision_score(y_true, y_pred):.3f}\n")
                    fout.write(f"Recall: {recall_score(y_true, y_pred):.3f}\n")
                    fout.write(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}\n")
                    fout.write(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")

    res.to_csv(os.path.join(output_path, "relational_threshold_grid_search.csv"), index=False)


def calc_soft_threhold(hs_score, **kwargs):
    if hs_score < kwargs["LOWER_BOUND"]:
        th = kwargs["high_th"]
    elif kwargs["LOWER_BOUND"] <= hs_score < kwargs["HIGHER_BOUND"]:
        th = kwargs["medium_th"]
    else:
        th = kwargs["low_th"]
    return th


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
    avg_hs_score_per_user = user2pred.groupby('user_id').agg({"predictions": "mean"}).reset_index() \
        .rename(columns={"predictions": "avg_hs_score"})

    train_labeled_users, test_labeled_users = train_test_split(labeled_users, test_size=test_ratio,
                                                               stratify=labeled_users["label"],
                                                               random_state=random_state)  # stratify by label

    print(f'Train Percent HS Users: {train_labeled_users["label"].mean()}')
    print(f'Test Percent HS Users: {test_labeled_users["label"].mean()}')

    train_avg_hs_score_per_user_with_true = pd.merge(train_labeled_users, avg_hs_score_per_user, on='user_id')
    test_avg_hs_score_per_user_with_true = pd.merge(test_labeled_users, avg_hs_score_per_user, on='user_id')
    train_user2pred = user2pred[user2pred["user_id"].isin(list(train_labeled_users["user_id"]))].reset_index(drop=True)
    test_user2pred = user2pred[user2pred["user_id"].isin(list(test_labeled_users["user_id"]))].reset_index(drop=True)

    train_g_df = train_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
        columns={"predictions": "hs_count"})
    test_g_df = test_user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
        columns={"predictions": "hs_count"})

    # hs_count_per_user = user2pred.groupby('user_id').predictions.agg(get_hs_count).reset_index().rename(
    #     columns={"predictions": "hs_count"})

    res = pd.DataFrame(
        columns=["lower_bound", "higher_bound", "low_th", "medium_th", "high_th", "f1_score", "precision_score",
                 "recall_score", "accuracy_score"])
    min_low_th, min_mid_th, min_high_th = 1, 2, 3
    max_low_th, max_mid_th, max_high_th = 10, 50, train_g_df['hs_count'].max()
    for LOWER_BOUND in np.linspace(0.1, 0.4, 4):
        for HIGHER_BOUND in np.linspace(0.2, 0.6, 5):
            if LOWER_BOUND >= HIGHER_BOUND:
                continue
            for low_th in tqdm(range(min_low_th, max_low_th + 1, 1)):
                for medium_th in range(min_mid_th, max_mid_th + 1, 1):
                    if low_th >= medium_th:
                        continue
                    for high_th in range(min_high_th, max_high_th + 1, 1):
                        if low_th >= high_th or medium_th >= high_th:
                            continue
                        kwargs = {"LOWER_BOUND": LOWER_BOUND, "HIGHER_BOUND": HIGHER_BOUND,
                                  "low_th": low_th, "medium_th": medium_th, "high_th": high_th}
                        #                         avg_hs_score_per_user_with_true_copy = train_avg_hs_score_per_user_with_true.copy()
                        # train_avg_hs_score_per_user_with_true[
                        #     f"soft_threshold_{LOWER_BOUND}_{HIGHER_BOUND}_{low_th}_{medium_th}_{high_th}"] = \
                        #     train_avg_hs_score_per_user_with_true["avg_hs_score"]. \
                        #         apply(lambda avg_hs_score: calc_soft_threhold(avg_hs_score, **kwargs))
                        train_avg_hs_score_per_user_with_true = pd.concat(
                            [train_avg_hs_score_per_user_with_true,
                             train_avg_hs_score_per_user_with_true["avg_hs_score"]. \
                                 apply(lambda avg_hs_score: calc_soft_threhold(avg_hs_score, **kwargs)).rename(
                                 f"soft_threshold_{LOWER_BOUND}_{HIGHER_BOUND}_{low_th}_{medium_th}_{high_th}")],
                            axis=1)
                        test_avg_hs_score_per_user_with_true = pd.concat(
                            [test_avg_hs_score_per_user_with_true, test_avg_hs_score_per_user_with_true["avg_hs_score"]. \
                                apply(lambda avg_hs_score: calc_soft_threhold(avg_hs_score, **kwargs)).rename(
                                f"soft_threshold_{LOWER_BOUND}_{HIGHER_BOUND}_{low_th}_{medium_th}_{high_th}")], axis=1)
    bound_cols = [c for c in train_avg_hs_score_per_user_with_true.columns if 'soft' in c]
    y_preds_cols = [f"y_pred_{b_col}" for b_col in bound_cols]
    train_avg_hs_score_per_user_with_true = pd.merge(train_avg_hs_score_per_user_with_true, train_g_df,
                                                     on='user_id')
    test_avg_hs_score_per_user_with_true = pd.merge(test_avg_hs_score_per_user_with_true, test_g_df,
                                                    on='user_id')
    y_train_true = train_avg_hs_score_per_user_with_true["label"]
    y_test_true = test_avg_hs_score_per_user_with_true["label"]

    def apply_soft_th_pred(col, hs_count):
        return hs_count >= col

    train_avg_hs_score_per_user_with_true[y_preds_cols] = train_avg_hs_score_per_user_with_true[bound_cols]. \
        apply(lambda col: apply_soft_th_pred(col, train_avg_hs_score_per_user_with_true['hs_count']), axis=0)

    test_avg_hs_score_per_user_with_true[y_preds_cols] = test_avg_hs_score_per_user_with_true[bound_cols]. \
        apply(lambda col: apply_soft_th_pred(col, test_avg_hs_score_per_user_with_true['hs_count']), axis=0)

    best_f1 = 0.0
    for col in tqdm(bound_cols):
        current_bound = col.split("soft_threshold_")[1]
        train_avg_hs_score_per_user_with_true[f"y_pred_{current_bound}"] = train_avg_hs_score_per_user_with_true.apply(
            lambda row: 1 if row["hs_count"] >= row[col] else 0, axis=1)

        y_train_pred = train_avg_hs_score_per_user_with_true[f"y_pred_{current_bound}"]

        f1 = f1_score(y_train_true, y_train_pred)
        if f1 > best_f1:
            best_f1 = f1
            test_avg_hs_score_per_user_with_true[
                f"y_pred_{current_bound}"] = test_avg_hs_score_per_user_with_true.apply(
                lambda row: 1 if row["hs_count"] >= row[col] else 0, axis=1)

            y_test_pred = test_avg_hs_score_per_user_with_true[f"y_pred_{current_bound}"]
            with open(os.path.join(output_path, "dynamic_threshold_evaluation.txt"), "w") as fout:
                fout.write(f"lower_bound: {current_bound.split('_')[0]}\n")
                fout.write(f"higher_bound: {current_bound.split('_')[1]}\n")
                fout.write(f"low_th: {current_bound.split('_')[2]}\n")
                fout.write(f"medium_th: {current_bound.split('_')[3]}\n")
                fout.write(f"high_th: {current_bound.split('_')[4]}\n")
                fout.write(f"F1-score: {f1_score(y_test_true, y_test_pred):.3f}\n")
                fout.write(f"Precision: {precision_score(y_test_true, y_test_pred):.3f}\n")
                fout.write(f"Recall: {recall_score(y_test_true, y_test_pred):.3f}\n")
                fout.write(f"Accuracy: {accuracy_score(y_test_true, y_test_pred):.3f}\n")
                fout.write(f"Balanced Accuracy: {balanced_accuracy_score(y_test_true, y_test_pred):.3f}")

        precision = precision_score(y_train_true, y_train_pred)
        recall = recall_score(y_train_true, y_train_pred)
        accuracy = accuracy_score(y_train_true, y_train_pred)
        scb = current_bound.split('_')
        res = pd.concat([res, pd.DataFrame([
            {"lower_bound": scb[0], "higher_bound": scb[1], "low_th": scb[2], "medium_th": scb[3], "high_th": scb[4],
             "f1_score": f1, "precision_score": precision, "recall_score": recall,
             "accuracy_score": accuracy}])], ignore_index=True)

    res.to_csv(os.path.join(output_path, "soft_threshold.csv"), index=False)
    return res


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
    # seed = 2334642105 #42 #338761188
    print("Seed is:", seed)
    output_path = f"detection/outputs/{dataset}/{model_name}/user_level/grid_search/{seed}"
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists.")
    else:
        os.makedirs(output_path)
        print(f"Output path {output_path} created.")
    # rng = random.Random(seed)
    # dataset = 'Echo' if 'echo' in dataset else dataset
    fixed_threshold_num_of_posts(user2pred, labeled_users, output_path, dataset, test_ratio=0.2, random_state=seed)
    relational_threshold(user2pred, labeled_users, output_path, dataset, test_ratio=0.2, random_state=seed)
    dynamic_threshold_hs_score(user2pred, labeled_users, output_path, test_ratio=0.2, random_state=seed)


if __name__ == '__main__':
    run_simple_ulm_experiments()
