import numpy as np
from scipy.special import comb
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from sklearn.cluster import KMeans
import math
class ModelValidation:
    def __init__(self, true_labels, pred_labels):
        self.true_labels = true_labels
        self.pred_labels = pred_labels

    def rand_index_score(self):
        tp_plus_fp = comb(np.bincount(self.pred_labels), 2).sum()
        tp_plus_fn = comb(np.bincount(self.true_labels), 2).sum()
        A = np.c_[(self.pred_labels, self.true_labels)]
        tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
                 for i in set(self.pred_labels))
        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp
        tn = comb(len(A), 2) - tp - fp - fn
        return (tp + tn) / (tp + fp + fn + tn)

    def adjusted_rand_index_score(self):
        """
        Calculates adjusted rand index score for the clustering
        :param labels_pred: Cluster labels to evaluate
        :param labels_true: Ground truth class labels to be used as a reference
        :return: ari (float) - Similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.

        """
        return adjusted_rand_score(self.true_labels, self.pred_labels)

    def kappa_score(self):
        return cohen_kappa_score(self.true_labels, self.pred_labels)

def get_annotation_scores(gold_labels, cluster_labels):
    model_validation = ModelValidation(gold_labels, cluster_labels)
    return model_validation.rand_index_score(), model_validation.adjusted_rand_index_score(), model_validation.kappa_score()

def cross_methods_ri_kappa(K=2):
    result_df_ri = pd.DataFrame(columns=["tm_max_topic", "tm_clustering", "w2v"], index=["tm_max_topic", "tm_clustering", "w2v"])
    result_df_kappa = pd.DataFrame(columns=["tm_max_topic", "tm_clustering", "w2v"], index=["tm_max_topic", "tm_clustering", "w2v"])

    w2v_model_name = "word2vec_" + str(300) + "size_" + "cbow" + "_" + str(10) + "window_" + \
                     str(5) + "min_count"
    users_embedding_df = pd.read_csv(
        "hate_networks/Echo networks/Word2Vec/Users embeddings/users_embedding_" + w2v_model_name + ".csv")
    all_embeddings = []
    users_embedding_df = users_embedding_df[users_embedding_df["embedding"] != "[]"]  # remove empty embeddings
    users_embedding_df[["embedding"]] = users_embedding_df[["embedding"]].applymap(literal_eval)

    for index, row in users_embedding_df.iterrows():
        current_emb = row["embedding"]
        all_embeddings.append(current_emb)
    X = np.array(all_embeddings)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    # users_embedding_df['w2v_cluster'] = kmeans.labels_
    users_embedding_df.loc[:, "w2v_cluster"] = kmeans.labels_
    # print(users_embedding_df)
    w2v_luster_df = users_embedding_df[['user_id', 'w2v_cluster']]
    topic_size = 10

    users_tm_dist_df = pd.read_csv(
        "hate_networks/Echo networks/Topic model/{0}/User distributions/topic_model_{1}_{2}_{3}.csv".format("tfidf",
                                                                                              "nmf",
                                                                                              "1000",
                                                                                              "10"))
    all_embeddings = []
    users_tm_dist_df.dropna(how="any", inplace=True)
    # keep only users with true labels
    for index, row in users_tm_dist_df.iterrows():
        current_emb = []
        for topic_num in range(topic_size):
            current_emb.append(row[str(topic_num)])
        if math.isnan(current_emb[0]):
            continue
        else:
            all_embeddings.append(current_emb)
    X = np.array(all_embeddings)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    users_tm_dist_df = users_tm_dist_df.assign(tm_kmeans_cluster=kmeans.labels_)
    cluster_df = users_tm_dist_df[['user_id', 'tm_kmeans_cluster']]
    users_clusters = pd.merge(w2v_luster_df, cluster_df, on="user_id")
    users_tm_dist_df = pd.read_csv(
        "hate_networks/Echo networks/Topic model/{0}/User distributions/topic_model_{1}_{2}_{3}.csv".format("tfidf",
                                                                                              "nmf",
                                                                                              "1000",
                                                                                              str(K)))
    users_tm_dist_df.dropna(how="any", inplace=True)
    max_topic_cluster_df = users_tm_dist_df[['user_id', 'user_topic']]
    users_clusters = pd.merge(max_topic_cluster_df, users_clusters, on="user_id")
    for cluster_0 in ["user_topic", "tm_kmeans_cluster", "w2v_cluster"]:
        for cluster_1 in ["user_topic", "tm_kmeans_cluster", "w2v_cluster"]:

            RI, ARI, kappa_score = get_annotation_scores([int(x) for x in list(users_clusters[cluster_0])],
                                                  [int(x) for x in list(users_clusters[cluster_1])])

            if cluster_0 == "user_topic":
                cluster0 = "tm_max_topic"
            elif cluster_0 == "tm_kmeans_cluster":
                cluster0 = "tm_clustering"
            else:
                cluster0 = "w2v"
            if cluster_1 == "user_topic":
                cluster1 = "tm_max_topic"
            elif cluster_1 == "tm_kmeans_cluster":
                cluster1 = "tm_clustering"
            else:
                cluster1 = "w2v"
            result_df_ri.loc[cluster0, cluster1] = RI
            result_df_kappa.loc[cluster0, cluster1] = kappa_score

            print(f"{cluster0} and {cluster1} has rand index of {RI},adjusted rand index of {ARI} and kappa agreement of {kappa_score}")

    result_df_ri.to_csv(f"./hate_networks/Echo networks/results/k_{K}_ri_cm.csv", index=True)
    result_df_kappa.to_csv(f"./hate_networks/Echo networks/results/k_{K}_kappa_cm.csv", index=True)

def tm_max_topic_rand_index(users_with_labels_df):
    labeled_users = list(users_with_labels_df["user_id"])
    ri_df = pd.DataFrame(columns=["tm_model_type", "tm_vec_type", "topic_size", "tm_feature_size", "ri", "ari"])
    for tm_model_type in ["nmf", "lda"]:
        for tm_vec_type in ["tfidf", "cv"]:
            for topic_size in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100]:
                for tm_feature_size in [500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]:
                    try:
                        users_tm_dist_df = pd.read_csv(
                            "hate_networks/Echo networks/Topic model/{0}/User distributions/topic_model_{1}_{2}_{3}.csv".format(tm_vec_type,
                                                                                                                  tm_model_type,
                                                                                                                  tm_feature_size,
                                                                                                                  topic_size))
                        users_tm_dist_df.dropna(how="any", inplace=True)
                        # keep only users with true labels
                        users_tm_dist_df = users_tm_dist_df[users_tm_dist_df["user_id"].apply(lambda ui: ui in labeled_users)]
                        cluster_df = users_tm_dist_df[['user_id', 'user_topic']]
                        users_with_labels_df_copy = users_with_labels_df.copy()
                        users_with_labels_df_copy = pd.merge(users_with_labels_df_copy, cluster_df, on="user_id")

                        tm_ri, tm_ari = get_annotation_scores(list(users_with_labels_df_copy["label"]), [int(x) for x in list(users_with_labels_df_copy["user_topic"])])
                        ri_df = ri_df.append({"tm_model_type": tm_model_type, "tm_vec_type": tm_vec_type,
                                              "topic_size": topic_size, "tm_feature_size": tm_feature_size,
                                              "ri": tm_ri, "ari": tm_ari}, ignore_index=True)

                        print("Topic model clustering has rand index of {0} and adjusted rand index of {1}".format(tm_ri, tm_ari))
                    except:
                        continue
    ri_df.to_csv("./hate_networks/Echo networks/results/tm_max_topic_ri_scores.csv", index=False)

def tm_clustering_rand_index(users_with_labels_df):
    labeled_users = list(users_with_labels_df["user_id"])
    ri_df = pd.DataFrame(columns=["tm_model_type", "tm_vec_type", "topic_size", "tm_feature_size", "cluster_size", "ri", "ari"])
    for tm_model_type in ["nmf", "lda"]:
        for tm_vec_type in ["tfidf", "cv"]:
            for topic_size in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100]:
                for tm_feature_size in [500, 1000, 1500, 2000, 3000, 5000, 7000, 10000]:
                    try:
                        users_tm_dist_df = pd.read_csv(
                            "hate_networks/Echo networks/Topic model/{0}/User distributions/topic_model_{1}_{2}_{3}.csv".format(tm_vec_type,
                                                                                                                  tm_model_type,
                                                                                                                  tm_feature_size,
                                                                                                                  topic_size))
                        all_embeddings = []
                        users_tm_dist_df.dropna(how="any", inplace=True)
                        # keep only users with true labels
                        users_tm_dist_df = users_tm_dist_df[users_tm_dist_df["user_id"].apply(lambda ui: ui in labeled_users)]
                        for index, row in users_tm_dist_df.iterrows():
                            current_emb = []
                            for topic_num in range(topic_size):
                                current_emb.append(row[str(topic_num)])
                            if math.isnan(current_emb[0]):
                                continue
                            else:
                                all_embeddings.append(current_emb)
                        X = np.array(all_embeddings)
                        # normalize X
                        # normalized_X = stats.zscore(X)
                        for K in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100]:
                            kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
                            users_tm_dist_df = users_tm_dist_df.assign(tm_kmeans_cluster=kmeans.labels_)
                            cluster_df = users_tm_dist_df[['user_id', 'tm_kmeans_cluster']]
                            users_with_labels_df_copy = users_with_labels_df.copy()
                            users_with_labels_df_copy = pd.merge(users_with_labels_df_copy, cluster_df, on="user_id")

                            tm_ri, tm_ari = get_annotation_scores(list(users_with_labels_df_copy["label"]), list(users_with_labels_df_copy["tm_kmeans_cluster"]))
                            ri_df = ri_df.append({"tm_model_type": tm_model_type, "tm_vec_type": tm_vec_type,
                                                  "topic_size": topic_size, "tm_feature_size": tm_feature_size,
                                                  "cluster_size": K, "ri": tm_ri, "ari": tm_ari}, ignore_index=True)

                            print("Topic model clustering has rand index of {0} and adjusted rand index of {1}".format(tm_ri, tm_ari))
                    except:
                        continue
    ri_df.to_csv("./hate_networks/Echo networks/results/tm_ri_scores.csv", index=False)

def word2vec_clustering_rand_index(users_with_labels_df):
    labeled_users = list(users_with_labels_df["user_id"])
    ri_df = pd.DataFrame(columns=["clusters_size", "w2v_arch", "embedding_size", "ri", "ari"])
    for w2v_arch in ["cbow", "skipgram"]:
        for embedding_size in [20,50,100,150,200,250,300,350,400]:
            w2v_model_name = "word2vec_" + str(embedding_size) + "size_" + w2v_arch + "_" + str(10) + "window_" + \
                             str(5) + "min_count"
            users_embedding_df = pd.read_csv(
                "hate_networks/Echo networks/Word2Vec/Users embeddings/users_embedding_" + w2v_model_name + ".csv")
            all_embeddings = []
            users_embedding_df = users_embedding_df[users_embedding_df["user_id"].apply(lambda ui: ui in labeled_users)]
            users_embedding_df = users_embedding_df[users_embedding_df["embedding"] != "[]"]  # remove empty embeddings
            users_embedding_df[["embedding"]] = users_embedding_df[["embedding"]].applymap(literal_eval)

            for index, row in users_embedding_df.iterrows():
                current_emb = row["embedding"]
                all_embeddings.append(current_emb)
            X = np.array(all_embeddings)
            for K in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100]:
                kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
                # users_embedding_df['w2v_cluster'] = kmeans.labels_
                users_embedding_df.loc[:, "w2v_cluster"] = kmeans.labels_
                # print(users_embedding_df)
                cluster_df = users_embedding_df[['user_id', 'w2v_cluster']]
                users_with_labels_df_copy = users_with_labels_df.copy()
                users_with_labels_df_copy = pd.merge(users_with_labels_df_copy, cluster_df, on="user_id")
                users_with_labels_df_copy.dropna(how="any", inplace=True)

                w2v_ri, w2v_ari = get_annotation_scores(list(users_with_labels_df_copy["label"]), list(users_with_labels_df_copy["w2v_cluster"]))
                ri_df = ri_df.append({"clusters_size": K, "w2v_arch": w2v_arch, "embedding_size": embedding_size, "ri": w2v_ri, "ari": w2v_ari}, ignore_index=True)
                print("Word2Vec clustering has rand index of {0} and adjusted rand index of {1}".format(w2v_ri, w2v_ari))
    ri_df.to_csv("./hate_networks/Echo networks/results/word2vec_ri_scores.csv", index=False)
