import sys
import os
f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, ".."))
from config.hate_networks_config import path_conf
from utils.my_timeit import timeit
import logging.config
from config.logger_config import logger_config
from hate_networks.topic_model import *
from hate_networks.word_embedding import *
from hate_networks.network import *
from utils.general import init_log
import warnings
warnings.filterwarnings('ignore')


# Full pipeline
def full_pipeline(train_network_type='echo', inference_network_type='echo', K=5, dist_topic_number=15, tm_model_type="lda", tm_vec_type="tfidf",
                  tm_feature_size=500, w2v_embedding_size=300, use_bert=False, w2v_arch="cbow", w2v_window=10,
                  w2v_min_count=5, edge_type="mention", min_edge_weight=3, layout_type="fr", with_labels=False):
    """
    Full pipeline for creating topic model and word2vec representations of users and then create a graph for each of
    the representations while coloring each user by its main topic or cluster.

    :param K: number of topics in topic model representation and number of clusters in word2vec representation.
    :param tm_feature_size: number of features (words) to consider in the topic model.
    :param w2v_embedding_size: size of the word2vec embedding.
    :param edge_type: The type of the edges between the users in the graph to use.
    May be one of the following: {"mentions", "retweets", "replies", "quotes"}.
    :param min_edge_weight: minimum edge weight to consider edges between users.
    :param layout_type: visual layout type of the igraph's graph.
    :return: None
    """
    raw_data_path = path_conf[inference_network_type]["raw_data"]
    if train_network_type is None:
        train_base_output_path = None
    else:
        train_base_output_path = path_conf[train_network_type]["base_output"]
    inference_base_output_path = path_conf[inference_network_type]["base_output"]
    create_dir_if_missing(inference_base_output_path)

    tsv_data_path = os.path.join(inference_base_output_path, "tsv_data")
    create_dir_if_missing(tsv_data_path)
    pickled_data_path = os.path.join(inference_base_output_path, "pickled_data")
    create_dir_if_missing(pickled_data_path)

    users_path = os.path.join(tsv_data_path, "users.tsv")
    ignore_punct = general_conf["ignore_punct"]
    ignore_rt = general_conf["ignore_retweets"]
    # TOPIC MODEL (MAX TOPIC)
    if not os.path.exists(users_path):  # extracting users data for the first time
        extract_data_from_tweets(data_path=raw_data_path, ignore_rt=ignore_rt, ignore_punct=ignore_punct, base_path=inference_base_output_path)
    users_df = pd.read_csv(users_path, sep='\t')
    # users_df = pd.read_csv("./hate_networks/Echo networks/permanent_users_df.csv")
    # users_df['user_id'] = users_df['user_id'].astype(str)
    corpora_path = os.path.join(inference_base_output_path, 'pickled_data', 'corpora_list.pkl')
    with open(corpora_path, 'rb') as f:
        tweets_corpora = pickle.load(f)

    # run_topic_model(tweets_corpora=tweets_corpora, users_df=users_df, topic_num=K, feature_num=tm_feature_size, train_path=train_base_output_path, inference_path=inference_base_output_path)
    # run again for the clustering (possibly with different topic number)
    # run_topic_model(tweets_corpora=tweets_corpora, users_df=users_df, topic_num=dist_topic_number, feature_num=tm_feature_size, train_path=train_base_output_path, inference_path=inference_base_output_path)
    # WORD2VEC

    run_word_embedding(tweets_corpora, users_df, w2v_arch, w2v_embedding_size, use_bert, w2v_window, w2v_min_count, base_path=inference_base_output_path)

    if use_bert:
        w2v_model_name = "bert"
    else:
        w2v_model_name = f"{w2v_arch}_{str(w2v_embedding_size)}_size_{str(w2v_window)}_window_{str(w2v_min_count)}_min_count"
    current_user_embeddings_path = os.path.join(inference_base_output_path, "word2vec", "user_embeddings", w2v_model_name)

    users_embedding_df = pd.read_csv(current_user_embeddings_path + ".tsv", sep='\t')


    w2v_path = os.path.join(inference_base_output_path, "word2vec")
    # create clustering based on the embedding of each user
    w2v_cluster_users(users_embedding_df, w2v_model_name, w2v_embedding_size, K, w2v_path)
    user_clusters_file_name = os.path.join(w2v_path, "user_clusters", f"{w2v_model_name}_{str(K)}_clusters.tsv")
    w2v_cluster_df = pd.read_csv(user_clusters_file_name, sep='\t')
    current_tm_user_distributions = os.path.join(inference_base_output_path, "topic_model", tm_vec_type, "user_distributions",
                                                 f"{tm_model_type}_{str(tm_feature_size)}_features_{str(K)}_topics.tsv")
    tm_topic_df = pd.read_csv(current_tm_user_distributions, sep='\t')[['user_id', 'user_max_topic']]  # take only max topic

    # CLUSTERING ON TOPIC MODEL DISTRIBUTION
    # 50 topics as default
    tm_cluster_users(dist_topic_number=dist_topic_number, K=K, model_type=tm_model_type,
                                        vec_type=tm_vec_type, feature_size=1000, base_path=inference_base_output_path)

    tm_clustering_path = os.path.join(inference_base_output_path, "topic_model", "tm_clustering")
    current_tm_clustering_file_name = os.path.join(tm_clustering_path, f"{tm_model_type}_{tm_vec_type}_vec_{str(tm_feature_size)}_features_"
                                                           f"{str(dist_topic_number)}_topics_{str(K)}_clusters.tsv")
    tm_clustering_df = pd.read_csv(current_tm_clustering_file_name, sep='\t')

    # merge tm dist and w2v clustering
    merged_users_topic_cluster_df = pd.merge(left=tm_topic_df, right=w2v_cluster_df, on="user_id")
    # todo: omit this part if already created with this config
    # merged_users_topic_cluster_df = get_overlapped_clusters(merged_users_topic_cluster_df, "user_max_topic", "w2v_cluster", K)

    merged_clusters_path = os.path.join(inference_base_output_path, "merged_user_clusters")
    create_dir_if_missing(merged_clusters_path)
    current_merged_clusters_file_name_1 = os.path.join(merged_clusters_path, f"{str(K)}_clusters__tm_max_topic_{tm_model_type}_"
                                                                             f"{tm_vec_type}__w2v_{w2v_embedding_size}_size.tsv")
    merged_users_topic_cluster_df.to_csv(current_merged_clusters_file_name_1, sep='\t', index=False)
    # merge tm dist and tm clustering
    merged_users_topic_cluster_df2 = pd.merge(left=tm_topic_df, right=tm_clustering_df, on="user_id")
    # merged_users_topic_cluster_df2 = get_overlapped_clusters(merged_users_topic_cluster_df2, "user_max_topic", "tm_dist_cluster", K)
    current_merged_clusters_file_name_2 = os.path.join(merged_clusters_path, f"{str(K)}_clusters__tm_max_topic_{tm_model_type}_"
                                                                             f"{tm_vec_type}__tm_dist_{dist_topic_number}_topic_num.tsv")
    merged_users_topic_cluster_df2.to_csv(current_merged_clusters_file_name_2, sep='\t', index=False)
    # merge w2v clustering and tm clustering
    # merged_users_topic_cluster_df3 = pd.merge(left=w2v_cluster_df, right=tm_clustering_df, on="user_id")
    # merged_users_topic_cluster_df3 = get_overlapped_clusters(merged_users_topic_cluster_df3, "w2v_cluster", "tm_kmeans_cluster", K)
    # merged_df_path3 = "hate_networks/Echo networks/merged_users_clusters/w2v_clustering_tm_clustering/" + str(K) + "_clusters_" + tm_model_type + \
    #                   "_" + tm_vec_type + "_tm_" + str(w2v_embedding_size) + "_w2v_size.csv"
    # merged_users_topic_cluster_df3.to_csv(merged_df_path3, index=False)

    # all_merged_df_path="hate_networks/Echo networks/merged_users_clusters/" + str(K) + "_clusters_" + tm_model_type + \
    #                   "_" + tm_vec_type + "_tm_" + str(w2v_embedding_size) + "_w2v_size.csv"


    # all_merged_df = pd.merge(left=merged_users_topic_cluster_df, right=merged_users_topic_cluster_df2, on="user_id", suffixes=("","__right"))
    all_merged_df = pd.merge(left=merged_users_topic_cluster_df, right=tm_clustering_df, on="user_id")
    # all_merged_df.to_csv(all_merged_df_path, index=False)

    output_fig_names = {"tm_max_topic": f"{str(K)}_clusters_{tm_model_type}_{tm_vec_type}_vec_{str(tm_feature_size)}_features",
                        "w2v": f"{str(K)}_clusters_{w2v_model_name}",
                        "tm_dist_cluster": f"{str(K)}_clusters_{tm_model_type}_{tm_vec_type}_vec_{str(tm_feature_size)}_features_"
                                                           f"{str(dist_topic_number)}_topics",
                        "pred": f"predicted_labels"}
    create_network(edge_type, all_merged_df, K,
                   min_edge_weight=min_edge_weight, layout_type=layout_type, with_labels=with_labels,
                   output_fig_names=output_fig_names, base_path=inference_base_output_path)
@timeit
def main():
    logger = init_log("hate_networks")
    train_network_type = general_conf["train_network_type"]
    inference_network_type = general_conf["inference_network_type"]
    logger.info(f"Started hate network flow on {inference_network_type} data using models trained on {train_network_type} data -- {datetime.today().strftime('%Y-%m-%d')}")
    logger.info("~" * 101)

    K = general_conf["K"]
    tm_type = general_conf["tm_type"]
    vec_type = general_conf["vec_type"]
    dist_topic_number = general_conf["dist_topic_number"]
    tm_feature_size = general_conf["tm_feature_size"]

    w2v_embedding_size = general_conf["w2v_embedding_size"]
    use_bert = general_conf["use_bert"]
    w2v_arch = general_conf["w2v_arch"]
    w2v_window = general_conf["w2v_window"]
    w2v_min_count = general_conf["w2v_min_count"]

    edge_type = general_conf["edge_type"]
    min_edge_weight = general_conf["min_edge_weight"]
    layout_type = general_conf["layout_type"]
    with_labels = general_conf["with_labels"]

    full_pipeline(train_network_type=train_network_type, inference_network_type=inference_network_type,
                  K=K, dist_topic_number=dist_topic_number, tm_model_type=tm_type, tm_vec_type=vec_type,
                  tm_feature_size=tm_feature_size, w2v_embedding_size=w2v_embedding_size, use_bert=use_bert,
                  w2v_arch=w2v_arch, w2v_window=w2v_window, w2v_min_count=w2v_min_count, edge_type=edge_type,
                  min_edge_weight=min_edge_weight, layout_type=layout_type, with_labels=with_labels)

if __name__ == '__main__':
    main()