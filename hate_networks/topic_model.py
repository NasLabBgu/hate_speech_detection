# from hate_networks.gensim_topic_models import gensim_topic_models
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
import multiprocessing as mp
import itertools
import pickle
import os
import math

from hate_networks.general import create_dir_if_missing
import logging
logger = logging.getLogger(__name__)

# Topic model
def display_topics(model, feature_names, no_top_words, file):
    '''
    Writes the topics with top words to a file
    :param model:  The topic model
    :param feature_names:  The words to write
    :param no_top_words: Number of top words in each topic to write
    :param file: File path to write in
    :return:
    '''
    for topic_idx, topic in enumerate(model.components_):
        # print("Topic %d:" % (topic_idx))
        # print(" ".join([feature_names[i]
        #                 for i in topic.argsort()[:-no_top_words - 1:-1]]))
        file.write("Topic %d:" % (topic_idx) + '\n')
        file.write(" ".join([feature_names[i]
                             for i in topic.argsort()[:-no_top_words - 1:-1]]) + '\n')

def create_topic_models(model_type='nmf', corpora=None, users=None, vec_type='tfidf', features_num=10000, topics_num=5,
                        top_words_num=50, train_path=None, dir_prefix=''):
    '''
    Creating a topic model
    :param model_type: type of topic model, currently sklearn's NMF or sklearn's LDA
    :param corpora: tweets per user list of corpus
    :param users: users df
    :param tfidf: use tfidf or only count
    :param features_num: max features number to use
    :param topics_num: number of topics to create the model with
    :param top_words_num: number of top words to display in each topic
    :return: model, processed_data, feature_extractor
    '''

    dir_prefix = os.path.join(dir_prefix, "tfidf") if vec_type=='tfidf' else  os.path.join(dir_prefix, "cv")
    create_dir_if_missing(dir_prefix)
    models_path = os.path.join(dir_prefix, "models")
    create_dir_if_missing(models_path)
    model = None
    feature_names = None
    feature_extractor = None
    processed_data = None
    filename = ''
    probs = None

    # if
    if vec_type=='tfidf':
        feature_extractor = TfidfVectorizer(max_df=0.95, min_df=2, lowercase=True,
                                            max_features=features_num, stop_words='english')
        vectorizer_path = os.path.join(models_path, f'vectorizer_{str(features_num)}.pkl')

    else:
        feature_extractor = CountVectorizer(max_df=0.95, min_df=2, lowercase=True,
                                            max_features=features_num, stop_words='english')
        vectorizer_path = os.path.join(models_path, f'vectorizer_{str(features_num)}.pkl')

    processed_data = feature_extractor.fit_transform(corpora)
    feature_names = feature_extractor.get_feature_names()

    if not os.path.exists(vectorizer_path):
        pickle.dump(feature_extractor, open(vectorizer_path, 'wb'))

    if model_type == 'nmf':
        # Run NMF
        model = NMF(n_components=topics_num, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(processed_data)
        transformed_nmf = model.transform(processed_data)
        probs = transformed_nmf / transformed_nmf.sum(axis=1, keepdims=True)

        tm_path = os.path.join(models_path,  f"nmf_model_{str(features_num)}_{str(topics_num)}.pkl")


    elif model_type == 'lda':
        # LDA can only use raw term counts for LDA because it is a probabilistic graphical model ?
        # Run LDA
        model = LatentDirichletAllocation(n_components=topics_num, max_iter=5, learning_method='online',
                                          learning_offset=50., random_state=0).fit(processed_data)
        transformed_lda = model.transform(processed_data)
        probs = transformed_lda / transformed_lda.sum(axis=1, keepdims=True)

        tm_path = os.path.join(models_path, f"lda_model_{str(features_num)}_{str(topics_num)}.pkl")

    elif model_type == 'gensim_lda':
        pass # todo: support gensim topic model as well
        # gensim_tm = gensim_topic_models()


    distribution = pd.DataFrame(data=probs[0:, 0:], columns=range(topics_num))
    distribution['user_max_topic'] = distribution.idxmax(axis=1)
    users_distribution = users.join(distribution)

    user_dist_path = os.path.join(dir_prefix, "user_distributions")
    create_dir_if_missing(user_dist_path)

    users_distribution.to_csv(os.path.join(user_dist_path, f"{model_type}_{str(features_num)}_features_"
                                                           f"{str(topics_num)}_topics.tsv"), sep='\t', index=False)
    top_words_path = os.path.join(dir_prefix, f"top_{top_words_num}_words_per_topic")
    create_dir_if_missing(top_words_path)

    with open(os.path.join(top_words_path,f"{model_type}_{str(features_num)}_feautres_{str(topics_num)}_topics__"
                                          f"{str(top_words_num)}_words.txt"), "w", encoding="utf-8") as file:
        file.write('topic number: ' + str(topics_num) + '\n')
        # print('Number of topics: %s' % (str(topics_num)))
        display_topics(model, feature_names, top_words_num, file)

    # save the model to disk
    pickle.dump(model, open(tm_path, 'wb'))
    return model, processed_data, feature_extractor, vectorizer_path

def run_topic_model(tweets_corpora, users_df, topic_num, feature_num, train_path, inference_path):
    """
    Given corpus per user and users df create topic model using NMF and LDA methods.
    :param tweets_corpora: corpora of all users in the data.
    :param users_df: DataFrame with all of the users' ids.
    :param topic_num: Number of topics to create the topic model with.
    :param feature_num: Number of features (words) to consider in the topic model.
    :return: None
    """
    logger.info(f"Started topic modeling with {topic_num} topics and {feature_num} features")
    sub_model_file_name = f"{feature_num}_{topic_num}"
    cv_dir_path = os.path.join(inference_path, "topic_model", "cv", "models")
    tfidf_dir_path = os.path.join(inference_path, "topic_model", "tfidf", "models")
    tm_path = os.path.join(inference_path, "topic_model")
    create_dir_if_missing(tm_path)
    if os.path.isdir(os.path.join(tm_path, "cv")) and \
        f"lda_model_{sub_model_file_name}.pkl" in os.listdir(cv_dir_path) and \
        f"nmf_model_{sub_model_file_name}.pkl" in os.listdir(cv_dir_path) and \
        f"lda_model_{sub_model_file_name}.pkl" in os.listdir(tfidf_dir_path) and \
        f"nmf_model_{sub_model_file_name}.pkl" in os.listdir(tfidf_dir_path):
        logger.info(f"Topic model with configuration of {topic_num} topics and {feature_num} features already created")
        return
    top_words_num = 50
    dir_prefix = os.path.join(inference_path, "topic_model")
    create_dir_if_missing(dir_prefix)

    for model_type in ['lda', 'nmf']:
        for vec_type in ['tfidf', 'cv']:
            # for feature_num in feature_nums:
            #   for topic_num in topic_numbers:
                create_topic_models(model_type=model_type, corpora=tweets_corpora, users=users_df, vec_type=vec_type,
                                    features_num=feature_num, topics_num=topic_num,
                                    top_words_num=top_words_num, train_path=train_path, dir_prefix=dir_prefix)

def parallelize_run_topic_model(tweets_corpora, users_df, topic_sizes, feature_sizes, processors_num=50):
    pool = mp.Pool(processors_num)
    all_combinations = itertools.product(topic_sizes, feature_sizes)
    [pool.apply_async(run_topic_model, args=(tweets_corpora, users_df, combination[0], combination[1])) for combination in all_combinations]
    pool.close()
    pool.join()




# Topic model -> Clustering
def tm_cluster_users(dist_topic_number, K, model_type, vec_type, feature_size, base_path):
    """
    Given a topic model's distribution of the users create Kmeans clustering
    :param users_tm_dist_df:
    :return:
    """
    current_user_distributions_path = os.path.join(base_path, "topic_model", vec_type, "user_distributions",
                        f"{model_type}_{str(feature_size)}_features_{str(dist_topic_number)}_topics.tsv")
    users_tm_dist_df = pd.read_csv(current_user_distributions_path, sep='\t')
    tm_clustering_path = os.path.join(base_path, "topic_model", "tm_clustering")
    create_dir_if_missing(tm_clustering_path)
    current_tm_clustering_file_name = os.path.join(tm_clustering_path, f"{model_type}_{vec_type}_vec_{str(feature_size)}_features_"
                                                           f"{str(dist_topic_number)}_topics_{str(K)}_clusters.tsv")
    if os.path.exists(current_tm_clustering_file_name):
        logger.info(f"already created clustering with the configuration: {current_tm_clustering_file_name}")
        return
    all_embeddings = []
    # users_embedding_df = users_embedding_df.drop([2715,4201,4370], axis=0)
    # users_embedding_df = users_embedding_df[users_embedding_df["embedding"] != "[]"]  # remove empty embeddings
    # users_embedding_df[["embedding"]] = users_embedding_df[["embedding"]].applymap(literal_eval)
    users_tm_dist_df.dropna(how="any", inplace=True)
    for index, row in users_tm_dist_df.iterrows():
        # current_emb = row['embedding'].replace('[', '').replace(']', '').split(',')
        current_emb = []
        for topic_num in range(dist_topic_number):
            current_emb.append(row[str(topic_num)])
        if math.isnan(current_emb[0]):
            logger.info(f"row {index} of user_id {row['user_id']} has no topic distribution.")
            continue
        else:
            all_embeddings.append(current_emb)
    X = np.array(all_embeddings)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    users_tm_dist_df = users_tm_dist_df.assign(tm_dist_cluster=kmeans.labels_)
    cluster_df = users_tm_dist_df[['user_id', 'tm_dist_cluster']]
    cluster_df.to_csv(current_tm_clustering_file_name, sep='\t', index=False)


def get_specific_clustering_tm():
    tm_model_type = "lda"
    tm_vec_type = "cv"
    topic_size = 30
    tm_feature_size = 500
    users_tm_dist_df = pd.read_csv(
        "hate_networks/Echo networks/Topic model/{0}/User distributions/topic_model_{1}_{2}_{3}.csv".format(
            tm_vec_type,
            tm_model_type,
            str(tm_feature_size),
            str(topic_size)))
    all_embeddings = []
    users_tm_dist_df.dropna(how="any", inplace=True)
    for index, row in users_tm_dist_df.iterrows():
        current_emb = []
        for topic_num in range(topic_size):
            current_emb.append(row[str(topic_num)])
        if math.isnan(current_emb[0]):
            continue
        else:
            all_embeddings.append(current_emb)
    X = np.array(all_embeddings)
    K = 3
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    users_tm_dist_df = users_tm_dist_df.assign(tm_kmeans_cluster=kmeans.labels_)
    cluster_df = users_tm_dist_df[['user_id', 'tm_kmeans_cluster']]
    cluster_df.to_csv("./hate_networks/Echo networks/Topic model/Kmeans/tm_kmeans_users_clusters_30_topics_lda_tm_type_cv_vec_type_3_clusters.csv", index=False)
