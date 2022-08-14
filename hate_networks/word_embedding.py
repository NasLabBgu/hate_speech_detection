# from bert_embedding import BertEmbedding
import gensim
import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
from statistics import mean
import multiprocessing as mp
import itertools
from sklearn.cluster import KMeans
from ast import literal_eval
from hate_networks.general import create_dir_if_missing

import logging
logger = logging.getLogger(__name__)

def run_word_embedding(tweets_corpora, users_df, arch, embedding_size, use_bert, window_size, min_count, base_path):
    pickled_data_path = os.path.join(base_path, "pickled_data")
    all_tweets_path = os.path.join(pickled_data_path, "all_tweets.pkl")

    sg = 0  # cbow
    if arch == "skipgram":
        sg = 1
    if use_bert:
        w2v_model_name = "bert"
    else:
        w2v_model_name = f"{arch}_{str(embedding_size)}_size_{str(window_size)}_window_{str(min_count)}_min_count"

    w2v_path = os.path.join(base_path, "word2vec")
    create_dir_if_missing(w2v_path)
    w2v_models_path = os.path.join(w2v_path, "models")
    create_dir_if_missing(w2v_models_path)
    current_model_dir = os.path.join(w2v_models_path, w2v_model_name)
    # word2vec model already exists with this configuration
    # if os.path.exists(current_model_dir):
    #     logger.info(f"{arch} word2vec model with configuration of {embedding_size} embedding size, {window_size} window"
    #                 f" size, and {min_count} min count already created")
    #     return

    # else:  # create new model
    #     if not use_bert:  # regular word2vec model
    #         with open(all_tweets_path, 'rb') as f:
    #             all_tweets = pickle.load(f)
    #         # create word2vec model
    #         create_word2vec_model(tweets=all_tweets, embedding_size=embedding_size, window_size=window_size,
    #                               min_count=min_count, sg=sg, model_dir_path=current_model_dir)
    #         # create embedding to each user using the word2vec model
    current_model_name = current_model_dir.split("/")[-1] + ".model"
    current_model_path = os.path.join(current_model_dir, current_model_name)
    create_users_embeddings(users_df, tweets_corpora, w2v_path, current_model_path)
        # else:  # use bert pre-trained model
        #     bert_embeddings(users_df, tweets_corpora)
# Word2Vec -> Clustering
def create_word2vec_model(tweets, embedding_size, window_size, min_count, sg, model_dir_path):
    """
    Create word2vec model given the parameters of the model
    cbow is default - sg=0,
    skipgram - sg=1

    :param tweets:
    :param embedding_size:
    :param window_size:
    :param min_count:
    :param sg:
    :param model_dir_path:
    :return:
    """
    if sg == 0:
        arch = "cbow"
    else:
        arch = "skipgram"
    create_dir_if_missing(model_dir_path)
    preprocessed_tweets = [gensim.utils.simple_preprocess(text, max_len=100) for text in tweets]
    logger.info(f"started word2vec training with {arch} architecute, {embedding_size} embedding size, {window_size} "
                f"window size and {min_count} min count.")
    model = gensim.models.Word2Vec(preprocessed_tweets, vector_size=embedding_size, window=window_size,
                                   min_count=min_count, workers=10, sg=sg)
    model.train(preprocessed_tweets, total_examples=len(preprocessed_tweets), epochs=10)
    print("finished training")
    # model.wv.save_word2vec_format('model.bin')
    # model.wv.save_word2vec_format('model.txt', binary=False)
    model_file_name = model_dir_path.split("/")[-1] + ".model"
    model.save(os.path.join(model_dir_path, model_file_name))

def create_multiple_word2vec_models(w2v_embedding_size=300, w2v_arch="cbow", w2v_window=10, w2v_min_count=5):
    users_df = pd.read_csv('hate_networks/Echo networks/csv_data/users.csv')
    with open('hate_networks/Echo networks/pickled_data/users_tweets_corpora_2.pkl', 'rb') as f:
        tweets_corpora = pickle.load(f)
    with open('hate_networks/Echo networks/pickled_data/all_tweets_list.pkl', 'rb') as f:
        all_tweets = pickle.load(f)
    sg = 0  # cbow
    if w2v_arch == "skipgram":
        sg = 1
    w2v_model_name = "word2vec_" + str(w2v_embedding_size) + "size_" + w2v_arch + "_" + str(w2v_window) + "window_" + \
                     str(w2v_min_count) + "min_count"
    # word2vec model already exists with this configuration
    if os.path.exists("hate_networks/Echo networks/Word2Vec/Models/" + w2v_model_name):
        print(
            "Word2Vec with configuration of {0} embedding size, and {1} window size, {2} min count and {3} architecture "
            "already created".format(w2v_embedding_size, w2v_window, w2v_min_count, w2v_arch))

        # users_embedding_df = create_users_embeddings(users_df, tweets_corpora, "hate_networks/Echo networks/Word2Vec/Models/" +
        #                                              w2v_model_name + "/" + w2v_model_name + ".model")
    else:
        # create word2vec model
        create_word2vec_model(tweets=all_tweets, embedding_size=w2v_embedding_size, window_size=w2v_window,
                              min_count=w2v_min_count, sg=sg,
                              model_dir_path="hate_networks/Echo networks/Word2Vec/Models/" + w2v_model_name)
    # create embedding to each user using the word2vec model
    word2vec_path = "hate_networks/Echo networks/Word2Vec/Models/" + w2v_model_name + "/" + w2v_model_name + ".model"
    if not os.path.exists("hate_networks/Echo networks/Word2Vec/Users embeddings/users_embedding_" +
                                              word2vec_path.split("/")[-1].split(".")[0] + ".csv"):
        create_users_embeddings(users_df, tweets_corpora, "hate_networks/Echo networks/Word2Vec/Models/" +
                                                     w2v_model_name + "/" + w2v_model_name + ".model")
    users_embedding_df = pd.read_csv(
        "hate_networks/Echo networks/Word2Vec/Users embeddings/users_embedding_" + w2v_model_name + ".csv")
    # create clustering based on the embedding of each user
    if not os.path.exists("hate_networks/Echo networks/Word2Vec/Users clusters/users_clusters_" + w2v_model_name + "_10clusters" + ".csv"):
        for K in [2,3,4,5,6,7,8,9,10]:
            w2v_cluster_users(users_embedding_df, w2v_model_name, w2v_embedding_size, K)

def parallelize_create_multiple_word2vec_models(embedding_sizes, w2v_archs=["cbow", "skipgram"], processors_num=50):
    pool = mp.Pool(processors_num)
    all_combinations = itertools.product(embedding_sizes, w2v_archs)
    [pool.apply_async(create_multiple_word2vec_models, args=(combination[0], combination[1])) for combination in all_combinations]
    pool.close()
    pool.join()

def bert_embeddings(users_df, users_corpora):
    # bc = BertClient()
    # bert_embedding = BertEmbedding()
    bert_embedding = BertEmbedding(max_seq_length=100)
    # ctx = mx.gpu(0)
    # bert_embedding = BertEmbedding(max_seq_length=100, ctx=ctx)
    users_embeddings_df = pd.DataFrame(columns=['user_id', 'embedding'])
    for i, corpus in enumerate(tqdm(users_corpora)):
        # text_embeddings = np.zeros((len(users_corpora, 768)))
        text_embeddings = []
        # bert_res = bert_embedding([corpus])
        splitted_corpus = corpus.replace("\n", "").split(".")[:50]
        # print(len(splitted_corpus))
        bert_res = bert_embedding(splitted_corpus)

        for emb_res in bert_res:
            for emb in emb_res[1]:
                text_embeddings.append(emb)
        tweets_mean_embedding = [mean(x) for x in zip(*text_embeddings)]
        users_embeddings_df.loc[i] = [users_df.iloc[i]['user_id'], tweets_mean_embedding]
    users_embeddings_df.to_csv("hate_networks/Echo networks/Word2Vec/Users embeddings/bert_users_embedding" + ".csv", index=False)
    return users_embeddings_df


def create_users_embeddings(users_df, users_corpora, w2v_base_path, word2vec_path):
    """
    Given a corpus per user and a word2vec model this function returns the mean embedding of each user
    :param users_corpora:
    :param word2vec_path:
    :return:
    """
    user_embeddings_path = os.path.join(w2v_base_path, "user_embeddings")
    create_dir_if_missing(user_embeddings_path)

    users_embeddings_df = pd.DataFrame(columns=['user_id', 'corpus', 'embedding'])
    # with open(path) as f:
    # abusive_tweets_json = simplejson.load(f)
    model = gensim.models.Word2Vec.load(word2vec_path)

    # get embeddings of tweets
    for i, corpus in enumerate(tqdm(users_corpora)):
        text_list = corpus.split()
        text_embeddings = defaultdict(lambda x: np.ndarray)
        vocab = list(model.wv.index_to_key)
        for word in text_list:
            if word in vocab:
                text_embeddings[word] = list(model.wv[word])
        # text_mat = []
        # for key, value in text_embeddings.items():
        # text_mat.append(value)
        text_mat = list(map(list, zip(*list(text_embeddings.values()))))
        tweet_mean_embedding = [np.mean([el for el in sublist]) for sublist in text_mat]
        tweet_mean_embedding = [x.item() for x in tweet_mean_embedding]
        users_embeddings_df.loc[i] = [users_df.iloc[i]['user_id'], corpus, tweet_mean_embedding]

    # write the dataframe with the corpus of each user
    # users_embeddings_df.to_csv("hate_networks/Echo networks/Word2Vec/Users embeddings/users_embedding_with_corpus_" +
    #                            word2vec_path.split("/")[-1].split(".")[0] + ".csv", index=False)
    # write the dataframe without the corpus
    users_embeddings_df_without_corpus = users_embeddings_df[["user_id", "embedding"]]
    user_embeddings_file_name = word2vec_path.split("/")[-1].split(".")[0] + ".tsv"
    current_user_embeddings_path = os.path.join(user_embeddings_path, user_embeddings_file_name)
    users_embeddings_df_without_corpus.to_csv(current_user_embeddings_path, sep='\t', index=False)

    # return users_embeddings_df_without_corpus


def w2v_cluster_users(users_embedding_df, w2v_model_name, embedding_size, K, w2v_base_path):
    """
    Given word2vec embedding per user on his document create KMeans clustering and return it as df
    :param users_embedding_df:
    :param w2v_model_name:
    :param K:
    :return:
    """
    user_clusters_path = os.path.join(w2v_base_path, "user_clusters")
    create_dir_if_missing(user_clusters_path)
    user_clusters_file_name = os.path.join(user_clusters_path, f"{w2v_model_name}_{str(K)}_clusters.tsv")
    if os.path.exists(user_clusters_file_name):
        logger.info(f"clustering on word2vec embeddings already created (model name: {w2v_model_name})")
        return
    all_embeddings = []
    # users_embedding_df = users_embedding_df.drop([2715,4201,4370], axis=0)
    no_embedding_users = 0
    users_embedding_df = users_embedding_df[users_embedding_df["embedding"] != "[]"]  # remove empty embeddings
    users_embedding_df[["embedding"]] = users_embedding_df[["embedding"]].applymap(literal_eval)

    for index, row in users_embedding_df.iterrows():
        # current_emb = row['embedding'].replace('[', '').replace(']', '').split(',')
        current_emb = row["embedding"]
        all_embeddings.append(current_emb)
        if len(current_emb) < embedding_size:
            no_embedding_users += 1
            print(index)
        # else:
        #   current_emb = [float(val) for val in current_emb]
        #   all_embeddings.append(current_emb)
    # print('no_embedding_users: %s' % no_embedding_users)
    X = np.array(all_embeddings)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    # users_embedding_df['w2v_cluster'] = kmeans.labels_
    users_embedding_df.loc[:, "w2v_cluster"] = kmeans.labels_
    # print(users_embedding_df)
    cluster_df = users_embedding_df[['user_id', 'w2v_cluster']]
    cluster_df.to_csv(user_clusters_file_name, sep='\t', index=False)



