# config file to configure the execution of hate networks creation and coloring

general_conf = {
    "train_network_type": 'parler',  # echo, gab, parler, covid, antisemitism, racial_slurs
    "inference_network_type": 'parler',  # echo, gab, parler, covid, antisemitism, racial_slurs
    "ignore_punct": False,
    "ignore_retweets": False,

    "K": 2,

    # tm config
    "tm_type": "lda",  # topic model type (lda or nmf)
    "vec_type": "tfidf",  # vectorizer type to use in topic model (tfidf, cv)
    "dist_topic_number": 15,
    "tm_feature_size": 1000,

    # w2v config
    "w2v_embedding_size": 300,
    "use_bert": False,
    "w2v_arch": "cbow",
    "w2v_window": 10,
    "w2v_min_count": 5,

    # network config
    "edge_type": "mention",  # mention, retweet or in_reply_to
    "min_edge_weight": 3,
    "keep_all_edges": False,
    "keep_all_nodes": False,
    "plot_unsupervised_networks": False,
    "plot_supervised_networks": False,
    "extract_network_features": True,
    "layout_type": "fr",
    "with_labels": False,
    "network_file_type": 'pdf'
}

path_conf = {
    # the df under user_preds must be a df from a csv file in the format of of user_id, label
    "echo": {
        "raw_data": "/sise/Yalla_work/data/sharp_power/abusive_language/echoes/recent_history.zip",
        "base_output": "./hate_networks/outputs/echo_networks",
        "user_preds": "data/networks_data/echo_predicted_users.csv"

    },
    "covid": {
        "raw_data": "/sise/Yalla_work/data/covid/data/tweets_by_user/",
        "base_output": "./hate_networks/outputs/covid_networks"
    },
    "antisemitism": {
        "raw_data": "/sise/Yalla_work/data/hate_speech/antisemitism/tweets_by_user/",
        "base_output": "./hate_networks/outputs/antisemitism_networks"
    },
    "racial_slurs": {
        "raw_data": "/sise/Yalla_work/data/hate_speech/racial_slurs/tweets_by_user/",
        "base_output": "./hate_networks/outputs/racial_slurs_networks"
    },
    "gab": {
        "raw_data": "/sise/Yalla_work/data/hate_speech/gab/data/gab_posts_jan_2018.json",
        "followers_data": "/sise/Yalla_work/data/hate_speech/gab/data/followers_2018_02_01.json",
        "followings_data": "/sise/Yalla_work/data/hate_speech/gab/data/followings_2018_02_01.json",
        "base_output": "./hate_networks/outputs/gab_networks"
    },
    "parler": {
        "raw_data": "/sise/home/tommarz/parler-hate-speech/data/parler_sampled_posts.tsv",
        "mentions": "/sise/home/tommarz/parler-hate-speech/data/parler_sampled_users_comments_dict.p",
        "echos": "/sise/home/tommarz/parler-hate-speech/data/parler_sampled_users_echos_dict.p",
        # "followers_data": "/sise/Yalla_work/data/hate_speech/gab/data/followers_2018_02_01.json",
        # "followings_data": "/sise/Yalla_work/data/hate_speech/gab/data/followings_2018_02_01.json",
        "base_output": "./hate_networks/outputs/parler_networks"
    }
}