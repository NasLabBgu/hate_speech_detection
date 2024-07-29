from datetime import datetime, date, timedelta

general_conf = {
    "ignore_retweets": True,
    "only_english": True,
    "processes_number": 30,
    # "data_to_process": "covid",   # not in use, sent as a flag when running the script
    # possible values: 'covid', 'antisemitism', 'racial_slurs', 'all_datasets'
    "dataset_type": "twitter",
}

trending_topic_conf = {
    "p_num": 15,  # number of processes for parallel execution
    "latest_date": datetime.today() - timedelta(days=1),  # datetime(2020, 6, 4),
    # "chunk_size": 5,  # number of days to consider in one chunk of data
    "chunks_back_num": 3,  # number of chunks to consider in total (including the last chunk)
    "window_slide_size": 1,  # size of rolling window to move to the next chunk
    # "unigram_threshold": 200,  # min unigram threshold for topic count
    # "bigram_threshold": 300,  # min bigram threshold for topic count
    # "emoji_threshold": 50,
    "factor": 3,  # the relative growth of the topic's popularity
    "factor_power": 0.1,
    "user_limit": False,  # to consider specific users' tweets
    "ignore_retweets": True,  # whether to ignore retweets along finding the relevant trending unigrams/bigrmas
    "ignore_punct": True,
    "only_english": True,
    "topn_to_save": 50,
}

network_output_dir = "/sise/home/tommarz/hate_speech_detection/data/networks_data"

path_confs = {
    "covid": {
        "root_path": "/sise/Yalla_work/data/covid/",
        "raw_data": "/sise/Yalla_work/data/covid/data/",
        "pickled_data": "/sise/Yalla_work/data/covid/processed_data/pickled_data/",
        "models": "/sise/Yalla_work/data/covid/processed_data/models/",
        "output_trending_topic_dir": f"/sise/Yalla_work/data/covid/trending_topics/",
        "output_trending_topic_fn": f"{trending_topic_conf['chunks_back_num']}ChunksBack_"
        f"XXXXXChunkSize_"
        f"{trending_topic_conf['latest_date'].strftime('%Y-%m-%d')}LastDate.tsv",
        "ts": "/sise/Yalla_work/data/covid/ts/",
    },
    "antisemitism": {
        "root_path": "/sise/Yalla_work/data/hate_speech/antisemitism/",
        "raw_data": "/sise/Yalla_work/data/hate_speech/antisemitism/data/",
        "pickled_data": "/sise/Yalla_work/data/hate_speech/antisemitism/processed_data/pickled_data/",
        "models": "/sise/Yalla_work/data/hate_speech/antisemitism/processed_data/models/",
        "output_trending_topic_dir": f"/sise/Yalla_work/data/hate_speech/antisemitism/trending_topics/",
        "output_trending_topic_fn": f"{trending_topic_conf['chunks_back_num']}ChunksBack_"
        f"XXXXXChunkSize_"
        f"{trending_topic_conf['latest_date'].strftime('%Y-%m-%d')}LastDate.tsv",
        "ts": "/sise/Yalla_work/data/hate_speech/antisemitism/ts/",
    },
    "racial_slurs": {
        "root_path": "/sise/Yalla_work/data/hate_speech/racial_slurs/",
        "raw_data": "/sise/Yalla_work/data/hate_speech/racial_slurs/data/",
        "pickled_data": "/sise/Yalla_work/data/hate_speech/racial_slurs/processed_data/pickled_data/",
        "models": "/sise/Yalla_work/data/hate_speech/racial_slurs/processed_data/models/",
        "output_trending_topic_dir": f"/sise/Yalla_work/data/hate_speech/racial_slurs/trending_topics/",
        "output_trending_topic_fn": f"{trending_topic_conf['chunks_back_num']}ChunksBack_"
        f"XXXXXChunkSize_"
        f"{trending_topic_conf['latest_date'].strftime('%Y-%m-%d')}LastDate.tsv",
        "ts": "/sise/Yalla_work/data/hate_speech/racial_slurs/ts/",
    },
    "all_datasets": {
        "root_path": "/sise/Yalla_work/data/hate_speech/all_datasets/",
        "pickled_data": "/sise/Yalla_work/data/hate_speech/all_datasets/pickled_data/",
        "models": "/sise/Yalla_work/data/hate_speech/all_datasets/models/",
        "output_trending_topic_dir": f"/sise/Yalla_work/data/hate_speech/all_datasets/trending_topics/",
        "output_trending_topic_fn": f"{trending_topic_conf['chunks_back_num']}ChunksBack_"
        f"XXXXXChunkSize_"
        f"{trending_topic_conf['latest_date'].strftime('%Y-%m-%d')}LastDate.tsv",
        "ts": "/sise/Yalla_work/data/hate_speech/all_datasets/ts/",
    },
    "gab": {
        "root_path": "/sise/Yalla_work/data/hate_speech/gab/",
        "raw_data": "/sise/Yalla_work/data/hate_speech/gab/data/",
        "pickled_data": "/sise/Yalla_work/data/hate_speech/gab/processed_data/pickled_data/",
        "models": "/sise/Yalla_work/data/hate_speech/gab/processed_data/models/",
        "ts": "/sise/Yalla_work/data/hate_speech/gab/ts/",
        "output_trending_topic_dir": "/sise/Yalla_work/data/hate_speech/trending_topics/",
        "output_trending_topic_fn": f"{trending_topic_conf['chunks_back_num']}ChunksBack_"
        f"XXXXXChunkSize_"
        f"{trending_topic_conf['latest_date'].strftime('%Y-%m-%d')}LastDate.tsv",
        "posts": "/sise/home/tommarz/hate_speech_detection/detection/outputs/gab/BertFineTuning/user_level/all_users_tweets.parquet",
        "predictions": "/sise/home/tommarz/hate_speech_detection/detection/outputs/gab/BertFineTuning/user_level/split_by_posts/no_text/",
        "posts_with_predictions": "/sise/home/tommarz/hate_speech_detection/detection/outputs/gab/BertFineTuning/user_level/split_by_posts/with_text/",
        "reposts": "/sise/home/tommarz/hate_speech_detection/hate_networks/outputs/gab_networks/network_data/edges/retweet_edges_df.tsv",
        "doc_vectors": "/sise/home/tommarz/Hateful-users-detection/Dataset/GabData/Doc2Vec100.p",
        "raw_network": "/sise/home/tommarz/hate_speech_detection/data/networks_data/gab/raw_network.p",
        "largest_cc_network": "/sise/home/tommarz/hate_speech_detection/data/networks_data/gab/largest_cc.p"
    },
    "echo_2": {
        "reposts": "/sise/Yalla_work/data/echoes/only_english/dfs_and_dicts/el_echo_users_rt.txt",
        "posts": "/sise/home/tommarz/hate_speech_detection/detection/outputs/echo_2/BertFineTuning/user_level/all_users_tweets.parquet",
        "predictions": "/sise/home/tommarz/hate_speech_detection/detection/outputs/echo_2/BertFineTuning/user_level/split_by_posts/no_text/",
        "posts_with_predictions": "/sise/home/tommarz/hate_speech_detection/detection/outputs/echo_2/BertFineTuning/user_level/split_by_posts/with_text/",
        "doc_vectors": "/sise/home/tommarz/Hateful-users-detection/Dataset/EchoData/Doc2Vec100.p",
        "raw_network": "/sise/home/tommarz/hate_speech_detection/data/networks_data/echo_2/raw_network.p",
        "largest_cc_network": "/sise/home/tommarz/hate_speech_detection/data/networks_data/echo_2/largest_cc.p"
    },
    "parler": {
        "reposts": "/sise/Yalla_work/data/parler/echos_edge_dict.p",
        "posts": "/sise/home/tommarz/hate_speech_detection/detection/outputs/parler/BertFineTuning/user_level/all_users_tweets.parquet",
        "predictions": "/sise/home/tommarz/hate_speech_detection/detection/outputs/parler/BertFineTuning/user_level/split_by_posts/no_text/",
        "posts_with_predictions": "/sise/home/tommarz/hate_speech_detection/detection/outputs/parler/BertFineTuning/user_level/split_by_posts/with_text/",
        "doc_vectors": "/sise/home/tommarz/Hateful-users-detection/Dataset/ParlerData/Doc2Vec100.p",
        "raw_network": "/sise/home/tommarz/hate_speech_detection/data/networks_data/parler/raw_network.p",
        "largest_cc_network": "/sise/home/tommarz/hate_speech_detection/data/networks_data/parler/largest_cc.p"
    },
    "truth": {
        "reposts": "/sise/home/tommarz/truth_social/truth_social_retruths_weighted_edge_df.tsv"
    },
}

models_config = {
    "word_embedding": {
        "cbow": {"embedding_size": 300, "window_size": 11, "min_count": 3},
        "skipgram": {"embedding_size": 300, "window_size": 11, "min_count": 3},
        "fasttext": {
            "embedding_size": 300,
            "window_size": 11,
            "min_count": 3,
            "min_n": 3,  # character n-gram
            "max_n": 6,
        },
    }
}
