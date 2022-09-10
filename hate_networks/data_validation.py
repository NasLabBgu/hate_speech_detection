import os
import multiprocessing as mp
import gzip
import math
from scipy import spatial
import zipfile, re, io
from io import BytesIO
import numpy as np
import pandas as pd
import json, simplejson
import pickle
import string
from collections import defaultdict
from tqdm import tqdm


def create_user_dicts(original_echoes_path):
    original_echoes_files = os.listdir(original_echoes_path)
    echo_text_users = defaultdict(tuple)  # user_id_str: (user_name, user_screen_name, tweet_count, user_language, {tweet_lang:tweet_lang_count})
    echo_username_users = defaultdict(tuple)  # user_id_str: (user_name, user_screen_name, tweet_count, user_language, {tweet_lang:tweet_lang_count})
    tweet_lang_dict = defaultdict(set)
    user_lang_dict = defaultdict(set)
    user_in_text_tweets = defaultdict(set)
    user_in_username_tweets = defaultdict(set)
    for file in tqdm(original_echoes_files):
        with open(original_echoes_path + file) as f:
            if "USR" in file:
                for line in f:
                    line_as_dict = json.loads(line)
                    tweet_id_str = line_as_dict["id_str"]
                    user_id_str = line_as_dict["user"]["id_str"]
                    # user_name = line_as_dict["user"]["name"]
                    # user_screen_name = line_as_dict["user"]["screen_name"]
                    user_lang = line_as_dict["user"]["lang"]
                    # tweet_lang = line_as_dict.get("lang", None)
                    user_lang_dict[user_lang].add(user_id_str)
                    # if line_as_dict['truncated'] == False:
                    #     text = line_as_dict['full_text']
                    # else:
                    #     text = line_as_dict['extended_tweet']['full_text']
                    # text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', '',
                    #               text)  # remove urls
                    user_in_text_tweets[user_id_str].add(tweet_id_str)

                    # if tweet_lang:
                    #     tweet_lang_dict[tweet_lang].add(tweet_id_str)
                    # if user_id_str in echo_text_users.keys():
                    #     current_user_tweet_count = echo_text_users[user_id_str][2]
                    #     current_tweet_lang_dict = echo_text_users[user_id_str][4]
                    #     if tweet_lang:
                    #         if tweet_lang in current_tweet_lang_dict.keys():
                    #             current_tweet_lang_count = current_tweet_lang_dict[tweet_lang]
                    #             current_tweet_lang_dict[tweet_lang] = current_tweet_lang_count + 1
                    #         else:
                    #             current_tweet_lang_dict[tweet_lang] = 1
                    #     echo_text_users[user_id_str] = (user_name, user_screen_name, current_user_tweet_count + 1,
                    #                                   user_lang, current_tweet_lang_dict)
                    # else:
                    #     current_tweet_lang_dict = {}
                    #     current_tweet_lang_dict[tweet_lang] = 1
                    #     echo_text_users[user_id_str] = (user_name, user_screen_name, 1,
                    #                                   user_lang, current_tweet_lang_dict)
            elif "RESP" in file:
                for line in f:
                    line_as_dict = json.loads(line)
                    tweet_id_str = line_as_dict["id_str"]
                    user_id_str = line_as_dict["user"]["id_str"]
                    # user_name = line_as_dict["user"]["name"]
                    # user_screen_name = line_as_dict["user"]["screen_name"]
                    # user_lang = line_as_dict["user"]["lang"]
                    # tweet_lang = line_as_dict.get("lang", None)
                    # user_lang_dict[user_lang].add(user_id_str)
                    # if tweet_lang:
                    #     tweet_lang_dict[tweet_lang].add(tweet_id_str)
                    # if user_id_str in echo_username_users.keys():
                    #     current_user_tweet_count = echo_username_users[user_id_str][2]
                    #     current_tweet_lang_dict = echo_username_users[user_id_str][4]
                    #     if tweet_lang:
                    #         if tweet_lang in current_tweet_lang_dict.keys():
                    #             current_tweet_lang_count = current_tweet_lang_dict[tweet_lang]
                    #             current_tweet_lang_dict[tweet_lang] = current_tweet_lang_count + 1
                    #         else:
                    #             current_tweet_lang_dict[tweet_lang] = 1
                    #     echo_username_users[user_id_str] = (user_name, user_screen_name, current_user_tweet_count + 1,
                    #                                   user_lang, current_tweet_lang_dict)
                    # else:
                    #     current_tweet_lang_dict = {}
                    #     current_tweet_lang_dict[tweet_lang] = 1
                    #     echo_username_users[user_id_str] = (user_name, user_screen_name, 1,
                    #                                   user_lang, current_tweet_lang_dict)
                    # if line_as_dict['truncated'] == False:
                    #     text = line_as_dict['full_text']
                    # else:
                    #     text = line_as_dict['extended_tweet']['full_text']
                    # text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', '',
                    #               text)  # remove urls
                    user_in_username_tweets[user_id_str].add(tweet_id_str)

    # save dicts
    # with open("./Echo networks/dicts/original_echoes/echo_text_users.pkl", "wb") as fout:
    #     pickle.dump(echo_text_users, fout)
    # with open("./Echo networks/dicts/original_echoes/echo_username_users.pkl", "wb") as fout:
    #     pickle.dump(echo_username_users, fout)
    # with open("./Echo networks/dicts/original_echoes/user_lang_dict.pkl", "wb") as fout:
    #     pickle.dump(user_lang_dict, fout)
    # with open("./Echo networks/dicts/original_echoes/tweet_lang_dict.pkl", "wb") as fout:
    #     pickle.dump(tweet_lang_dict, fout)
    with open("./Echo networks/dicts/original_echoes/user_in_text_tweets.pkl", "wb") as fout:
        pickle.dump(user_in_text_tweets, fout)
    with open("./Echo networks/dicts/original_echoes/user_in_username_tweets.pkl", "wb") as fout:
        pickle.dump(user_in_username_tweets, fout)


def distinct_tweets_and_users_languages(path=None):
    '''
    A function that given a path to a csv file with tweets returns the corpora which includes tweets of each user
    :param path: csv file path containing tweets
    :return: corpora of each user
    '''
    tweet_lang_dict = defaultdict(set)
    user_lang_dict = defaultdict(set)
    retweets_tweet_lang_dict = defaultdict(set)
    retweets_user_lang_dict = defaultdict(set)
    tweets_set_per_user = defaultdict(set)
    retweet_count = 0
    not_retweet_count = 0
    with zipfile.ZipFile(path, "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        line_as_dict = json.loads(line)
                        # json = simplejson.loads(line)
                        tweet_lang = line_as_dict.get("lang", None)
                        user_lang = line_as_dict["user"]["lang"]
                        tweet_str_id = line_as_dict["id_str"]
                        user_str_id = line_as_dict["user"]["id_str"]
                        if "retweeted_status" in line_as_dict.keys():  # is a retweet
                            retweet_count += 1
                        #     original_tweet_lang = line_as_dict["retweeted_status"]["lang"]
                        #     original_user_lang = line_as_dict["retweeted_status"]["user"]["lang"]
                        #     original_tweet_str_id = line_as_dict["retweeted_status"]["id_str"]
                        #     original_user_str_id = line_as_dict["retweeted_status"]["user"]["id_str"]
                        #     retweets_tweet_lang_dict[original_tweet_lang].add(original_tweet_str_id)
                        #     retweets_user_lang_dict[original_user_lang].add(original_user_str_id)
                        else:
                            not_retweet_count += 1

                        # if 'retweeted_status' in line_as_dict.keys():  # a retweet
                        #     if line_as_dict['retweeted_status']['truncated'] == False:
                        #         text = line_as_dict['retweeted_status']['full_text']
                        #     else:
                        #         text = line_as_dict['retweeted_status']['extended_tweet']['full_text']
                        if line_as_dict['truncated'] == False:
                            text = line_as_dict['full_text']
                        else:
                            text = line_as_dict['extended_tweet']['full_text']
                        text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', '',
                                      text)  # remove urls
                        tweets_set_per_user[user_str_id].add((tweet_str_id,text))
                        # tweet_lang_dict[tweet_lang].add(tweet_str_id)
                        # user_lang_dict[user_lang].add(user_str_id)
                zf.close()
    print("There are {} retweets".format(retweet_count))
    print("There are {} non-retweets".format(not_retweet_count))
    # with open("./Echo networks/dicts/recent_echoes/user_lang_dict.pkl", "wb") as fout:
    #     pickle.dump(user_lang_dict, fout)
    # with open("./Echo networks/dicts/recent_echoes/tweet_lang_dict.pkl", "wb") as fout:
    #     pickle.dump(tweet_lang_dict, fout)
    # with open("./Echo networks/dicts/recent_echoes/retweets_user_lang_dict.pkl", "wb") as fout:
    #     pickle.dump(retweets_user_lang_dict, fout)
    # with open("./Echo networks/dicts/recent_echoes/retweets_tweet_lang_dict.pkl", "wb") as fout:
    #     pickle.dump(retweets_tweet_lang_dict, fout)
    with open("./Echo networks/dicts/recent_echoes/tweets_set_per_user.pkl", "wb") as fout:
        pickle.dump(tweets_set_per_user, fout)

if __name__ == '__main__':
    base_path = "/work/data/echoes/"
    original_echoes_path = base_path + "original_hate_networks/data/"
    recent_echoes_path = base_path + "only_english/recent_history.zip"
    create_user_dicts(original_echoes_path)
    # distinct_tweets_and_users_languages(recent_echoes_path)
    # distinct_tweets_and_users_languages(original_echoes_path)