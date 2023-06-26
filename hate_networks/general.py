import string
from emoji_extractor.extract import Extractor

extract = Extractor()
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
# import gensim
import gzip
import os
import json
from collections import Counter, defaultdict
import zipfile, re
from io import BytesIO
# import pyLDAvis
import simplejson
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from datetime import datetime
from ast import literal_eval
import multiprocessing as mp
import itertools
from collections import Counter, defaultdict

import logging

logger = logging.getLogger(__name__)


# General functions
def lookup(json, k):
    '''
    A function that given a dictionary and a key (or keys) returns the value of the inner most key.
    :param json: dictionary
    :param k: key or multiple nested keys separated by dots.
    :return:
    '''
    # return json[k]
    if '.' in k:
        # jpath path
        ks = k.split('.')
        v = json
        for k in ks: v = v.get(k, {})
        return v  # or ""
    return json.get(k, "")


def create_dir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)


# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield (gensim.detection_utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Preprocessing
def text_preprocessing(texts):
    alphabet = list(string.ascii_lowercase + string.ascii_uppercase)
    preprocessed_tweets = []
    special_characters = ["卐", "…", "!", "-", "?", ",", ".", "=", "&", "*", "{", "}", "[", "]", ":", ";", '\\',
                          '`', ]
    for tweet in tqdm(texts):
        tweet = tweet.lower()
        tweet = tweet.replace('u.s.', 'united states').replace('US', 'united states')
        tweet = tweet.replace("\n", "")
        tweet = tweet.replace('&amp;', ' and ').replace('&lt;', ' < ').replace('&gt;', ' > '). \
            replace('&le;', ' ≤ ').replace('&ge;', ' ≥ ')
        for special_char in special_characters:
            tweet = tweet.replace(special_char, ' ' + special_char + ' ')

        count_emojis_dict = extract.count_emoji(tweet, check_first=True)
        for em in count_emojis_dict.keys():
            tweet = tweet.replace(em, ' ' + em + ' ')
        tweet = tweet.split()
        tweet = [word for word in tweet if word != '']
        tweet = [word for word in tweet if '@' not in word]  # omit mentions from text
        deleted = True
        while deleted:
            if len(tweet) < 2:
                break
            for i in range(len(tweet) - 1):
                if tweet[i] == tweet[i + 1] and tweet[i] not in [')', '(']:
                    del tweet[i]
                    break
                if i == len(tweet) - 2:
                    deleted = False
        preprocessed_tweets.append(" ".join(tweet))
    with open('hate_networks/Echo networks/all_preprocessed_tweets_list_1-10-19.pkl', 'wb') as f:
        pickle.dump(preprocessed_tweets, f)
    return preprocessed_tweets


def extract_data_from_tweets(data_path, ignore_rt, ignore_punct, base_path):
    '''
    A function that given a path to a csv file with tweets returns the corpora which includes tweets of each user
    :param path: csv file path containing tweets
    :return: corpora of each user
    '''
    logger.info("Extracting users, their corpora and network edges (mention, RT, replies) from the data...")

    network_dir_path = os.path.join(base_path, "network_data")
    create_dir_if_missing(network_dir_path)
    edges_dir_path = os.path.join(network_dir_path, "edges")
    create_dir_if_missing(edges_dir_path)
    # mention_edges_file_path = os.path.join(edges_dir_path, "mention_edges.tsv")
    # if os.path.exists(mention_edges_file_path):
    #     logger.info(f"{base_path.split('/')[-1]} edges already created.")
    #     return

    mentions_dict = defaultdict(dict)
    retweets_dict = defaultdict(dict)
    in_reply_to_dict = defaultdict(dict)

    tweets_per_user = defaultdict(list)
    users_list = []
    corpora = []
    all_tweets = []
    users_corpora = defaultdict(str)
    file_type = data_path.split('.')[-1]
    rt_count = 0
    URL_RE = re.compile(
        r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    PUNCT_RE = re.compile('[“",?;:!\\-\[\]_.%/\n]')
    MENTION_RE = re.compile(r'@\w+')
    if 'parler' in data_path:  # file_type == 'tsv': # Parler
        df = pd.read_csv(data_path, sep='\t', encoding='utf-8')
        # remove dupliactes of same post text and same person ID
        df.drop_duplicates(subset=['text', 'username'], keep='first', inplace=True)
        # remove urls from the tweet's text
        df['text'].replace(to_replace=r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', value='',
                           regex=True,
                           inplace=True)
        for index, row in df.iterrows():  # append all tweets to each user
            tweet_text = row['text']
            tweets_per_user[row['username']].append(tweet_text)
            all_tweets.append(tweet_text)

        # for user, tweets in tweets_per_user.items():  # extract all corpora of the different users to a list
        #     # users_list.append(user)
        #     current_user_tweets = ''
        #     for tweet in tweets:
        #         current_user_tweets += tweet + '\n'
        #     corpora.append(current_user_tweets)

        with open("/sise/home/tommarz/parler-hate-speech/data/parler_users_comments_dict.p", "rb") as f:
            mentions_dict = pickle.load(f)

        with open("/sise/home/tommarz/parler-hate-speech/data/parler_users_echos_dict.p", "rb") as f:
            retweets_dict = pickle.load(f)

    elif file_type == 'csv':  # todo: support the sayiqan data as well
        df = pd.read_csv(data_path)
        # remove dupliactes of same post text and same person ID
        df.drop_duplicates(subset=['PostText', 'PersonID'], keep='first', inplace=True)
        # remove urls from the tweet's text
        df['PostText'].replace(to_replace=r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', value='',
                               regex=True,
                               inplace=True)
        for index, row in df.iterrows():  # append all tweets to each user
            tweet_text = row['PostText']
            tweets_per_user[row['PersonID']].append(tweet_text)

        for user, tweets in tweets_per_user.items():  # extract all corpora of the different users to a list
            users_list.append(user)
            current_user_tweets = ''
            for tweet in tweets:
                current_user_tweets += tweet + '\n'
            corpora.append(current_user_tweets)
    elif file_type == 'zip':
        with zipfile.ZipFile(data_path, "r") as zfile:
            for name in tqdm(zfile.namelist()):
                # We have a zip within a zip
                if re.search('\.gz$', name) != None:
                    zfiledata = BytesIO(zfile.read(name))
                    with gzip.open(zfiledata) as zf:
                        for line in zf:
                            json_tweet = simplejson.loads(line)

                            # get text
                            if 'retweeted_status' in json_tweet.keys():  # a retweet
                                rt_count += 1
                                if lookup(json_tweet, 'retweeted_status.truncated') == False:
                                    text = lookup(json_tweet,
                                                  'retweeted_status.full_text')  # todo: check if should be text or full text
                                else:
                                    text = lookup(json_tweet, 'retweeted_status.extended_tweet.full_text')
                            elif lookup(json_tweet, 'truncated') == False:
                                text = lookup(json_tweet, 'full_text')  # todo: check if should be text or full text
                            else:
                                text = lookup(json_tweet, 'extended_tweet.full_text')

                            user_id_str = lookup(json_tweet, 'user.id_str')

                            # in reply to
                            user_in_reply_to_id = lookup(json_tweet, 'in_reply_to_user_id_str')
                            if user_in_reply_to_id != None:
                                if user_id_str != user_in_reply_to_id:
                                    if user_in_reply_to_id not in in_reply_to_dict[user_id_str].keys():
                                        in_reply_to_dict[user_id_str][user_in_reply_to_id] = []
                                    in_reply_to_dict[user_id_str][user_in_reply_to_id].append(text)
                            # retweets
                            if 'retweeted_status' in json_tweet.keys():
                                user_retweeted_id = lookup(json_tweet, 'retweeted_status.user.id_str')
                                if user_id_str != user_retweeted_id:
                                    if user_retweeted_id not in retweets_dict[user_id_str].keys():
                                        retweets_dict[user_id_str][user_retweeted_id] = []
                                    retweets_dict[user_id_str][user_retweeted_id].append(text)

                            # mentions
                            user_mentions = lookup(json_tweet, 'entities.user_mentions')

                            for user_mentioned in user_mentions:
                                user_mentioned_id = lookup(user_mentioned, 'id_str')
                                if user_id_str != user_mentioned_id:  # self mentioning handling
                                    if user_mentioned_id not in mentions_dict[user_id_str].keys():
                                        mentions_dict[user_id_str][user_mentioned_id] = []
                                    mentions_dict[user_id_str][user_mentioned_id].append(text)

                                text = URL_RE.sub('', text)
                            if ignore_punct:
                                text = PUNCT_RE.sub(' ', text)
                            if ignore_rt:
                                continue
                            else:
                                tweets_per_user[user_id_str].append(text)
                                all_tweets.append(text)
    elif file_type == 'json':  # gab data
        username2id_mapping_fn = os.path.join(base_path, "pickled_data", "username2id_mapping_with_duplicates.pkl")
        username2id_mapping = pickle.load(open(username2id_mapping_fn, "rb"))
        with open(data_path, "r", encoding='utf-8') as fin:
            for line in tqdm(fin):
                json_content = json.loads(line[:-1])
                user_id_str = str(json_content["actuser"]["id"])
                json_type = json_content["type"]
                text = json_content['post']['body']
                if json_type == 'repost':  # like RT
                    user_reposted_id = str(json_content['post']['user']['id'])
                    if user_id_str != user_reposted_id:
                        if user_reposted_id not in retweets_dict[user_id_str].keys():
                            retweets_dict[user_id_str][user_reposted_id] = []
                        retweets_dict[user_id_str][user_reposted_id].append(text)
                elif json_type == 'post':  # here take the mentions from text, as there is no field in gab API for it.
                    text = URL_RE.sub('', text)
                    mentions = [mention.replace('@', '') for mention in MENTION_RE.findall(text)]
                    for mention in mentions:
                        if mention in username2id_mapping.keys():
                            user_mentioned_id = username2id_mapping[mention]
                        else:
                            user_mentioned_id = mention  # keep the username as is if no mapping to id
                        if user_mentioned_id != user_id_str:
                            if user_mentioned_id not in mentions_dict[user_id_str].keys():
                                mentions_dict[user_id_str][user_mentioned_id] = []
                            mentions_dict[user_id_str][user_mentioned_id].append(text)
                text = URL_RE.sub('', text)
                if ignore_punct:
                    text = PUNCT_RE.sub(' ', text)

                if ignore_rt:
                    continue
                else:
                    tweets_per_user[user_id_str].append(text)
                    all_tweets.append(text)

    else:  # directory containing multiple data files, currently jsons of which each is with a list of tweets
        for user_idx, file in enumerate(tqdm(os.listdir(data_path))):
            if user_idx == 10000:
                break
            with open(os.path.join(data_path, file), "r") as fin:
                tweet_list = json.load(fin)
                for tweet_dict in tweet_list:
                    if tweet_dict['lang'] != 'en':
                        continue
                    user_id_str = tweet_dict['user']['id_str']
                    if 'retweeted_status' in tweet_dict.keys():  # a retweet
                        rt_count += 1
                        text = tweet_dict['retweeted_status']['full_text']
                    else:
                        text = tweet_dict['full_text']

                    # in reply to
                    user_in_reply_to_id = tweet_dict['in_reply_to_user_id_str']
                    if user_in_reply_to_id != None:
                        if user_id_str != user_in_reply_to_id:
                            if user_in_reply_to_id not in in_reply_to_dict[user_id_str].keys():
                                in_reply_to_dict[user_id_str][user_in_reply_to_id] = []
                            in_reply_to_dict[user_id_str][user_in_reply_to_id].append(text)
                    # retweets
                    if 'retweeted_status' in tweet_dict.keys():
                        user_retweeted_id = tweet_dict['retweeted_status']['user']['id_str']
                        if user_id_str != user_retweeted_id:
                            if user_retweeted_id not in retweets_dict[user_id_str].keys():
                                retweets_dict[user_id_str][user_retweeted_id] = []
                            retweets_dict[user_id_str][user_retweeted_id].append(text)

                    # mentions
                    user_mentions = tweet_dict['entities']['user_mentions']
                    for user_mentioned in user_mentions:
                        user_mentioned_id = lookup(user_mentioned, 'id_str')
                        if user_id_str != user_mentioned_id:  # self mentioning handling
                            if user_mentioned_id not in mentions_dict[user_id_str].keys():
                                mentions_dict[user_id_str][user_mentioned_id] = []
                            mentions_dict[user_id_str][user_mentioned_id].append(text)

                    text = URL_RE.sub('', text)
                    if ignore_punct:
                        text = PUNCT_RE.sub(' ', text)

                    if ignore_rt:
                        continue
                    else:
                        tweets_per_user[user_id_str].append(text)
                        all_tweets.append(text)

    logger.info(f"Number of retweets: {rt_count:,}")

    # in reply to
    if file_type != 'json':

        in_reply_to_dict_to_df = {}
        i = 0
        for user_id, mentioned_user in in_reply_to_dict.items():
            for mentioned_user_id, texts in mentioned_user.items():
                in_reply_to_dict_to_df[i] = {'source': user_id, 'dest': mentioned_user_id, 'weight': len(texts)}
                i += 1
    # retweets
    retweet_dict_to_df = {}
    i = 0
    # print(len(retweets_dict))
    for user_id, mentioned_user in retweets_dict.items():
        for mentioned_user_id, texts in mentioned_user.items():
            retweet_dict_to_df[i] = {'source': user_id, 'dest': mentioned_user_id,
                                     'weight': texts if 'parler' in data_path else len(texts)}
        i += 1
    # print(i)
    # mentions
    mention_dict_to_df = {}
    i = 0
    for user_id, mentioned_user in mentions_dict.items():
        for mentioned_user_id, texts in mentioned_user.items():
            mention_dict_to_df[i] = {'source': user_id, 'dest': mentioned_user_id,
                                     'weight': texts if 'parler' in data_path else len(texts)}
            i += 1
    # print(i)
    if file_type != 'json' and 'parler' not in data_path:
        in_reply_to_edges = pd.DataFrame.from_dict(in_reply_to_dict_to_df, orient="index")
        in_reply_to_edges.to_csv(os.path.join(edges_dir_path, "in_reply_to_edges_df.tsv"), sep='\t', index=False)
    retweet_edges = pd.DataFrame.from_dict(retweet_dict_to_df, orient="index")
    mention_edges = pd.DataFrame.from_dict(mention_dict_to_df, orient="index")

    logger.info(f"Saving edges to tsv files...")
    # date = datetime.today().strftime('%Y-%m-%d')

    retweet_edges.to_csv(os.path.join(edges_dir_path, "retweet_edges_df.tsv"), sep='\t', index=False)
    mention_edges.to_csv(os.path.join(edges_dir_path, "mention_edges_df.tsv"), sep='\t', index=False)

    pickled_edges_dir_path = os.path.join(base_path, "pickled_data", "edges")
    create_dir_if_missing(pickled_edges_dir_path)
    if file_type != 'json':
        with open(os.path.join(pickled_edges_dir_path, "in_reply_to_dict_with_text.pkl"), "wb") as fout:
            pickle.dump(in_reply_to_dict, fout)
    with open(os.path.join(pickled_edges_dir_path, "retweets_dict_with_text.pkl"), "wb") as fout:
        pickle.dump(retweets_dict, fout)
    with open(os.path.join(pickled_edges_dir_path, "mentions_dict_with_text.pkl"), "wb") as fout:
        pickle.dump(mentions_dict, fout)

    logger.info(f"saving users ids and their corpora...")
    for user, tweets in tweets_per_user.items():  # extract all corpora of the different users to a list
        users_list.append(user)
        current_user_tweets = ''
        for tweet in tweets:
            current_user_tweets += tweet + ' '
        corpora.append(current_user_tweets)
        users_corpora[user] = current_user_tweets

    data_users_mentions_df = mention_edges[mention_edges['dest'].isin(users_list)].reset_index(drop=True)
    data_users_retweet_df = retweet_edges[retweet_edges['dest'].isin(users_list)].reset_index(drop=True)
    if file_type != 'json' and 'parler' not in data_path:
        data_users_in_reply_to_df = in_reply_to_edges[in_reply_to_edges['dest'].isin(users_list)].reset_index(drop=True)
        data_users_in_reply_to_df.to_csv(os.path.join(edges_dir_path, "data_users_in_reply_to_edges_df.tsv"), sep='\t',
                                         index=False)

    data_users_mentions_df.to_csv(os.path.join(edges_dir_path, "data_users_mention_edges_df.tsv"), sep='\t',
                                  index=False)
    data_users_retweet_df.to_csv(os.path.join(edges_dir_path, "data_users_retweet_edges_df.tsv"), sep='\t', index=False)

    users_df = pd.DataFrame(np.array(users_list), columns=['user_id'])
    users_df.to_csv(os.path.join(base_path, 'tsv_data', 'users.tsv'), sep='\t', index=False)

    with open(f"{os.path.join(base_path, 'pickled_data', 'corpora_list.pkl')}", 'wb') as f:
        pickle.dump(corpora, f)
    with open(f"{os.path.join(base_path, 'pickled_data', 'corpora_list_per_user.pkl')}", 'wb') as f:
        pickle.dump(tweets_per_user, f)
    with open(f"{os.path.join(base_path, 'pickled_data', 'corpora_str_per_user_.pkl')}", 'wb') as f:
        pickle.dump(users_corpora, f)
    logger.info(f"There are {len(all_tweets)} posts before dropping duplicates.")
    all_tweets = list(set(all_tweets))
    logger.info(f"There are {len(all_tweets)} unique posts.")
    with open(f"{os.path.join(base_path, 'pickled_data', 'all_tweets.pkl')}", 'wb') as f:
        pickle.dump(all_tweets, f)

    # return users_df, corpora, tweets_per_user, users_corpora


def get_all_echo_tweets(path):
    all_tweets = []
    with zipfile.ZipFile(path, "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json = simplejson.loads(line)
                        if 'retweeted_status' in json.keys():  # a retweet
                            if lookup(json, 'retweeted_status.truncated') == False:
                                text = lookup(json, 'retweeted_status.full_text')
                            else:
                                text = lookup(json, 'retweeted_status.extended_tweet.full_text')
                            # text = lookup(json, 'retweeted_status.full_text')
                        elif lookup(json, 'truncated') == False:
                            text = lookup(json, 'full_text')
                        else:
                            text = lookup(json, 'extended_tweet.full_text')
                        text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', '',
                                      text)  # remove urls
                        # splitted_text = text.split(' ')
                        # words_to_remove = []
                        # for word in splitted_text:
                        # if word.startswith('@') or word == 'RT':  # remove mentions and rt if exists
                        #     words_to_remove.append(word)
                        # for word_to_remove in words_to_remove:
                        #     splitted_text.remove(word_to_remove)
                        # text = ' '.join(splitted_text)
                        all_tweets.append(text)
                zf.close()
    with open('hate_networks/Echo networks/all_tweets_list_1-10-19.pkl', 'wb') as f:
        pickle.dump(all_tweets, f)
    return all_tweets


def distinct_tweets_and_users_languages(path=None):
    '''
    A function that given a path to a csv file with tweets returns the corpora which includes tweets of each user
    :param path: csv file path containing tweets
    :return: corpora of each user
    '''
    tweets_per_language = defaultdict(list)
    users_dict = defaultdict(str)

    with zipfile.ZipFile(path, "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json = simplejson.loads(line)
                        tweet_lang = json["lang"]
                        user_lang = json["user"]["lang"]
                        if lookup(json, 'truncated') == True:
                            text = lookup(json, 'extended_tweet.full_text')
                        else:
                            text = lookup(json, 'text')
                        text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', '', text)
                        tweets_per_language[tweet_lang].append(text)
                        users_dict[lookup(json, 'user.id')] = user_lang
                zf.close()

    with open("hate_networks/Echo networks/tweets_lang_dict.pkl", "wb") as f:
        pickle.dump(tweets_per_language, f)
    with open("hate_networks/Echo networks/users_lang_dict.pkl", "wb") as f:
        pickle.dump(users_dict, f)

    return tweets_per_language, users_dict


def get_start_end_dates(path):
    print("Started time start end")
    min_date = datetime.strptime('Mon Jun 8 10:51:32 +0000 2019', '%a %b %d %H:%M:%S +0000 %Y')
    max_date = datetime.strptime('Mon Jun 8 10:51:32 +0000 2001', '%a %b %d %H:%M:%S +0000 %Y')
    with zipfile.ZipFile(path, "r") as zfile:
        for name in zfile.namelist():
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json = simplejson.loads(line)
                        created_at = lookup(json, 'created_at')
                        # created_at = 'Mon Jun 8 10:51:32 +0000 2009'  # Get this string from the Twitter API
                        dt = datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
                        if dt < min_date:
                            min_date = dt
                        if dt > max_date:
                            max_date = dt
                        #
                        # if 'retweeted_status' in json.keys():  # a retweet
                        #   text = lookup(json, 'retweeted_status.full_text')
                        # elif lookup(json, 'truncated') == False:
                        #   text = lookup(json, 'full_text')
                        # else:
                        #   text = lookup(json, 'extended_tweet.full_text')
                        # text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', '', text)  # remove urls
                        # splitted_text = text.split(' ')
                        # words_to_remove = []
                        # for word in splitted_text:
                        #   if word.startswith('@') or word == 'RT':  # remove mentions and rt if exists
                        #     words_to_remove.append(word)
                        # for word_to_remove in words_to_remove:
                        #   splitted_text.remove(word_to_remove)
                        # text = ' '.join(splitted_text)
                        # all_tweets.append(text)
                        # print(min_date)
                        # print(max_date)
                zf.close()
    print(min_date)
    print(max_date)


def create_network_edges(base_path):
    # edges_dir_path = os.path.join(base_path, "network_edges")
    # create_dir_if_missing(edges_dir_path)
    # mention_edges_file_path = os.path.join(edges_dir_path, "mention_edges.tsv")
    # if os.path.exists(mention_edges_file_path):
    #     logger.info(f"{base_path.split('/')[-1]} edges already created.")
    #     return
    # mention_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    # retweet_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    # in_reply_to_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    # mentions_dict = defaultdict(dict)
    # retweets_dict = defaultdict(dict)
    # in_reply_to_dict = defaultdict(dict)

    with zipfile.ZipFile("/data/home/eyalar/antisemite_hashtags/Resources/recent_history.zip", "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        pass

    print("Before creating dfs")


def get_famous_antisemites():
    fam_user_handles = ['RichardBSpencer', 'Cernovich', 'DrDavidDuke', 'ExIronMarch', 'AndrewAnglinPhD', 'dailystormer',
                        'ThaRightStuff', 'GoyimGoddess', 'm_enoch']
    fam_user_ids = ['402181258', '358545917', '72931184', '1192567241482678272', '357512807', '709943695', '1014432451',
                    '754987972127186946', '324429651']
    mention_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    retweet_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    in_reply_to_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    mentions_dict = defaultdict(dict)
    retweets_dict = defaultdict(dict)
    in_reply_to_dict = defaultdict(dict)
    famouse_in_our_data = []
    with zipfile.ZipFile("/data/home/eyalar/antisemite_hashtags/Resources/recent_history.zip", "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json = simplejson.loads(line)
                        user = lookup(json, 'user')
                        user_id = lookup(user, 'id_str')
                        user_screen_name = lookup(user, 'screen_name')
                        if user_id in fam_user_ids or user_screen_name in fam_user_handles:
                            famouse_in_our_data.append(user_id)
                        user_mentions = lookup(json, 'entities.user_mentions')
                        if 'retweeted_status' in json.keys():  # a retweet
                            if lookup(json, 'retweeted_status.truncated') == False:
                                text = lookup(json, 'retweeted_status.full_text')
                            else:
                                text = lookup(json, 'retweeted_status.extended_tweet.full_text')
                        elif lookup(json, 'truncated') == False:
                            text = lookup(json, 'full_text')
                        else:
                            text = lookup(json, 'extended_tweet.full_text')

                        # in reply to
                        user_mentioned_id = lookup(json, 'in_reply_to_user_id_str')
                        if user_mentioned_id != None:
                            in_reply_to_screen_name = lookup(json, 'in_reply_to_screen_name')
                            if (
                                    user_mentioned_id in fam_user_ids or in_reply_to_screen_name in fam_user_handles) and user_id != user_mentioned_id:  # self mentioning handling
                                if user_mentioned_id not in in_reply_to_dict[user_id].keys():
                                    in_reply_to_dict[user_id][user_mentioned_id] = []
                                in_reply_to_dict[user_id][user_mentioned_id].append(text)

                        # retweets
                        if 'retweeted_status' in json.keys():
                            user_mentioned_id = lookup(json, 'retweeted_status.user.id_str')
                            user_mentioned_screen_name = lookup(json, 'retweeted_status.user.screen_name')
                            if (
                                    user_mentioned_id in fam_user_ids or user_mentioned_screen_name in fam_user_handles) and user_id != user_mentioned_id:  # self mentioning handling
                                if user_mentioned_id not in retweets_dict[user_id].keys():
                                    retweets_dict[user_id][user_mentioned_id] = []
                                retweets_dict[user_id][user_mentioned_id].append(text)

                        # mentions
                        for user_mentioned in user_mentions:
                            user_mentioned_id = lookup(user_mentioned, 'id_str')
                            user_mentioned_screen_name = lookup(user_mentioned, 'screen_name')
                            if (
                                    user_mentioned_id in fam_user_ids or user_mentioned_screen_name in fam_user_handles) and user_id != user_mentioned_id:  # self mentioning handling
                                if user_mentioned_id not in mentions_dict[user_id].keys():
                                    mentions_dict[user_id][user_mentioned_id] = []
                                mentions_dict[user_id][user_mentioned_id].append(text)
                zf.close()
    zfile.close()
    # in reply to
    for user_id, mentioned_user in in_reply_to_dict.items():
        for mentioned_user_id, texts in mentioned_user.items():
            in_reply_to_edges = in_reply_to_edges.append(
                {'source': user_id, 'dest': mentioned_user_id, 'weight': len(texts), 'texts': texts}, ignore_index=True)

    # retweets
    for user_id, mentioned_user in retweets_dict.items():
        for mentioned_user_id, texts in mentioned_user.items():
            retweet_edges = retweet_edges.append(
                {'source': user_id, 'dest': mentioned_user_id, 'weight': len(texts), 'texts': texts}, ignore_index=True)

    # mentions
    for user_id, mentioned_user in mentions_dict.items():
        for mentioned_user_id, texts in mentioned_user.items():
            mention_edges = mention_edges.append(
                {'source': user_id, 'dest': mentioned_user_id, 'weight': len(texts), 'texts': texts}, ignore_index=True)

    mention_edges.to_csv('hate_networks/Echo networks/Edges/famous_antisemite/mention_edges_with_text_df_.csv',
                         index=False)
    in_reply_to_edges.to_csv('hate_networks/Echo networks/Edges/famous_antisemite/in_reply_to_edges_with_text_df_.csv',
                             index=False)
    retweet_edges.to_csv('hate_networks/Echo networks/Edges/famous_antisemite/retweet_edges_with_text_df_.csv',
                         index=False)
    with open('hate_networks/Echo networks/Edges/famous_antisemite/in_reply_to_dict_with_text_df_.pkl', "wb") as fout:
        pickle.dump(in_reply_to_dict, fout)
    with open('hate_networks/Echo networks/Edges/famous_antisemite/mentions_dict_with_text_df_.pkl', "wb") as fout:
        pickle.dump(mentions_dict, fout)
    with open('hate_networks/Echo networks/Edges/famous_antisemite/retweets_dict_with_text_df_.pkl', "wb") as fout:
        pickle.dump(retweets_dict, fout)


def get_users_statistics():
    # labeled_df = pd.read_csv("./hate_networks/Echo networks/csv_data/users_labeled.csv")
    # N_users = list(labeled_df[labeled_df["label"] == 0]["user_id"])
    # HS_users = list(labeled_df[labeled_df["label"] == 1]["user_id"])
    # R_users = list(labeled_df[labeled_df["label"] == 2]["user_id"])
    # tm_clustering_df = pd.read_csv(
    #     "./hate_networks/Echo networks/Topic model/Kmeans/tm_kmeans_users_clusters_30_topics_lda_tm_type_cv_vec_type_3_clusters.csv")
    # tm_N_R_users = tm_clustering_df[tm_clustering_df["tm_kmeans_cluster"].apply(lambda tkc: tkc in [0,2])]
    # tm_HS_users = tm_clustering_df[tm_clustering_df["tm_kmeans_cluster"] == 1]
    users_data = defaultdict(dict)
    max_date = datetime.strptime('Mon Jun 8 10:51:32 +0000 2001', '%a %b %d %H:%M:%S +0000 %Y')

    with zipfile.ZipFile("/data/home/eyalar/antisemite_hashtags/Resources/recent_history.zip", "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json = simplejson.loads(line)
                        user = lookup(json, 'user')
                        user_id = lookup(user, 'id_str')
                        if user_id not in users_data.keys():
                            users_data[user_id] = {"tweet_count": 0, "created_at": datetime.strptime(user["created_at"],
                                                                                                     '%a %b %d %H:%M:%S +0000 %Y'),
                                                   "twitter_age": 0, "tweets_per_day": -1,
                                                   "last_tweet_date": datetime.strptime('Mon Jun 8 10:51:32 +0000 2001',
                                                                                        '%a %b %d %H:%M:%S +0000 %Y'),
                                                   "last_tweet_obj": None, "statuses_count": -1, "friends_count": -1,
                                                   "followers_count": -1, "replies": 0, "retweets": 0, "mentions": 0,
                                                   "all_mentions": 0}
                        created_at = lookup(json, 'created_at')
                        dt = datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
                        if dt > max_date:
                            max_date = dt
                        if dt > users_data[user_id]["last_tweet_date"]:  # need to update last tweet obj
                            users_data[user_id]["last_tweet_date"] = dt
                            users_data[user_id]["last_tweet_obj"] = json
                        users_data[user_id]["tweet_count"] += 1
                        if 'retweeted_status' in json.keys():  # a retweet
                            users_data[user_id]["retweets"] += 1
                        user_in_reply_to_id = lookup(json, 'in_reply_to_user_id_str')
                        if user_in_reply_to_id != None:
                            users_data[user_id]["replies"] += 1
                        user_mentions = lookup(json, 'entities.user_mentions')
                        if user_mentions:
                            users_data[user_id]["mentions"] += 1
                            for i in range(len(user_mentions)):
                                users_data[user_id]["all_mentions"] += 1

    for user_id, user_data in users_data.items():
        users_data[user_id]["twitter_age"] = (max_date - users_data[user_id]["created_at"]).days
        users_data[user_id]["friends_count"] = users_data[user_id]["last_tweet_obj"]["user"]["friends_count"]
        users_data[user_id]["followers_count"] = users_data[user_id]["last_tweet_obj"]["user"]["followers_count"]
        users_data[user_id]["statuses_count"] = users_data[user_id]["last_tweet_obj"]["user"]["statuses_count"]
        users_data[user_id]["tweets_per_day"] = users_data[user_id]["statuses_count"] / users_data[user_id][
            "twitter_age"]
    with open("./hate_networks/Echo networks/pickled_data/users_statistics.pkl", "wb") as fout:
        pickle.dump(users_data, fout)


def get_users_hashtags_and_words():
    users_hashtags = defaultdict(dict)
    users_search_words = defaultdict(dict)
    users_hashtag_counts = defaultdict(int)
    search_words = ['muslim', 'arab', 'nazi', 'immigrant', 'immigration', 'woman', 'parasite', 'israel', 'zionist',
                    'zionism', 'pedophile', 'fake', 'fakenews',
                    'fake news', 'fuck', 'shit', 'white', 'black', 'genocide', 'white genocide', 'hitler', 'hh', 'war',
                    'hate', 'european', 'kike', 'skype',
                    'skittle', 'propaganda', 'media', 'mainstream', 'hate', 'vaccine', 'vaccines', 'vaccination',
                    'cunt', 'whore',
                    'pussy', 'snowflake', 'commie', 'cuck', 'hillary', 'clinton', 'obama', 'trump', '@realdonaldtrump',
                    '@hillaryclinton']
    with zipfile.ZipFile("/data/home/eyalar/antisemite_hashtags/Resources/recent_history.zip", "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json = simplejson.loads(line)
                        user = lookup(json, 'user')
                        user_id = lookup(user, 'id_str')
                        if user_id not in users_hashtags.keys():
                            users_hashtags[user_id] = {}
                        if user_id not in users_hashtag_counts.keys():
                            users_hashtag_counts[user_id] = 0
                        if user_id not in users_search_words.keys():
                            users_search_words[user_id] = {}
                        hashtags = lookup(json, 'entities.hashtags')
                        if hashtags:
                            users_hashtag_counts[user_id] += 1
                            for hashtag in hashtags:
                                hashtag_text = hashtag["text"]
                                if hashtag_text not in users_hashtags[user_id].keys():
                                    users_hashtags[user_id][hashtag_text] = 1
                                else:
                                    users_hashtags[user_id][hashtag_text] += 1

                        if 'retweeted_status' in json.keys():  # a retweet
                            if lookup(json, 'retweeted_status.truncated') == False:
                                text = lookup(json, 'retweeted_status.full_text')
                            else:
                                text = lookup(json, 'retweeted_status.extended_tweet.full_text')
                        elif lookup(json, 'truncated') == False:
                            text = lookup(json, 'full_text')
                        else:
                            text = lookup(json, 'extended_tweet.full_text')
                        text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/?\S', '',
                                      text)  # remove urls
                        text = text.lower()
                        splitted_text = text.split()
                        for word in search_words:
                            if len(word.split()) == 1:
                                for text_word in splitted_text:
                                    if word == text_word:
                                        if word not in users_search_words[user_id].keys():
                                            users_search_words[user_id][word] = 1
                                        else:
                                            users_search_words[user_id][word] += 1
                            else:
                                splitted_word = word.split()
                                if splitted_word[0] in text and splitted_word[1] in text:
                                    if word not in users_search_words[user_id].keys():
                                        users_search_words[user_id][word] = 1
                                    else:
                                        users_search_words[user_id][word] += 1
    with open("./hate_networks/Echo networks/pickled_data/users_hashtags.pkl", "wb") as fout:
        pickle.dump(users_hashtags, fout)
    with open("./hate_networks/Echo networks/pickled_data/users_search_words.pkl", "wb") as fout:
        pickle.dump(users_search_words, fout)
    with open("./hate_networks/Echo networks/pickled_data/users_hashtag_counts.pkl", "wb") as fout:
        pickle.dump(users_hashtag_counts, fout)
