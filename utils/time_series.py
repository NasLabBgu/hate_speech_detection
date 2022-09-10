import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from datetime import datetime
from collections import defaultdict
import pickle
import os
import time
import gzip
import json
import multiprocessing as mp
from config.data_config import path_confs


def plot_agg_counts_ts(dict_type, word_list, agg_size=3):
    if dict_type == 'unigram':
        unigram_path = path_confs["unigram_dict"]
        with open(unigram_path, "rb") as fin:
            count_dict = pickle.load(fin)
    elif dict_type == 'bigram':
        bigram_path = path_confs["bigram_dict"]
        with open(bigram_path, "rb") as fin:
            count_dict = pickle.load(fin)
    else:
        raise ValueError(f"Supported dicts: 'unigram', 'bigram'. got: {dict_type}")
    fig = plt.figure(figsize=(14, 7))
    sorted_dates = sorted([str(date) for date in list(count_dict.keys())])
    for word in word_list:
        agg_dates = []
        zero_appearances = True
        for date in sorted_dates:
            if word in count_dict[date].keys():
                zero_appearances = False
        if zero_appearances:
            print(f"No appearance for the word '{word}' in the given data.")
            continue
        else:
            for date in sorted_dates:
                if word not in count_dict[date].keys():
                    count_dict[date][word] = 0
        agg_counts = []
        for i in range(0, (len(sorted_dates) - agg_size + 1), agg_size):
            if agg_size == 1:
                agg_dates.append(sorted_dates[i])
            else:
                agg_dates.append(sorted_dates[i] + "-" + sorted_dates[i + agg_size - 1])
            current_count_to_add = 0
            for j in range(agg_size):
                current_count_to_add += count_dict[sorted_dates[i + j]][word]

            agg_counts.append(current_count_to_add)
        plt.plot(agg_dates, agg_counts, label=word)
    plt.legend()
    plt.xticks(rotation=65)
    plt.margins(x=0.1, y=0.05)
    plt.gca().set_ylabel('Term count', fontsize=16)
    plt.title(f'Term count time series - {agg_size} days aggregation (antisemitism data)', fontsize=22)
    plt.gca().xaxis.set_tick_params(labelsize=12)
    plt.gca().yaxis.set_tick_params(labelsize=14)
    plt.show()
    fig_dir_path = path_confs["ts_plots"]
    fig_file_path = os.path.join(fig_dir_path, f"{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}.png")
    plt.savefig(fig_file_path)


def tweet_and_limit_count(input_path, file):
    print(file)
    tweet_dict = defaultdict(dd_list)  # key - date, value - (tweet_count, retweet_count, reply_count, limit_count)
    file_date = file.split('.')[2]
    for i in range(0,24):
        tweet_dict[file_date][i] = [0,0,0,0]

    # for i_0, i_1 in zip(range(0,24), range(1,25)):
    #     tweet_dict[file_date][f"{i_0}-{i_1}"] = [0,0,0,0]
    with gzip.open(input_path + file, 'rb') as f:
        for tweet in f:
            tweet_json = json.loads(tweet)
            if 'limit' in tweet_json.keys():
                limit_time = datetime.fromtimestamp(int(tweet_json['limit']['timestamp_ms']) / 1000)
                hour = limit_time.hour
                tweet_dict[file_date][hour][3] += 1
            else:
                created_at = tweet_json["created_at"]
                hour = time.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y').tm_hour

                if 'retweeted_status' in tweet_json.keys():  # retweet
                    tweet_dict[file_date][hour][1] += 1
                elif tweet_json['in_reply_to_status_id'] != None:  # reply
                    tweet_dict[file_date][hour][2] += 1
                else:  # normal tweet
                    tweet_dict[file_date][hour][0] += 1

    with open(f"/data/work/data/covid/ts/{file_date}_hourly_counts.pkl", "wb") as fout:
        pickle.dump(tweet_dict, file=fout)
def parallelize_tweet_and_limit_count(input_path):
    done_files = [file for file in os.listdir(f"/data/work/data/covid/ts/") if 'hourly' in file]
    done_file_dates = [file.split('_')[0] for file in done_files]
    sorted_files = sorted([file for file in os.listdir(input_path) if 'gz' in file])[:-1]
    tbd_files = [file for file in sorted_files if file.split('.')[2] not in done_file_dates]
    processors_num=30
    pool = mp.Pool(processors_num)
    [pool.apply_async(tweet_and_limit_count, args=(input_path, file)) for file in tbd_files]
    pool.close()
    pool.join()
def dd_list():
    return defaultdict(list)
def dd_int():
    return defaultdict(int)
def plot_count_ts(input_path='/data/work/data/covid/ts/'):
    sorted_unigrams = sorted([file for file in os.listdir(input_path) if 'hour' in file])
    all_counts_dict = defaultdict(dd_list)
    for file in sorted_unigrams:
        with open(input_path + file, "rb") as fin:
            current_dict = pickle.load(fin)
            all_counts_dict.update(current_dict)
    df = pd.DataFrame(columns=['date', 'tweet_count', 'retweet_count', 'reply_count', 'limit_count'])
    for date_str, hours in all_counts_dict.items():
        for hour, counts in hours.items():
            date = datetime.strptime(date_str, "%Y-%m-%d")
            date = date.replace(hour=hour)
            df = df.append({'date': date,
                            'tweet_count':int(counts[0]),
                            'retweet_count':int(counts[1]),
                            'reply_count':int(counts[2]),
                            'limit_count':int(counts[3])}, ignore_index=True)
    cols = df.columns.tolist()
    cols.remove('date')
    for col in cols:
        df[col] = df[col].astype(int)
    fig = plt.figure(figsize=(16,9))
    # sns.lineplot(x='date', y='tweet_count', data=df, label='tweet_count')
    # sns.lineplot(x='date', y='retweet_count', data=df, label='retweet_count')
    # sns.lineplot(x='date', y='reply_count', data=df, label='reply_count')
    # sns.lineplot(x='date', y='limit_count', data=df, label='limit_count')
    # plt.show()


if __name__ == '__main__':
    plot_agg_counts_ts(dict_type='unigram', word_list=['jewishvirus', '#jewishsupremacy', '#jewishvirus', '#zionismvirus', 'likudvirus16'],
                       agg_size=1)
    plot_agg_counts_ts(dict_type='bigram', word_list=['jewish virus', 'jewhate virus', 'jews coronavirus', 'israeli coronavirus'],
                       agg_size=1)
