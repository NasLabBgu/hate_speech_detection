import simplejson
import gzip
import zipfile, re, os
from tqdm import tqdm
from io import BytesIO
f = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(f, ".."))
from utils.constants import URL_RE
from collections import Counter, defaultdict
from config.hate_networks_config import path_conf

import logging
logger = logging.getLogger(__name__)


def get_tweets_per_user_echo_data():
    tweets_per_user = defaultdict(list)
    tweets_per_user_no_RT = defaultdict(list)
    all_tweets = []
    tweets_with_no_RT = []
    rt_count = 0
    with zipfile.ZipFile(path_conf["echo"]["raw_data"], "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            # name = '1368291252.json.gz'
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json_tweet = simplejson.loads(line)
                        user_id_str = json_tweet['user']['id_str']

                        if json_tweet['lang'] != 'en':
                            continue
                        is_rt = False
                        # get text
                        if 'retweeted_status' in json_tweet.keys():  # a retweet
                            is_rt = True
                            rt_count += 1
                            if json_tweet['retweeted_status']['truncated'] == False:
                                text = json_tweet['retweeted_status']['full_text']  # todo: check if should be text or full text
                            else:
                                text = json_tweet['retweeted_status']['extended_tweet']['full_text']
                        elif json_tweet['truncated'] == False:
                            text = json_tweet['full_text']  # todo: check if should be text or full text
                        else:
                            text = json_tweet['extended_tweet']['full_text']


                        text = URL_RE.sub('', text)

                        if not is_rt:
                            tweets_with_no_RT.append(text)
                            tweets_per_user_no_RT[user_id_str].append(text)
                        tweets_per_user[user_id_str].append(text)
                        all_tweets.append(text)
    import pickle
    output_path = "./hate_networks/echo_networks"
    logger.info(f"Number of retweets: {rt_count:,}")
    with open(f"{os.path.join(output_path, 'pickled_data', 'english_corpora_list_per_user.pkl')}", 'wb') as f:
        pickle.dump(tweets_per_user, f)
    with open(f"{os.path.join(output_path, 'pickled_data', 'english_corpora_list_per_user_no_RT.pkl')}", 'wb') as f:
        pickle.dump(tweets_per_user_no_RT, f)
    with open(f"{os.path.join(output_path, 'pickled_data', 'all_english_tweets.pkl')}", 'wb') as f:
        pickle.dump(all_tweets, f)
    with open(f"{os.path.join(output_path, 'pickled_data', 'all_english_tweets_no_RT.pkl')}", 'wb') as f:
        pickle.dump(tweets_with_no_RT, f)


get_tweets_per_user_echo_data()