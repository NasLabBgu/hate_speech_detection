import sys
import os
f = os.path.dirname(__file__)
sys.path.append(os.path.join(f, ".."))
import pandas as pd
import os
import json
from config.hate_networks_config import path_conf
from tqdm import tqdm
import pickle
from collections import defaultdict

gab_data_path = path_conf["gab"]["raw_data"]
followers_data_path = path_conf["gab"]["followers_data"]
followings_data_path = path_conf["gab"]["followings_data"]
output_path = path_conf["gab"]["base_output"]
follower_edges = []

# x = pd.read_csv(os.path.join(output_path, "network_data", "edges", "labeled_users_followers_df.tsv"), sep='\t')
# y = pd.read_csv(os.path.join(output_path, "network_data", "edges", "labeled_users_followings_df.tsv"), sep='\t')
# followers_df = x.append(y, ignore_index=True)
# followers_df.drop_duplicates(subset=['source', 'dest'], inplace=True)
# followers_df.to_csv(os.path.join(output_path, "network_data", "edges", "labeled_users_followers.tsv"), index=False, sep='\t')


def read_followers_followings_dfs():
    labeled_users = pd.read_csv("data/user_level/gab_users_2_labels.csv")["user_id"].tolist()
    followers_df = pd.read_csv(os.path.join(output_path, "network_data", "edges", "followers.tsv"), sep='\t')
    followings_df = pd.read_csv(os.path.join(output_path, "network_data", "edges", "followings.tsv"), sep='\t')
    username2id_mapping_fn = os.path.join("hate_networks", "gab_networks", "pickled_data", "username2id_mapping_with_duplicates.pkl")
    username2id_mapping = pickle.load(open(username2id_mapping_fn, "rb"))
    for col in ["source", "dest"]:
        followers_df[col] = followers_df[col].apply(lambda c_val: username2id_mapping[c_val] if c_val in username2id_mapping.keys() else c_val)
        followings_df[col] = followings_df[col].apply(lambda c_val: username2id_mapping[c_val] if c_val in username2id_mapping.keys() else c_val)

    followers_df.dropna(inplace=True)
    followings_df.dropna(inplace=True)
    labeled_users = [str(x) for x in labeled_users]
    x = followers_df[(followers_df["source"].isin(labeled_users)) | (followers_df["dest"].isin(labeled_users))]
    y = followings_df[(followings_df["source"].isin(labeled_users)) | (followings_df["dest"].isin(labeled_users))]
    print(len(x))
    print(len(y))
    x = x[(x["source"].apply(lambda s: s.isdigit())) & (x["dest"].apply(lambda d: d.isdigit()))]
    y = y[(y["source"].apply(lambda s: s.isdigit())) & (y["dest"].apply(lambda d: d.isdigit()))]
    print(len(x))
    print(len(y))
    # labeled_users_followers_df = followers_df[(followers_df["source"].isin(labeled_users)) | (followers_df["dest"].isin(labeled_users))]
    # labeled_users_followings_df = followings_df[(followings_df["source"].isin(labeled_users)) | (followings_df["dest"].isin(labeled_users))]

    x.to_csv(os.path.join(output_path, "network_data", "edges", "labeled_users_followers_df.tsv"), index=False, sep='\t')
    y.to_csv(os.path.join(output_path, "network_data", "edges", "labeled_users_followings_df.tsv"), index=False, sep='\t')

    followers_df.to_csv(os.path.join(output_path, "network_data", "edges", "followers_ids.tsv"), index=False, sep='\t')
    followings_df.to_csv(os.path.join(output_path, "network_data", "edges", "followings_ids.tsv"), index=False, sep='\t')


def create_followers_followings_dfs():
    with open(followers_data_path, "r", encoding='utf-8') as fin:
        all_lines = [x for x in fin.readlines()]
        followers = [x.split('"')[9] for x in all_lines]
        users = [x.split('"')[13] for x in all_lines]
    followers_df = pd.DataFrame({"source": followers, "dest": users})
    followers_df.to_csv(os.path.join(output_path, "network_data", "edges", "followers.tsv"), index=False, sep='\t')
    with open(followings_data_path, "r", encoding='utf-8') as fin:
        all_lines = [x for x in fin.readlines()]
        followings = [x.split('"')[9] for x in all_lines]
        users = [x.split('"')[13] for x in all_lines]
    followings_df = pd.DataFrame({"source": users, "dest": followings})
    followings_df.to_csv(os.path.join(output_path, "network_data", "edges", "followings.tsv"), index=False, sep='\t')

# read_followers_followings_dfs()
# with open(followers_data_path, "r", encoding='utf-8') as fin:
#     for i, line in enumerate(fin):
#         if i % 1000000 == 0:
#             print(i)
#         json_content = json.loads(line.encode().decode("utf-8"))
#         edge_tup = (json_content['follower'], json_content['user'])
#
#         if edge_tup not in follower_edges:
#             follower_edges.append(edge_tup)
#         else:
#             print(edge_tup)
#
# followers_df = pd.DataFrame({"source": followers, "dest": users})
# followers_df.to_csv(os.path.join(output_path, "network_data", "edges", "followers.tsv"), index=False, sep='\t')
#
#
# following_edges = []
# with open(followings_data_path, "r", encoding='utf-8') as fin:
#     print(f"followings has {len([line for line in fin])} lines...")
# with open(followings_data_path, "r", encoding='utf-8') as fin:
#     for i, line in enumerate(fin):
#         if i % 1000000 == 0:
#             print(i)
#         json_content = json.loads(line.encode().decode("utf-8"))
#
#         edge_tup = (json_content['user'], json_content['following'])
#         if edge_tup not in following_edges:
#             following_edges.append(edge_tup)
#         else:
#             print(edge_tup)
#
# followings_df = pd.DataFrame(follower_edges, columns=['source', 'dest'])
# followings_df.to_csv(os.path.join(output_path, "network_data", "edges", "followings.tsv"), index=False, sep='\t')


username2id_mapping = {}
with open(gab_data_path, "r", encoding='utf-8') as fin:
    for line in tqdm(fin):
        json_content = json.loads(line[:-1])
        json_type = json_content["type"]
        user_id_str = str(json_content["actuser"]["id"])
        username = json_content["actuser"]["username"]
        # if user_id_str in username2id_mapping.values() and username2id_mapping[username] != user_id_str:
        #     print(f"Different username for user id {user_id_str}. Previous was: {[k for k,v in username2id_mapping.items() if v == user_id_str][0]}; new is: {username}")
        if username not in username2id_mapping.keys():
            username2id_mapping[username] = user_id_str
        if json_type == 'repost':
            reposted_user_id = str(json_content['post']['user']['id'])
            reposted_username = json_content['post']['user']['username']
            # if reposted_user_id in username2id_mapping.values() and username2id_mapping[reposted_username] != reposted_user_id:
            #     print(f"Different username for user id {reposted_user_id}. Previous was: {[k for k,v in username2id_mapping.items() if v == reposted_user_id][0]}; new is: {reposted_username}")
            if reposted_username not in username2id_mapping.keys():
                username2id_mapping[reposted_username] = reposted_user_id
# username2id_mapping = dict(username2id_mapping)

with open(os.path.join(output_path, "pickled_data", "username2id_mapping_with_duplicates.pkl"), "wb") as fout:
    pickle.dump(username2id_mapping, fout)



# username2id_mapping_fn = os.path.join(base_path, "pickled_data", "username2id_mapping_with_duplicates.pkl")
#         username2id_mapping = pickle.load(open(username2id_mapping_fn, "rb"))
#         with open(data_path, "r", encoding='utf-8') as fin:
#             for line in tqdm(fin):
#                 json_content = json.loads(line[:-1])
#                 user_id_str = str(json_content["actuser"]["id"])
#                 json_type = json_content["type"]
#                 text = json_content['post']['body']
#                 if json_type == 'repost':  # like RT
#                     user_reposted_id = str(json_content['post']['user']['id'])
#                     if user_id_str != user_reposted_id:
#                         if user_reposted_id not in retweets_dict[user_id_str].keys():
#                             retweets_dict[user_id_str][user_reposted_id] = []
#                         retweets_dict[user_id_str][user_reposted_id].append(text)
#                 elif json_type == 'post':  # here take the mentions from text, as there is no field in gab API for it.
#                     text = URL_RE.sub('', text)
#                     mentions = [mention.replace('@', '') for mention in MENTION_RE.findall(text)]
#                     for mention in mentions:
#                         if mention in username2id_mapping.keys():
#                             user_mentioned_id = username2id_mapping[mention]
#                         else:
#                             user_mentioned_id = mention  # keep the username as is if no mapping to id
#                         if user_mentioned_id != user_id_str:
#                             if user_mentioned_id not in mentions_dict[user_id_str].keys():
#                                 mentions_dict[user_id_str][user_mentioned_id] = []
#                             mentions_dict[user_id_str][user_mentioned_id].append(text)
#                 text = URL_RE.sub('', text)
#                 if ignore_punct:
#                     text = PUNCT_RE.sub(' ', text)
#
#                 if ignore_rt:
#                     continue
#                 else:
#                     tweets_per_user[user_id_str].append(text)
#                     all_tweets.append(text)