def get_common_users_with_ira():
    ira_troll_ids = pd.read_csv("/data/work/data/sharp_power/canada_15/ira_can15_count.tsv", sep="\t").iloc[:, 0]
    ira_troll_ids = [str(id) for id in ira_troll_ids]
    # relevant_users = list(pd.read_csv("hate_networks/Echo networks/Edges/ira_users/mention_edges_df.csv")["source"])
    # relevant_users = [str(user_id) for user_id in relevant_users]
    trolls_in_our_data = []
    mention_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    retweet_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    in_reply_to_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    mentions_dict = defaultdict(dict)
    retweets_dict = defaultdict(dict)
    in_reply_to_dict = defaultdict(dict)

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
                        if user_id in ira_troll_ids:
                            trolls_in_our_data.append(user_id)
                        # if user_id in relevant_users: # only then continue
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
                        user_in_reply_to_id = lookup(json, 'in_reply_to_user_id_str')
                        if user_in_reply_to_id != None:
                            #if user_mentioned_id in ira_troll_ids and user_id != user_mentioned_id:  # self mentioning handling
                            if user_in_reply_to_id in ira_troll_ids and user_id != user_in_reply_to_id:
                                if user_in_reply_to_id not in in_reply_to_dict[user_id].keys():
                                    in_reply_to_dict[user_id][user_in_reply_to_id] = []
                                in_reply_to_dict[user_id][user_in_reply_to_id].append(text)

                        # retweets
                        if 'retweeted_status' in json.keys():
                            user_retweeted_id = lookup(json, 'retweeted_status.user.id_str')
                            if user_retweeted_id in ira_troll_ids and user_id != user_retweeted_id:
                            # if user_mentioned_id in ira_troll_ids and user_id != user_mentioned_id:  # self mentioning handling
                                if user_retweeted_id not in retweets_dict[user_id].keys():
                                    retweets_dict[user_id][user_retweeted_id] = []
                                retweets_dict[user_id][user_retweeted_id].append(text)

                        # mentions
                        for user_mentioned in user_mentions:
                            user_mentioned_id = lookup(user_mentioned, 'id_str')
                            if user_mentioned_id in ira_troll_ids and user_id != user_mentioned_id:  # self mentioning handling
                                if user_mentioned_id not in mentions_dict[user_id].keys():
                                    mentions_dict[user_id][user_mentioned_id] = []
                                mentions_dict[user_id][user_mentioned_id].append(text)
                    # if len(mentions_dict) > 10:
                    #     break
    print("Before creating dfs")
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
    print("Before saving dfs and dicts")
    date = datetime.today().strftime('%Y-%m-%d')
    mention_edges.to_csv('hate_networks/Echo networks/Edges/ira_users/mention_edges_with_text_df_' + date + '.tsv', sep='\t', index=False)
    in_reply_to_edges.to_csv('hate_networks/Echo networks/Edges/ira_users/in_reply_to_edges_with_text_df_' + date + '.tsv', sep='\t', index=False)
    retweet_edges.to_csv('hate_networks/Echo networks/Edges/ira_users/retweet_edges_with_text_df_' + date + '.tsv', sep='\t', index=False)
    with open('hate_networks/Echo networks/Edges/ira_users/in_reply_to_dict_with_text_df_' + date + '.pkl', "wb") as fout:
        pickle.dump(in_reply_to_dict, fout)
    with open('hate_networks/Echo networks/Edges/ira_users/mentions_dict_with_text_df_' + date + '.pkl', "wb") as fout:
        pickle.dump(mentions_dict, fout)
    with open('hate_networks/Echo networks/Edges/ira_users/retweets_dict_with_text_df_' + date + '.pkl', "wb") as fout:
        pickle.dump(retweets_dict, fout)
    with open('hate_networks/Echo networks/Edges/ira_users/trolls_in_our_data_' + date + '.pkl', "wb") as fout:
        pickle.dump(trolls_in_our_data, fout)
