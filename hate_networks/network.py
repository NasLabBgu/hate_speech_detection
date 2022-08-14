import networkx as nx
import igraph
from hate_networks.general import *
from hate_networks.models_validation import get_annotation_scores
from config.hate_networks_config import general_conf, path_conf

import logging
logger = logging.getLogger(__name__)


def create_network_nodes_and_edges(users_list):
    '''
    A function that given a user list creates edges of mentions, replies, ReTweets and quotes between the users
    :param users_list:
    :return:
    '''
    nodes = pd.DataFrame(columns=['user_id', 'screen_name', 'verified', 'location', 'language', 'followers_count',
                                  'statuses_count', 'friends_count'])
    mention_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    retweet_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    quote_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    in_reply_to_edges = pd.DataFrame(columns=['source', 'dest', 'weight'])
    mentions_dict = defaultdict(dict)
    in_reply_to_dict = defaultdict(dict)
    retweets_dict = defaultdict(dict)
    quotes_dict = defaultdict(dict)
    with zipfile.ZipFile("/data/home/eyalar/antisemite_hashtags/Resources/recent_history.zip", "r") as zfile:
        for name in tqdm(zfile.namelist()):
            # We have a zip within a zip
            if re.search('\.gz$', name) != None:
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json = simplejson.loads(line)
                        user = lookup(json, 'user')
                        user_id = lookup(user, 'id')
                        user_mentions = lookup(json, 'entities.user_mentions')
                        if user_id not in list(nodes['user_id']):
                            nodes = nodes.append({'user_id': user_id, 'screen_name': lookup(user, 'screen_name'),
                                                  'verified': lookup(user, 'verified'),
                                                  'location': lookup(user, 'location'),
                                                  'language': lookup(user, 'lang'),
                                                  'followers_count': lookup(user, 'followers_count'),
                                                  'statuses_count': lookup(user, 'statuses_count'),
                                                  'friends_count': lookup(user, 'friends_count')}, ignore_index=True)
                        # if user_id not in nodes:
                        #   node.append(user)
                        # exists = False
                        # for node in nodes:
                        #   if user_id == node.id:
                        #     exists = True
                        # if not exists:
                        # create user and add it to network nodes
                        # new_user = User(user_id, self.lookup(user,'name'),self.lookup(user,'verified'),self.lookup(user,'location'),self.lookup(user,'description'),self.lookup(user,'followers_count'),self.lookup(user,'statuses_count'), self.lookup(user, 'friends_count'))
                        # nodes.append(new_user)
                        # create hashtags for the tweet and add it to network hashtags
                        # hashtags = self.lookup(json, 'entities.hashtags')
                        # created_at = self.lookup(json, 'created_at')
                        # if first == True:
                        #   self.min_tweet_time = created_at
                        #   self.max_tweet_time= created_at
                        #   first = False
                        # else:
                        #   min_compare = self.compare_dates(self.min_tweet_time, created_at)
                        #   max_compare = self.compare_dates(self.max_tweet_time, created_at)
                        #   if min_compare == -1:
                        #     self.min_tweet_time = created_at
                        #   if max_compare == 1:
                        #     self.max_tweet_time = created_at
                        # for hashtag in hashtags:
                        #   text = hashtag['text']
                        #   exists = False
                        #   for hash in self.hashtags:
                        #     if text == hash.text:
                        #       hash.time_stamps.append(created_at)
                        #       hash.count += 1
                        #       exists = True
                        #   if not exists:
                        #     time_stamps = [created_at]
                        #     new_hashtag = Hashtag(text, 1, time_stamps)
                        #     self.hashtags.append(new_hashtag)

                        # creating edges for the network (20 mentions and above)

                        # in reply to
                        user_mentioned_id = lookup(json, 'in_reply_to_user_id')
                        if user_mentioned_id != None:
                            if user_mentioned_id in users_list and user_id != user_mentioned_id:  # self mentioning handling
                                if user_mentioned_id in in_reply_to_dict[user_id].keys():
                                    in_reply_to_dict[user_id][user_mentioned_id] += 1
                                else:
                                    in_reply_to_dict[user_id][user_mentioned_id] = 0
                        if 'retweeted_status' in json.keys() and 'quoted_status' in json.keys():
                            logger.info("RT AND QUOTE!")

                        # retweets
                        if 'retweeted_status' in json.keys():
                            user_mentioned_id = lookup(json, 'retweeted_status.user.id')
                            if user_mentioned_id in users_list and user_id != user_mentioned_id:  # self mentioning handling
                                if user_mentioned_id in retweets_dict[user_id].keys():
                                    retweets_dict[user_id][user_mentioned_id] += 1
                                else:
                                    retweets_dict[user_id][user_mentioned_id] = 0

                        # quotes
                        if 'quoted_status' in json.keys():
                            user_mentioned_id = lookup(json, 'quoted_status.user.id')
                            if user_mentioned_id in users_list and user_id != user_mentioned_id:  # self mentioning handling
                                if user_mentioned_id in quotes_dict[user_id].keys():
                                    quotes_dict[user_id][user_mentioned_id] += 1
                                else:
                                    quotes_dict[user_id][user_mentioned_id] = 0

                        # mentions
                        for user_mentioned in user_mentions:
                            user_mentioned_id = lookup(user_mentioned, 'id')
                            if user_mentioned_id in users_list and user_id != user_mentioned_id:  # self mentioning handling
                                if user_mentioned_id in mentions_dict[user_id].keys():
                                    mentions_dict[user_id][user_mentioned_id] += 1
                                else:
                                    mentions_dict[user_id][user_mentioned_id] = 0
                zf.close()
    zfile.close()
    # in reply to
    for user_id, mentioned_user in in_reply_to_dict.items():
        for mentioned_user_id, mention_count in mentioned_user.items():
            in_reply_to_edges = in_reply_to_edges.append(
                {'source': user_id, 'dest': mentioned_user_id, 'weight': mention_count + 1}, ignore_index=True)

    # retweets
    for user_id, mentioned_user in retweets_dict.items():
        for mentioned_user_id, mention_count in mentioned_user.items():
            retweet_edges = retweet_edges.append(
                {'source': user_id, 'dest': mentioned_user_id, 'weight': mention_count + 1}, ignore_index=True)

    # quotes
    for user_id, mentioned_user in quotes_dict.items():
        for mentioned_user_id, mention_count in mentioned_user.items():
            quote_edges = quote_edges.append(
                {'source': user_id, 'dest': mentioned_user_id, 'weight': mention_count + 1}, ignore_index=True)

    # mentions
    for user_id, mentioned_user in mentions_dict.items():
        for mentioned_user_id, mention_count in mentioned_user.items():
            mention_edges = mention_edges.append(
                {'source': user_id, 'dest': mentioned_user_id, 'weight': mention_count + 1}, ignore_index=True)

    nodes.to_csv('hate_networks/node_df.csv', index=False)

    mention_edges.to_csv('hate_networks/mention_edges_df.csv', index=False)
    in_reply_to_edges.to_csv('hate_networks/in_reply_to_edges_df.csv', index=False)
    retweet_edges.to_csv('hate_networks/retweet_edges_df.csv', index=False)
    quote_edges.to_csv('hate_networks/quote_edges_df.csv', index=False)

    return nodes, in_reply_to_edges


# Constructing & plotting the network
def get_overlapped_clusters(merged_users_topic_cluster_df, cluster_name1, cluster_name2, K):
    """
    Transform the cluster of w2v to the same numerical value of the topic by observing number of mutual users in each
    of the clusters.
    :param merged_users_topic_cluster_df: DataFrame object with users ids, user topic and word2vec cluster.
    :param K: The number of topics or clusters.
    :return: transformed DataFrame.
    """
    topic_list = list(range(K))
    w2v_list = list(range(K))
    cluster_to_topic = defaultdict(int)
    tm_users_dict = defaultdict(list)
    # w2v_users_clusters = defaultdict(list)

    for i in range(K):
        current_tm_df = merged_users_topic_cluster_df[merged_users_topic_cluster_df[cluster_name1] == i]
        # current_w2v_df = merged_users_topic_cluster_df[merged_users_topic_cluster_df['cluster'] == i]
        tm_users_dict[i] = Counter(current_tm_df[cluster_name2])
    for val in tm_users_dict.values():
        if len(list(val)) != K:
            raise Exception("There is no overlapping instances for all clusters and topics")
    # naive option
    # for topic, cnt in tm_users_dict.items():
    #   tm_users_dict[topic] = cnt.most_common(1)[0]

    # better option
    cur_max_cluster = -1
    cur_topic = -1
    while len(topic_list) > 0:
        cur_max_count = -1
        for topic, cnt in tm_users_dict.items():
            # check that the topic was not assigned already
            if topic in topic_list:
                for cluster in cnt.most_common(K):
                    cur_cluster = cluster[0]
                    cur_count = cluster[1]
                    # check if higher count and if w2v cluster not assigned already
                    if cur_cluster in w2v_list:
                        if cur_count > cur_max_count:
                            cur_max_count = cur_count
                            cur_max_cluster = cur_cluster
                            cur_topic = topic
        cluster_to_topic[cur_max_cluster] = cur_topic
        topic_list.remove(cur_topic)
        w2v_list.remove(cur_max_cluster)
    merged_users_topic_cluster_df["new_" + cluster_name2] = -1
    for cluster, topic in cluster_to_topic.items():
        merged_users_topic_cluster_df.loc[
            merged_users_topic_cluster_df[cluster_name2] == cluster, "new_" + cluster_name2] = topic
    merged_users_topic_cluster_df[cluster_name2] = merged_users_topic_cluster_df["new_" + cluster_name2]
    merged_users_topic_cluster_df = merged_users_topic_cluster_df.drop(["new_" + cluster_name2], axis=1)
    return merged_users_topic_cluster_df


def export_legend(legend, filename, expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def triangles(g):
    cliques = g.cliques(min=3, max=3)
    num_of_cliques = len(cliques)
    result = {v_id: 0 for v_id in range(len(g.vs))}
    for i, j, k in cliques:
        result[i] += 1
        result[j] += 1
        result[k] += 1
    max_num_of_cliques = max(list(result.values()))
    return result, num_of_cliques, max_num_of_cliques


def log_network_features(g, g_name):
    logger.info(f"~~~~~~~~~{g_name} properties~~~~~~~~~~")
    logger.info(f"Number of vertices: {len(g.vs)}")
    logger.info(f"Number of edges: {len(g.es)}")

    # density, diameter, triangles
    density = g.density(loops=False)
    diameter = g.diameter(directed=True)
    _, num_of_cliques, max_num_of_cliques = triangles(g)
    logger.info(f"Density: {density:.4f}")
    logger.info(f"Diameter {diameter}")
    logger.info(f"Number of cliques: {num_of_cliques}")
    logger.info(f"Max number of cliques for a node: {max_num_of_cliques}")
    # number of CC
    strong_cc = g.components(mode="STRONG")
    weak_cc = g.components(mode="WEAK")
    logger.info(f"Strongly connected components: {len(strong_cc)} - ({len(strong_cc) / len(g.vs):.2f}% node normalized),"
                f" ({len(strong_cc) / len(g.es):.2f}% edge normalized)")
    logger.info(f"Weakly connected components: {len(weak_cc)} - ({len(weak_cc) / len(g.vs):.2f}% node normalized),"
                f" ({len(weak_cc) / len(g.es):.2f}% edge normalized)")
    logger.info(f"\n")


def compute_communities_feautres(g):

    log_network_features(g, "Full graph")
    largest_cc = g.clusters(mode="STRONG").giant()
    log_network_features(largest_cc, "Largest connected component")
    network_type = general_conf['inference_network_type']

    if network_type == 'echo':
        predicted_users_df = pd.read_csv(
            "/data/work/data/echoes/classifier_data/results_unlabeld_full_words_last2epochs.csv")  # user_id, y_pred, y_pred_class
        r_n_ids = [str(x) for x in list(predicted_users_df.loc[predicted_users_df["y_pred_class"] == 0, "user_id"])]
        hm_ids = [str(x) for x in list(predicted_users_df.loc[predicted_users_df["y_pred_class"] == 1, "user_id"])]
        labeled_users_df = pd.read_csv("/data/work/data/echoes/classifier_data/users_labeled.csv")
        r_n_ids.extend([str(x) for x in list(labeled_users_df.loc[labeled_users_df["label"].isin([0, 2]), "user_id"])])
        hm_ids.extend([str(x) for x in list(labeled_users_df.loc[labeled_users_df["label"].isin([1]), "user_id"])])
    elif network_type == 'gab':
        pass
    hm_g = g.subgraph([v for v in g.vs if v['label'] in hm_ids])
    r_n_g = g.subgraph([v for v in g.vs if v['label'] in r_n_ids])

    lcc_intersection_hm = len(
        set([v['label'] for v in largest_cc.vs]).intersection(set([v['label'] for v in hm_g.vs])))
    lcc_intersection_r_n = len(
        set([v['label'] for v in largest_cc.vs]).intersection(set([v['label'] for v in r_n_g.vs])))
    log_network_features(hm_g, "Hatemongers sub-graph")
    log_network_features(r_n_g, "Responders and neutral sub-graph")

    logger.info(f"There are {lcc_intersection_hm} mutual nodes between LCC and HM")
    logger.info(f"There are {lcc_intersection_r_n} mutual nodes between LCC and R+N")


def compute_network_features(g, edge_type, keep_all_edges, keep_all_nodes, base_path):
    """
        Given an edge df and users with their topic model and word2vec clusters df, create two directed igraph graphs where
        nodes are the users colored by the cluster they have been assigned (both by topic model and word2vec) and
        the edges are taken from the edge_df.
        :param g: graph to compute on the nodes centralities
        :param edge_df_path: path to DataFrame of the edges between the users (can be created using mentions, retweets,
         replies or quotes).
        :param edge_type: The type of the edges in edge_df {"mentions", "retweets", "replies", "quotes"}.
        :param users_dist_df_path: path to DataFrame with topic (using topic model) and cluster (using word2vec) of each user.
        :return: None
        """
    logger.info(f"Keep all edges: {keep_all_edges}")
    logger.info(f"Keep all nodes: {keep_all_nodes}")
    betweenness_centrality = g.betweenness()
    all_degree_values = []
    out_degree_values = []
    in_degree_values = []
    strong_cc_sizes = []
    weak_cc_sizes = []
    strong_cc = g.components(mode="STRONG")
    weak_cc = g.components(mode="WEAK")
    for v in g.vs:
        all_degree_centrality = g.maxdegree(vertices=v.index, mode="ALL")
        out_degree_centrality = g.maxdegree(vertices=v.index, mode="IN")
        in_degree_centrality = g.maxdegree(vertices=v.index, mode="OUT")
        all_degree_values.append(all_degree_centrality)
        out_degree_values.append(out_degree_centrality)
        in_degree_values.append(in_degree_centrality)
        for cc in strong_cc:
            if v.index in cc:
                strong_cc_sizes.append(len(cc))
                break
        for cc in weak_cc:
            if v.index in cc:
                weak_cc_sizes.append(len(cc))
                break

    closeness_centrality = g.closeness()
    page_rank_centrality = g.personalized_pagerank()
    eigenvector_centrality = g.eigenvector_centrality()

    triangle_num_per_user, _, _ = triangles(g)
    triangles_count = list(triangle_num_per_user.values())

    centralities_df = pd.DataFrame({"user_id": [str(x) for x in list(g.vs["label"])], "betweenness": betweenness_centrality, "all_degree": all_degree_values,
                                    "out_degree": out_degree_values, "in_degree": in_degree_values,
                                    "closeness": closeness_centrality, "pagerank": page_rank_centrality,
                                    "eigenvector": eigenvector_centrality, "triangles_count": triangles_count,
                                    "strong_cc_size": strong_cc_sizes, "weak_cc_size": weak_cc_sizes})

    #todo: possible to add additional shallow features (like number of followers, statuses, etc.)
    ## users_df = pd.read_csv("hate_networks/Echo networks/csv_data/node_df.csv")
    # centralities_df = pd.merge(centralities_df, users_df[["user_id", "screen_name", "verified", "followers_count"]], on="user_id")

    fn_suffix = "all_edges_" if keep_all_edges else "edges_filtered_"
    fn_suffix += "all_nodes" if keep_all_nodes else "singletons_filtered"
    features_dir_path = os.path.join(base_path, "network_data", "features")
    create_dir_if_missing(features_dir_path)
    centralities_df.to_csv(os.path.join(features_dir_path, f"centralities_{edge_type}_{fn_suffix}.tsv"), sep='\t', index=False)


def construct_network(K, nodes_dict, users_dist_df, min_edge_weight, cluster_name, edge_df, layout_type, keep_all_edges=False, keep_all_nodes=False):
    pal = igraph.drawing.colors.ClusterColoringPalette(K)
    g = igraph.Graph(directed=True)
    g.add_vertices(range(len(nodes_dict)))
    g.vs["label"] = [str(user_id) for user_id in list(nodes_dict.keys())]

    # color nodes (users) by their topic model (salient topic / full topic dist) or word2vec representations
    g.vs['color'] = pal.get_many(list(users_dist_df[cluster_name]))
    if keep_all_edges:
        relevant_edges = edge_df
    else:
        relevant_edges = edge_df[edge_df['weight'] >= min_edge_weight].reset_index(drop=True)
    sources = list(relevant_edges['source'])
    destinations = list(relevant_edges['dest'])
    edges = [(nodes_dict[src][0], nodes_dict[dest][0]) for src, dest in zip(sources, destinations) if src in nodes_dict.keys() and dest in nodes_dict.keys()]
    g.add_edges(edges)
    weights = list(relevant_edges['weight'])
    g.es["color"] = "gray"
    g.es["weight"] = weights
    if not keep_all_nodes:
        to_delete_ids = [v.index for v in g.vs if v.degree() == 0]
        logger.info(f"Deleting {len(to_delete_ids)} singletons from the graph")
        g.delete_vertices(to_delete_ids)

    plot_unsupervised_networks = general_conf["plot_unsupervised_networks"]
    plot_supervised_networks = general_conf["plot_supervised_networks"]

    visual_style = {}
    if plot_unsupervised_networks or plot_supervised_networks:
        layout = g.layout(layout_type)
        visual_style["layout"] = layout  #g.layout_fruchterman_reingold(weights=g.es["weight"], maxiter=1000, area=N**3, repulserad=N**3)
        visual_style["label_size"] = 0.5
        visual_style["vertex_size"] = 4  # [a/5 for a in g.degree()]
        visual_style["edge_arrow_width"] = 0.2
        visual_style["edge_arrow_size"] = 0.2
        visual_style["edge_color"] = "gray"
        visual_style["edge_width"] = 2
        visual_style["edge_curved"] = False
    return g, visual_style


def create_network(edge_type, users_dist_df, K, min_edge_weight=3, layout_type="fr", with_labels=False,
                   output_fig_names=None, base_path=None):
    """
    Given an edge df and users with their topic model and word2vec clusters df, create two directed igraph graphs where
    nodes are the users colored by the cluster they have been assigned (both by topic model and word2vec) and
    the edges are taken from the edge_df.
    :param node_df:
    :param edge_df_path: path to DataFrame of the edges between the users (can be created using mentions, retweets,
     replies or quotes).
    :param edge_type: The type of the edges in edge_df {"mentions", "retweets", "replies", "quotes"}.
    :param users_dist_df_path: path to DataFrame with topic (using topic model) and cluster (using word2vec) of each user.
    :return: None
    """
    logger.info("creating networks...")
    current_edge_file_path = os.path.join(base_path, "network_data", "edges", f"data_users_{edge_type}_edges_df.tsv")
    edge_df = pd.read_csv(current_edge_file_path, sep='\t')
    edge_df['source'] = edge_df['source'].astype(str)
    edge_df['dest'] = edge_df['dest'].astype(str)
    users_dist_df['user_id'] = users_dist_df['user_id'].astype(str)

    nodes_dict = defaultdict(tuple)
    counter = 0
    for index, row in users_dist_df.iterrows():
        nodes_dict[row['user_id']] = (counter, row['user_max_topic'], row['w2v_cluster'], row["tm_dist_cluster"])
        counter += 1

    file_type = general_conf["network_file_type"]
    tm_max_topic_graph_file_name = f"{output_fig_names['tm_max_topic']}_{str(min_edge_weight)}_min_edge_weight_{layout_type}.{file_type}"
    w2v_graph_file_name = f"{output_fig_names['w2v']}_{str(min_edge_weight)}_min_edge_weight_{layout_type}.{file_type}"
    tm_dist_graph_file_name = f"{output_fig_names['tm_dist_cluster']}_{str(min_edge_weight)}_min_edge_weight_{layout_type}.{file_type}"
    pred_graph_file_name = f"{output_fig_names['pred']}_{str(min_edge_weight)}_min_edge_weight_{layout_type}.{file_type}"

    plots_dir_path = os.path.join(base_path, "plots")
    edge_type_plot_dir_path = os.path.join(plots_dir_path, edge_type)
    create_dir_if_missing(edge_type_plot_dir_path)
    for modeling_type in ["tm_max_topic", "w2v", "tm_dist", "pred"]:
        create_dir_if_missing(os.path.join(edge_type_plot_dir_path, modeling_type))

    keep_all_edges = general_conf["keep_all_edges"]
    keep_all_nodes = general_conf["keep_all_nodes"]
    plot_unsupervised_networks = general_conf["plot_unsupervised_networks"]
    plot_supervised_networks = general_conf["plot_supervised_networks"]
    extract_network_features = general_conf["extract_network_features"]

    # coloring using the unsupervised methods
    if plot_unsupervised_networks:
        g_tm_max_topic, tm_max_topic_visual_style = construct_network(K, nodes_dict, users_dist_df, min_edge_weight,
                                                "user_max_topic", edge_df, layout_type, keep_all_edges, keep_all_nodes)
        g_w2v_clustering, w2v_clustering_visual_style = construct_network(K, nodes_dict, users_dist_df, min_edge_weight,
                                                "w2v_cluster", edge_df, layout_type, keep_all_edges, keep_all_nodes)
        g_tm_dist, tm_dist_visual_style = construct_network(K, nodes_dict, users_dist_df, min_edge_weight,
                                                "tm_dist_cluster", edge_df, layout_type, keep_all_edges, keep_all_nodes)

        logger.info(f"saving network figures to {edge_type_plot_dir_path}...")
        g_tm_max_topic_copy = g_tm_max_topic.copy()
        g_w2v_clustering_copy = g_w2v_clustering.copy()
        g_tm_dist_copy = g_tm_dist.copy()
        if not with_labels:
            g_tm_max_topic_copy.vs['label'] = [''] * len(g_tm_max_topic_copy.vs['label'])
            g_w2v_clustering_copy.vs['label'] = [''] * len(g_tm_max_topic_copy.vs['label'])
            g_tm_dist_copy.vs['label'] = [''] * len(g_tm_max_topic_copy.vs['label'])
        igraph.plot(obj=g_tm_max_topic_copy, target=os.path.join(edge_type_plot_dir_path, "tm_max_topic", tm_max_topic_graph_file_name),
                    bbox=(0, 0, 1000, 1000), **tm_max_topic_visual_style)
        igraph.plot(obj=g_w2v_clustering_copy, target=os.path.join(edge_type_plot_dir_path, "w2v", w2v_graph_file_name),
                    bbox=(0, 0, 1000, 1000), **w2v_clustering_visual_style)
        igraph.plot(obj=g_tm_dist_copy, target=os.path.join(edge_type_plot_dir_path, "tm_dist", tm_dist_graph_file_name),
                    bbox=(0, 0, 1000, 1000), **tm_dist_visual_style)
        logger.info(f"Finished saving network figures using unsupervised methods to color users.")

    else:
        g_tm_max_topic, _ = construct_network(K, nodes_dict, users_dist_df, min_edge_weight,
                                                "user_max_topic", edge_df, layout_type, keep_all_edges, keep_all_nodes)

    # coloring the network according to predictions (from file set in config)
    if plot_supervised_networks:
        labeled_users_df = pd.read_csv(path_conf[general_conf['inference_network_type']]['user_preds'])
        K = labeled_users_df['label'].nunique()
        g_pred, pred_visual_style = construct_network(K, nodes_dict, labeled_users_df, min_edge_weight,
        "label", edge_df, layout_type, keep_all_edges, keep_all_nodes)
        if not with_labels:
            g_pred.vs['label'] = [''] * len(g_pred.vs['label'])
        igraph.plot(obj=g_pred,
                    target=os.path.join(edge_type_plot_dir_path, "pred", pred_graph_file_name),
                    bbox=(0, 0, 1000, 1000), **pred_visual_style)
        logger.info(f"Finished saving network figures using provided predictions to color users.")

    if extract_network_features:
        logger.info(f"extracting network features...")
        compute_communities_feautres(g=g_tm_max_topic)
        compute_network_features(g_tm_max_topic, edge_type, keep_all_edges, keep_all_nodes, base_path)


def igraph2networkx(g):
    A = g.get_edgelist()
    G = nx.DiGraph(A)  # In case your graph is directed
    return G


def community_detection(edge_type, users_with_labels_df):
    users_df = pd.read_csv('hate_networks/Echo networks/csv_data/users.csv')
    # labeled_users = list(users_with_labels_df["user_id"])
    # users_df = users_df[users_df["user_id"].apply(lambda ui: ui in labeled_users)]

    edge_df = pd.read_csv("hate_networks/Echo networks/Edges/" + edge_type + "_edges_df.csv")
    layout_type = "fr"
    min_edge_weight = 3
    # pal = igraph.drawing.colors.ClusterColoringPalette(K)
    # color_dict = {0: "blue", 1: "red", 2: "green", 3: "pink", 4: "yellow"}

    nodes_dict = defaultdict(tuple)
    counter = 0
    for index, row in users_df.iterrows():
        nodes_dict[row['user_id']] = counter
        counter += 1

    file_type = "png"
    community_detection_graph_file_name = "community_detection_" + edge_type + "_" + str(
        min_edge_weight) + "min_edge_weight_" + layout_type + "_layout." + file_type

    g = igraph.Graph(directed=True)
    g.add_vertices(range(len(nodes_dict)))
    # if with_labels:
    #     g.vs["label"] = [str(user_id) for user_id in list(nodes_dict.keys())]
    #     g.vs["label_size"] = 0.5
    # color nodes (users) by their topic model (or w2v) dist (or clustering) representations
    # g.vs['color'] = pal.get_many(list(users_dist_df['label']))

    edges = []
    weights = []
    for index, row in edge_df.iterrows():
        current_weight = row['weight']
        if current_weight >= min_edge_weight:
            if current_weight >= min_edge_weight and row['source'] in nodes_dict.keys() and row['dest'] in nodes_dict.keys():
                try:
                    edges.append((nodes_dict[row['source']], nodes_dict[row['dest']]))
                    weights.append(current_weight)
                except:  # at least one of the nodes are not labeled
                    continue

    g.add_edges(edges)
    to_delete_ids = [v.index for v in g.vs if v.degree() == 0]
    logger.info("{0} has {1} singletons".format(edge_type, len(to_delete_ids)))
    g.delete_vertices(to_delete_ids)
    logger.info("Number of Vertices: {}".format(len(g.vs)))
    logger.info("Number of Edges: {}".format(len(g.es)))
    g["weight"] = weights

    v_clust = g.community_leading_eigenvector(clusters=3)
    comunities = v_clust.subgraphs()
    # largest_c = g.clusters().giant()
    # v_clust = largest_c.community_leading_eigenvector(clusters=3)
    # comunities = v_clust.subgraphs()

    logger.info(f'Vertex clustering sizes: {[len(i.vs) for i in comunities]}')
    logger.info(f'summary {v_clust.summary()}')
    logger.info(f'rnd_g[v_clust[0]]:  {g.vs[v_clust[0]]}')
    g.vs[v_clust[1]]['color'] = '#ff605a'
    for i in range(len(v_clust)):
        if i != 1:
            g.vs[v_clust[i]]['color'] = '#5AA5FF'
    # communities_generator = community.girvan_newman(G)
    # top_level_communities = next(communities_generator)
    # next_level_communities = next(communities_generator)
    # next_comm = sorted(map(sorted, next_level_communities))

    # dendrogram = generate_dendrogram(G)
    # for level in range(len(dendrogram) - 1):
    #     print("partition at level", level, "is", partition_at_level(dendrogram, level))  # NOQA
    # g.simplify()
    # calculate dendrogram
    # dendrogram = g.community_edge_betweenness()
    # dendrogram = g.community_leading_eigenvector(weights=g.es["weight"], clusters=3)
    # dendrogram = g.community_edge_betweenness(clusters=3, directed=False)
    # num_communities = dendrogram.optimal_count
    # print(num_communities)
    # convert it into a flat clustering
    # clusters = dendrogram.as_clustering(3)
    # get the membership vector
    # membership = clusters.membership
    # K = len(set(clusters.membership))
    # pal = igraph.drawing.colors.PrecalculatedPalette(["green", "red", "blue"])
    # pal = igraph.drawing.colors.ClusterColoringPalette(K)

    # g.vs['color'] = pal.get_many(list(clusters.membership))


    layout = g.layout(layout_type)

    visual_style = {}
    visual_style["layout"] = layout
    visual_style["vertex_size"] = 4
    visual_style["edge_arrow_width"] = 0.3
    # visual_style["label_size"] = 1
    visual_style["edge_arrow_size"] = 0.3

    # g_tm_dist, tm_dist_visual_style = construct_network(K, nodes_dict, users_dist_df, min_edge_weight, "user_topic", edge_df, layout_type)
    logger.info("BEFORE SAVING FIGS")
    igraph.plot(g, "hate_networks/Echo networks/plots/community_detection/" + community_detection_graph_file_name, **visual_style)
    logger.info("AFTER SAVING FIGS")
    invs_nodes_dict = {v: k for k, v in nodes_dict.items()}
    cluster_df = pd.DataFrame(columns=["user_id", "community"])
    for v in g.vs[v_clust[1]]:
        cluster_df = cluster_df.append({"user_id": invs_nodes_dict[v.index], "community":1}, ignore_index=True)
    for i in range(len(v_clust)):
        if i != 1:
            for v in g.vs[v_clust[i]]:
                cluster_df = cluster_df.append({"user_id": invs_nodes_dict[v.index], "community":0}, ignore_index=True)

    labeles_with_community_df = users_with_labels_df.merge(cluster_df, on="user_id")
    tm_ri, tm_ari = get_annotation_scores(list(labeles_with_community_df["label"]),
                                          list(labeles_with_community_df["community"]))

    logger.info("Topic model clustering has rand index of {0} and adjusted rand index of {1}".format(tm_ri, tm_ari))


def plot_graph_with_labels(edge_type, users_dist_df_path, with_labels=False):

    # users_df = pd.read_csv('hate_networks/Echo networks/csv_data/users.csv')
    # # users_df = pd.read_csv("./hate_networks/Echo networks/permanent_users_df.csv")
    # # users_df['user_id'] = users_df['user_id'].astype(str)
    # with open('hate_networks/Echo networks/pickled_data/users_tweets_corpora_2.pkl', 'rb') as f:
    #     tweets_corpora = pickle.load(f)
    #
    # run_topic_model(tweets_corpora=tweets_corpora, users_df=users_df, topic_num=K, feature_num=tm_feature_size)
    edge_df = pd.read_csv("hate_networks/Echo networks/Edges/" + edge_type + "_edges_df.csv")
    users_dist_df = pd.read_csv(users_dist_df_path)
    layout_type ="fr"
    min_edge_weight = 0
    # pal = igraph.drawing.colors.ClusterColoringPalette(K)
    # color_dict = {0: "blue", 1: "red", 2: "green", 3: "pink", 4: "yellow"}
    nodes_dict = defaultdict(tuple)
    counter = 0
    for index, row in users_dist_df.iterrows():
        nodes_dict[row['user_id']] = (counter, int(row['label']))
        counter += 1

    file_type = "png"
    labels_graph_file_name = "labeled_graph_with_singletons_" + edge_type + "_" + str(min_edge_weight) + "min_edge_weight_" + layout_type + "_layout." + file_type
    pal = igraph.drawing.colors.PrecalculatedPalette(["blue", "red", "green"])
    g = igraph.Graph(directed=True)
    g.add_vertices(range(len(nodes_dict)))
    if with_labels:
        g.vs["label"] = [str(user_id) for user_id in list(nodes_dict.keys())]
        g.vs["label_size"] = 0.5
    # color nodes (users) by their topic model (or w2v) dist (or clustering) representations
    g.vs['color'] = pal.get_many(list(users_dist_df['label']))

    edges = []
    weights = []
    for index, row in edge_df.iterrows():
        current_weight = row['weight']
        if current_weight >= min_edge_weight:
            try:
                edges.append((nodes_dict[row['source']][0], nodes_dict[row['dest']][0]))
                weights.append(current_weight)
            except:  # at least one of the nodes are not labeled
                continue

    g.add_edges(edges)
    # to_delete_ids = [v.index for v in g.vs if v.degree() == 0]
    # print("{0} has {1} singletons".format(edge_type, len(to_delete_ids)))
    # g.delete_vertices(to_delete_ids)
    logger.info("Number of Vertices: {}".format(len(g.vs)))
    logger.info("Number of Edges: {}".format(len(g.es)))
    g["weight"] = weights

    layout = g.layout(layout_type)

    visual_style = {}
    visual_style["layout"] = layout
    visual_style["vertex_size"] = 8
    visual_style["edge_arrow_width"] = 0.3
    # visual_style["label_size"] = 1
    visual_style["edge_arrow_size"] = 0.3


    # g_tm_dist, tm_dist_visual_style = construct_network(K, nodes_dict, users_dist_df, min_edge_weight, "user_topic", edge_df, layout_type)
    logger.info("BEFORE SAVING FIGS")
    igraph.plot(g, "hate_networks/Echo networks/plots/" + labels_graph_file_name, **visual_style)
    logger.info("AFTER SAVING FIGS")
