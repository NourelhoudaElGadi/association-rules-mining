import logging
from assocrulext.utils.timing import timeit
import pandas as pd

import numpy as np


from scipy.cluster.hierarchy import linkage, fcluster
import scipy.linalg.blas


from tqdm import tqdm


from assocrulext.ml.autoencoder import AutoEncoderDimensionReduction
from assocrulext.clustering.hac import *
from assocrulext.clustering.community import *
from assocrulext.clustering.cooc import *
from assocrulext.rules.fitering import delete_redundant_group
from assocrulext.eval import *
from assocrulext.utils.data import *
from assocrulext.rules.fpgrowth_algo import fp_growth_per_cluster
logger = logging.getLogger(__name__)


# @timeit
# def fp_growth_with_com_auto(one_hot, group, index, max_len, min_confidence):
#     """
#     Apply FP-Growth algorithm and generate rules to each cluster.

#     Parameters :
#         one_hot : DataFrame
#             One hot encoding DataFrame (rows : articles, columns : labels)
#         group : DataFrame
#             A DataFrame assigning a group for each article
#         index : list of int
#             Index of group
#         max_len : int
#             Maximum length of rules (e.g : 3 -> 2 antecedents and 1 consequents or 1 antecedents and 2 consequents)
#         min_confidence : float
#             Minimum confidence i.e the probability to have B when A occurs.

#     Returns :
#         rules_fp_clustering : list of DataFrame
#             List of Association rules DataFrame
#     """

#     rules_fp_clustering = []

#     for i in index:
#         one_hot_cluster = one_hot.loc[
#             :, one_hot.columns.isin(list(group[group.index == i]["Labels"]))
#         ]
#         frequent_itemsets_fp = fpgrowth(
#             one_hot_cluster,
#             min_support=5 / one_hot_cluster.shape[0],
#             max_len=max_len,
#             use_colnames=True,
#         )
#         if len(frequent_itemsets_fp) != 0:
#             rules_fp_clustering.append(
#                 association_rules(
#                     frequent_itemsets_fp,
#                     metric="confidence",
#                     min_threshold=min_confidence,
#                 ).sort_values(by="lift", ascending=False)
#             )
#         else:
#             rules_fp_clustering.append(pd.DataFrame([]))
#     return rules_fp_clustering


# def interestingness_measure_com_auto(rules_fp_clustering, one_hot, group, index):
#     """
#     Apply interestingness_measure to each cluster
#     """
#     i = 0
#     rules_fp_clustering_new = []
#     for rules in rules_fp_clustering:
#         if rules.shape[0] != 0:
#             one_hot_group = one_hot.loc[
#                 :, one_hot.columns.isin(list(group[group.index == i]["Labels"]))
#             ]
#             rules = interestingness_measure(rules, one_hot_group)
#             rules_fp_clustering_new.append(rules)
#         else:
#             rules_fp_clustering_new.append(pd.DataFrame([]))
#         i = i + 1
#     return rules_fp_clustering_new


def rules_HAC_communities(
    one_hot,
    communities,
    nb_cluster,
    metrics,
    max_length,
    min_confidence,
    interestingness,
    encoded_data,
):
    """
    Generate Association rules after applying clustering method to
    one hot encoding matrix with only labels from the same community.

    Parameters :
        one_hot : DataFrame
            One hot encoding DataFrame (rows : articles, columns : labels)
        communities : list of int
            List of communities which each named entities belonged
        nb_cluster : int
            Maximum number of clusters
        metrics : string
            The metric used for the HAC
        nb_min_articles : int
            The minimum number of articles in a cluster
        max_length : int
            Maximum length of rules (e.g : 3 -> 2 antecedents and 1 consequents or 1 antecedents and 2 consequents)
        min_confidence : float
            Minimum confidence i.e the probability to have B when A occurs.
        interestingness: float
            Threshold for the interestingness measure

    Returns :
        all_rules_clustering_communities : DataFrame
            Association rules DataFrame
    """

    all_rules_clustering_communities = pd.DataFrame()

    for i in communities:
        label = [x for x in one_hot.columns if x.startswith("label_")]
        label_drop = [x for x in label if x not in ["label_" + s for s in i]]
        one_hot_cluster = one_hot.drop(label_drop, axis=1)

        groupe_communities, _, index_clusters, index_reclustering = clustering_CAH(
            encoded_data, nb_cluster, metrics
        )

        drop = [x for x in one_hot_cluster.columns if not x.startswith("label_")]
        one_hot_cluster = one_hot_cluster.drop(drop, axis=1)
        one_hot_cluster.columns = list(
            pd.DataFrame(one_hot_cluster.columns)[0].apply(lambda x: x.split("_")[-1])
        )

        rules_fp_clustering_communities = fp_growth_per_cluster(
            one_hot_cluster,
            groupe_communities,
            index_clusters,
            max_length,
            min_confidence,
        )
        print(
            f"Number of rules : {str(pd.concat(rules_fp_clustering_communities).shape[0])}"
        )
        rules_fp_clustering_communities = interestingness_measure_clustering(
            rules_fp_clustering_communities,
            one_hot_cluster,
            groupe_communities,
            index_clusters,
        )
        rules_fp_clustering_communities = delete_redundant_group(
            rules_fp_clustering_communities
        )
        rules_clustering_communities = create_rules_df_group(
            rules_fp_clustering_communities, interestingness
        )

        rules_clustering_communities_final = pd.DataFrame()

        for j in range(len(rules_clustering_communities)):
            rules_clustering_communities[j][
                "cluster"
            ] = f"communities{str(communities.index(i))}_clust{str(j + 1)}"
            rules_clustering_communities_final = (
                rules_clustering_communities_final.append(
                    rules_clustering_communities[j]
                )
            )

        all_rules_clustering_communities = all_rules_clustering_communities.append(
            rules_clustering_communities_final
        )

    all_rules_clustering_communities.reset_index(inplace=True, drop=True)

    return all_rules_clustering_communities
