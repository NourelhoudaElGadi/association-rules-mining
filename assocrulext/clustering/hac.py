import numpy as np
import pandas as pd
from assocrulext.utils.timing import timeit


from scipy.cluster.hierarchy import linkage, fcluster
import scipy.linalg.blas


@timeit
def elbow_method(one_hot, nb_max_cluster, metric):
    """
    Use the elbow method and a rule (having at least 2 groups with more than 50 articles) to determine the number of clusters

    Return the number of clusters (k), a dataframe assigning a group for each article and the group index with more than 50 articles
    """

    Z = linkage(one_hot, method="complete", metric=metric)  # sokalmichener
    last = Z[-(nb_max_cluster + 2) :, 2]
    acceleration = np.diff(last, 2)
    k = acceleration.argmax() + 2

    cah_groups = fcluster(Z, t=k, criterion="maxclust")
    idg = np.argsort(cah_groups)
    groupe = pd.DataFrame(one_hot.index[idg], cah_groups[idg])
    groupe.columns = ["article"]
    index = (
        groupe.groupby([groupe.index])
        .count()
        .index[groupe.groupby([groupe.index]).count()["article"] > 50]
    )

    while (groupe.groupby([groupe.index]).count() > 50).sum()["article"] < 2:
        acceleration[acceleration.argmax()] = 0
        k = acceleration.argmax() + 2
        cah_groups = fcluster(Z, t=k, criterion="maxclust")
        idg = np.argsort(cah_groups)
        groupe = pd.DataFrame(one_hot.index[idg], cah_groups[idg])
        groupe.columns = ["article"]
        index = (
            groupe.groupby([groupe.index])
            .count()
            .index[groupe.groupby([groupe.index]).count()["article"] > 50]
        )
        if (acceleration > 0).sum() == 0:
            k = 0
            groupe = groupe[:0]
            index = []
            break

    return k, groupe, index


@timeit
def repeat_cluster(
    one_hot, group, index_cluster, nb_max_article, nb_cluster=10, metric="cosine"
):
    """

    Repeat elbow method for each group containing more than nb_max_article.

    """

    count = (
        group[group.index.isin(index_cluster)]
        .groupby([group[group.index.isin(index_cluster)].index])
        .count()
    )
    count.reset_index(inplace=True, drop=True)
    index_for_new_cluster = count[count["article"] >= nb_max_article].index
    new_cluster = []

    for i in index_for_new_cluster:
        one_hot_reclustering = one_hot[
            one_hot.index.isin(group[group.index == index_cluster[i]]["article"])
        ]
        nb_cluster_reclustering, groupe_reclustering, index_reclustering = elbow_method(
            one_hot_reclustering, nb_cluster, metric
        )

        if len(index_reclustering) != 0:
            new_cluster.append(
                [one_hot_reclustering, groupe_reclustering, index_reclustering]
            )
        else:
            new_cluster.append([pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])])

    return new_cluster, index_for_new_cluster


@timeit
def clustering_CAH(encoded_data, number_of_clusters=10, metric="cosine"):
    nb_cluster, groupe, index = elbow_method(encoded_data, number_of_clusters, metric)

    groupe[groupe.index.isin(index)].groupby(
        [groupe[groupe.index.isin(index)].index]
    ).count()

    # Apply again elbow method to the groups with more than 500 articles #

    new_cluster, index_of_cluster = repeat_cluster(encoded_data, groupe, index, 500, 5)

    return groupe, new_cluster, index, index_of_cluster
