from functools import wraps
import logging
import time
import pandas as pd

import numpy as np


from scipy.cluster.hierarchy import linkage, fcluster
import scipy.linalg.blas

from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
from tqdm import tqdm

import umap

from autoencoder import AutoEncoderDimensionReduction

logger = logging.getLogger(__name__)


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def delete_Label_number(label):
    k = 0
    for i in label:
        if i.isdigit():
            k = k + 1
    return k == len(label)


@timeit
def transform_data(df, min_occur):
    """

    Pre-processing data :
        Keep articles with more than 1 named entity
        Sort data by article
        Labels in lower case
        Remove "," and "."
        Delete Labels containing only numbers
        Keep Labels occurring more than 4 times

    Parameters :
        df : DataFrame

    Returns :
        df_article_sort : DataFrame
    """

    url_article = df["article"].value_counts()[df["article"].value_counts() > 1].index
    df_article = df.loc[df["article"].isin(url_article)]
    df_article_sort = df_article.sort_values(by=["article"])

    df_article_sort = df_article_sort.astype(
        {"article": str, "label": str}
    )  # , "year": str})
    # df_article_sort['Label'] = df_article_sort['Label'].apply(lambda x: x.lower())
    df_article_sort["label"] = df_article_sort[
        "label"
    ]  # .apply(lambda x: x.replace('.', '').replace(',', ''))
    # df_article_sort = df_article_sort.drop(df_article_sort[df_article_sort['label']])
    # apply(lambda x: delete_Label_number(x)) == True].index)
    df_article_sort = df_article_sort.dropna(subset=["label"])

    df_article_sort = df_article_sort.loc[
        df_article_sort["label"].isin(
            list(
                df_article_sort["label"]
                .value_counts(sort=True)[
                    df_article_sort["label"].value_counts(sort=True) > min_occur
                ]
                .index
            )
        )
    ]

    df_article_sort = df_article_sort.loc[
        ~df_article_sort["label"].isin(
            list(df_article_sort["label"].value_counts(sort=True).head(15).index)
        )
    ]

    df_article_sort["label"] = df_article_sort["label"].astype("category")
    # df_article_sort["year"] = df_article_sort["year"].astype('category')
    # df_article_sort["body"] = df_article_sort["body"].astype('category')

    return df_article_sort


def coocc_matrix_Label(one_hot_label):
    """
    Create Labels co-occurrences matrix

    Parameters :
        one_hot_label : DataFrame
            One hot encoding DataFrame with only labels

    Returns :
        coocc : DataFrame
            The co-occurrence matrix
    """

    coocc = scipy.linalg.blas.dgemm(
        alpha=1.0, a=one_hot_label.T, b=one_hot_label.T, trans_b=True
    )
    np.fill_diagonal(coocc, 0)  # replace the diagonal by 0

    return coocc


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
def repeat_cluster(one_hot, group, index_cluster, nb_max_article, nb_cluster):
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
            one_hot_reclustering, nb_cluster, "cosine"
        )

        if len(index_reclustering) != 0:
            new_cluster.append(
                [one_hot_reclustering, groupe_reclustering, index_reclustering]
            )
        else:
            new_cluster.append([pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])])

    return new_cluster, index_for_new_cluster


@timeit
def fp_growth(one_hot, max_len, min_confidence):
    """

    Apply FP-Growth and generate rules with parameter maximum length and minimum confidence
    minimum support is computes in order to have at least 5 articles.

    """
    # / one_hot.shape[0]
    frequent_itemsets_fp = fpgrowth(
        one_hot, min_support=5 / one_hot.shape[0], max_len=max_len, use_colnames=True
    )
    return association_rules(
        frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence
    ).sort_values(by="lift", ascending=False)


@timeit
def fp_growth_with_clustering(one_hot, group, index, max_len, min_confidence):
    """
    Apply FP-Growth algorithm and generate rules to each cluster.

    Parameters :
        one_hot : DataFrame
            One hot encoding DataFrame (rows : articles, columns : labels)
        group : DataFrame
            A DataFrame assigning a group for each article
        index : list of int
            Index of group
        max_len : int
            Maximum length of rules (e.g : 3 -> 2 antecedents and 1 consequents or 1 antecedents and 2 consequents)
        min_confidence : float
            Minimum confidence i.e the probability to have B when A occurs.

    Returns :
        rules_fp_clustering : list of DataFrame
            List of Association rules DataFrame
    """

    rules_fp_clustering = []

    for i in index:
        one_hot_cluster = one_hot[
            one_hot.index.isin(list(group[group.index == i]["article"]))
        ]
        frequent_itemsets_fp = fpgrowth(
            one_hot_cluster,
            min_support=5 / one_hot_cluster.shape[0],
            max_len=max_len,
            use_colnames=True,
        )
        if len(frequent_itemsets_fp) != 0:
            rules_fp_clustering.append(
                association_rules(
                    frequent_itemsets_fp,
                    metric="confidence",
                    min_threshold=min_confidence,
                ).sort_values(by="lift", ascending=False)
            )
        else:
            rules_fp_clustering.append(pd.DataFrame([]))
    return rules_fp_clustering


@timeit
def fp_growth_with_community(one_hot, communities, max_len, min_confidence):
    """
    Apply FP-Growth algorithm and generate rules for selected clusters
    index = the groups with more than 50 articles (see elbow method)
    """

    rules_fp_clustering = []

    for i in range(len(communities)):
        one_hot_cluster = one_hot.T[one_hot.columns.isin(communities[i])].T
        frequent_itemsets_fp = fpgrowth(
            one_hot_cluster,
            min_support=5 / one_hot_cluster.shape[0],
            max_len=max_len,
            use_colnames=True,
        )
        if len(frequent_itemsets_fp) != 0:
            rules_fp_clustering.append(
                association_rules(
                    frequent_itemsets_fp,
                    metric="confidence",
                    min_threshold=min_confidence,
                ).sort_values(by="lift", ascending=False)
            )
        else:
            rules_fp_clustering.append(pd.DataFrame([]))
    return rules_fp_clustering


def is_symmetric(rule1, rule2):
    """
    Check if a rule is symmetric
    """

    return bool(
        (rule1.antecedents == rule2.consequents)
        & (rule1.consequents == rule2.antecedents)
    )


def find_symmetric(x, rules):
    """
    Find a symmetric of x among the rules
    """
    for y in rules.itertuples():
        if is_symmetric(x, y):
            x["isSymmetric"] = True
            break
    return x


def interestingness_measure(rules_fp, one_hot):
    """
    Compute a measure of the interestingness
    """

    size = one_hot.shape[0]
    rules_fp["interestingness"] = (
        (rules_fp["support"] ** 2)
        / (rules_fp["antecedent support"] * rules_fp["consequent support"])
    ) * (1 - (rules_fp["support"] / size))

    return rules_fp


def interestingness_measure_clustering(rules_fp_clustering, one_hot, group, index):
    """
    Apply interestingness_measure to each cluster
    """
    i = 0
    rules_fp_clustering_new = []
    for rules in rules_fp_clustering:
        if rules.shape[0] != 0:
            one_hot_group = one_hot[
                one_hot.index.isin(group[group.index == index[i]]["article"])
            ]
            rules = interestingness_measure(rules, one_hot_group)
            rules_fp_clustering_new.append(rules)
        else:
            rules_fp_clustering_new.append(pd.DataFrame([]))
        i = i + 1
    return rules_fp_clustering_new


def interestingness_measure_community(rules_fp_clustering, one_hot, communities):
    """
    Apply interestingness_measure to each cluster
    """
    i = 0
    rules_fp_clustering_new = []
    for rules in rules_fp_clustering:
        if rules.shape[0] != 0:
            one_hot_group = one_hot.T[one_hot.columns.isin(communities[i])].T
            rules = interestingness_measure(rules, one_hot_group)
            rules_fp_clustering_new.append(rules)
        else:
            rules_fp_clustering_new.append(pd.DataFrame([]))
        i = i + 1
    return rules_fp_clustering_new


def create_rules_df(rules_fp, interestingness):
    """
    Create the final rules dataframe by keeping rules with a value of interestingness grater than a threshold
    and finding symmetric rules.
    """

    rules = rules_fp.loc[
        :, ["antecedents", "consequents", "confidence", "interestingness", "support"]
    ]
    rules = rules[rules["interestingness"] >= interestingness]
    rules.reset_index(inplace=True, drop=True)
    rules["isSymmetric"] = False
    rules = rules.apply(lambda x: find_symmetric(x, rules), axis=1)

    return rules


def create_rules_df_clustering(rules_fp_clustering, interestingness):
    """
    Apply create_rules_df to each cluster
    """

    rules_clustering = []
    for rules in rules_fp_clustering:
        if len(rules) != 0:
            rules = create_rules_df(rules, interestingness)
            rules_clustering.append(rules)
        else:
            rules_clustering.append(pd.DataFrame([]))

    return rules_clustering


def create_rules_df_community(rules_fp_clustering, interestingness):
    """
    Apply create_rules_df to each cluster
    """

    rules_clustering = []
    for rules in rules_fp_clustering:
        if len(rules) != 0:
            rules = create_rules_df(rules, interestingness)
            rules_clustering.append(rules)
        else:
            rules_clustering.append(pd.DataFrame([]))

    return rules_clustering


@timeit
def delete_redundant(rules):
    """
    Delete redundant rules. A rule is redundant if there is a subset of this rule with the same or higher confidence.

    (A,B,C) -> D is redundant if (A,B) -> D has the same or higher confidence.

    """

    redundant = []
    for i in tqdm(rules.itertuples()):
        redundant.extend(
            j.Index
            for j in rules.itertuples()
            if (
                (
                    i.consequents == j.consequents
                    and i.confidence >= j.confidence
                    and i.Index != j.Index
                    and i.antecedents.issubset(j.antecedents)
                )
                or (
                    i.antecedents == j.antecedents
                    and i.confidence >= j.confidence
                    and i.Index != j.Index
                    and i.consequents.issubset(j.consequents)
                )
            )
        )
    redundant = list(dict.fromkeys(redundant))
    rules = rules.drop(redundant)
    return rules


@timeit
def delete_redundant_clustering_or_communities(rules_clustering):
    """
    Apply delete_redundant to each cluster
    """

    rules_without_redundant = []
    for rules in tqdm(rules_clustering, desc="Deleting redundant rules"):
        rules = delete_redundant(rules)
        rules_without_redundant.append(rules)
    return rules_without_redundant


@timeit
def generate_article_rules(test, rules):
    """
    For each article in the test set,  the method checks if labels and pair of labels of the article
    are antecedent in the created rules. If yes, it adds the consequents to the list of new rules.

    Return a list of list of new rules for each article.

    """

    new_rules = []

    for article in test["article"].unique():
        new_rules_article = []
        for i in test[test["article"] == article]["label"]:
            if rules[rules["antecedents"].eq({i})].shape[0] != 0:
                new_rules_article.append(
                    list(rules[rules["antecedents"].eq({i})]["consequents"])
                )

            new_rules_article.extend(
                list(rules[rules["antecedents"].eq({i, j})]["consequents"])
                for j in test[test["article"] == article]["label"]
                if (rules[rules["antecedents"].eq({i, j})].shape[0] != 0)
            )
        new_rules.append(new_rules_article)

    new_rules_list = []

    for new_rule in new_rules:
        rules_i = []
        for j in range(len(new_rule)):
            rules_i.extend(list(new_rule[j][k])[0] for k in range(len(new_rule[j])))
        new_rules_list.append(list(dict.fromkeys(rules_i)))

    return new_rules_list


@timeit
def elbow_method_community(one_hot, nb_max_cluster, metric):
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
    groupe.columns = ["Labels"]
    index = (
        groupe.groupby([groupe.index])
        .count()
        .index[groupe.groupby([groupe.index]).count()["Labels"] > 20]
    )

    while (groupe.groupby([groupe.index]).count() > 20).sum()["Labels"] < 2:
        acceleration[acceleration.argmax()] = 0
        k = acceleration.argmax() + 2
        cah_groups = fcluster(Z, t=k, criterion="maxclust")
        idg = np.argsort(cah_groups)
        groupe = pd.DataFrame(one_hot.index[idg], cah_groups[idg])
        groupe.columns = ["Labels"]
        index = (
            groupe.groupby([groupe.index])
            .count()
            .index[groupe.groupby([groupe.index]).count()["Labels"] > 20]
        )
        if (acceleration > 0).sum() == 0:
            k = 0
            groupe = groupe[:0]
            index = []
            break

    return k, groupe, index


@timeit
def fp_growth_with_com_auto(one_hot, group, index, max_len, min_confidence):
    """
    Apply FP-Growth algorithm and generate rules to each cluster.

    Parameters :
        one_hot : DataFrame
            One hot encoding DataFrame (rows : articles, columns : labels)
        group : DataFrame
            A DataFrame assigning a group for each article
        index : list of int
            Index of group
        max_len : int
            Maximum length of rules (e.g : 3 -> 2 antecedents and 1 consequents or 1 antecedents and 2 consequents)
        min_confidence : float
            Minimum confidence i.e the probability to have B when A occurs.

    Returns :
        rules_fp_clustering : list of DataFrame
            List of Association rules DataFrame
    """

    rules_fp_clustering = []

    for i in index:
        one_hot_cluster = one_hot.loc[
            :, one_hot.columns.isin(list(group[group.index == i]["Labels"]))
        ]
        frequent_itemsets_fp = fpgrowth(
            one_hot_cluster,
            min_support=5 / one_hot_cluster.shape[0],
            max_len=max_len,
            use_colnames=True,
        )
        if len(frequent_itemsets_fp) != 0:
            rules_fp_clustering.append(
                association_rules(
                    frequent_itemsets_fp,
                    metric="confidence",
                    min_threshold=min_confidence,
                ).sort_values(by="lift", ascending=False)
            )
        else:
            rules_fp_clustering.append(pd.DataFrame([]))
    return rules_fp_clustering


def interestingness_measure_com_auto(rules_fp_clustering, one_hot, group, index):
    """
    Apply interestingness_measure to each cluster
    """
    i = 0
    rules_fp_clustering_new = []
    for rules in rules_fp_clustering:
        if rules.shape[0] != 0:
            one_hot_group = one_hot.loc[
                :, one_hot.columns.isin(list(group[group.index == i]["Labels"]))
            ]
            rules = interestingness_measure(rules, one_hot_group)
            rules_fp_clustering_new.append(rules)
        else:
            rules_fp_clustering_new.append(pd.DataFrame([]))
        i = i + 1
    return rules_fp_clustering_new


from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding


# Reduction of the number of variables + Clustering
# The autoencoder allows to reduce the dimension and to be able to apply the CAH which is not robust in the face of too many variables
def dimensionality_reduction(one_hot_matrix, n_components, method):
    methods = {
        "autoencoder": AutoEncoderDimensionReduction(
            encoding_dim=n_components,
            epochs=100,
            batch_size=128,
            lr=1e-2,
        ),
        "pca": PCA(n_components=n_components),
        "tsne": TSNE(
            n_components=n_components,
            method="exact",
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
        ),
        "umap": umap.UMAP(n_components=n_components),
        "isomap": Isomap(n_neighbors=5, n_components=n_components),
        "mds": MDS(n_components=n_components),
        "ica": FastICA(n_components=n_components),
        "lle": LocallyLinearEmbedding(n_components=n_components),
    }

    # The above code is implementing dimensionality reduction techniques such as autoencoder, PCA, ICA,
    # LLE, etc. on a given input matrix `one_hot_matrix`. If the chosen method is autoencoder, it creates
    # an autoencoder model using Keras and trains it on the input matrix. If the chosen method is any
    # other dimensionality reduction technique, it applies the chosen method on the input matrix and
    # returns the reduced data in a pandas DataFrame format. The number of components for the
    # dimensionality reduction is specified by the `n_components` parameter. The resulting reduced data is
    # returned with column names based

    # Convert the input matrix to a dense or sparse matrix depending on the chosen method
    if method in ["pca", "ica", "lle", "ica", "isomap", "mds"]:
        one_hot_matrix_dense = csr_matrix(one_hot_matrix).toarray()
    else:
        one_hot_matrix_dense = one_hot_matrix.values

    # Dimensionality reduction
    reducer = methods[method]
    reduced_data = pd.DataFrame(reducer.fit_transform(one_hot_matrix_dense))

    # Define column names for the reduced data
    reduced_data.columns = [method + "_" + str(i + 1) for i in range(n_components)]
    reduced_data.index = one_hot_matrix.index

    return reduced_data


def rules_clustering_communities_reduction(
    one_hot,
    communities,
    nb_cluster,
    metrics,
    max_length,
    min_confidence,
    interestingness,
    encoding_dim=64,
    method="pca",
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

        input_dim = one_hot.shape[1]
        # Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders
        logger.info(f"Reducing dimensions to: {str(encoding_dim)} with {method}")
        encoded_data = dimensionality_reduction(one_hot, encoding_dim, method=method)

        nb_cluster_communities, groupe_communities, index_communities = elbow_method(
            encoded_data, nb_cluster, metrics
        )

        drop = [x for x in one_hot_cluster.columns if not x.startswith("label_")]
        one_hot_cluster = one_hot_cluster.drop(drop, axis=1)
        one_hot_cluster.columns = list(
            pd.DataFrame(one_hot_cluster.columns)[0].apply(lambda x: x.split("_")[-1])
        )

        rules_fp_clustering_communities = fp_growth_with_clustering(
            one_hot_cluster,
            groupe_communities,
            index_communities,
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
            index_communities,
        )
        rules_fp_clustering_communities = delete_redundant_clustering_or_communities(
            rules_fp_clustering_communities
        )
        rules_clustering_communities = create_rules_df_clustering(
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


def rules_clustering_communities_embedding_autoencoder(
    one_hot,
    groupe,
    index,
    nb_cluster,
    metrics,
    max_length,
    min_confidence,
    interestingness,
    encoding_dim=32,
    method="pca",
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

    for i in index:
        label = [x for x in one_hot.columns if x.startswith("Label_")]
        label_drop = [
            x
            for x in label
            if x
            not in ["label_" + s for s in list(groupe[groupe.index == i]["Labels"])]
        ]
        one_hot_cluster = one_hot.drop(label_drop, axis=1)
        encoding_dim = 32
        logger.info(f"Reducing dimensions to: {encoding_dim} with {method}")
        encoded_data = dimensionality_reduction(
            one_hot_cluster, encoding_dim, method=method
        )

        nb_cluster_communities, groupe_communities, index_communities = elbow_method(
            encoded_data, nb_cluster, metrics
        )

        drop = [x for x in one_hot_cluster.columns if not x.startswith("label_")]
        one_hot_cluster = one_hot_cluster.drop(drop, axis=1)
        one_hot_cluster.columns = list(
            pd.DataFrame(one_hot_cluster.columns)[0].apply(lambda x: x.split("_")[-1])
        )

        rules_fp_clustering_communities = fp_growth_with_clustering(
            one_hot_cluster,
            groupe_communities,
            index_communities,
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
            index_communities,
        )
        rules_fp_clustering_communities = delete_redundant_clustering_or_communities(
            rules_fp_clustering_communities
        )
        rules_clustering_communities = create_rules_df_clustering(
            rules_fp_clustering_communities, interestingness
        )

        rules_clustering_communities_final = pd.DataFrame()

        for j in range(len(rules_clustering_communities)):
            rules_clustering_communities[j][
                "cluster"
            ] = f"communities{str(i + 1)}_clust{str(j + 1)}"
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


def dataframe_difference(df1, df2):
    """Find rows which are equal between two DataFrames."""
    comparison_df = df1.merge(df2, indicator=True, how="outer")
    diff_df = comparison_df[comparison_df["_merge"] == "both"]

    return diff_df.shape[0]


def comparison(rules1, rules2):
    print(f"Number of rules 1 : {str(rules1.shape[0])}")
    print(f"Number of rules 2 : {str(rules2.shape[0])}")
    print(
        "Number of same rows : "
        + str(
            dataframe_difference(
                rules1.loc[:, ["antecedents", "consequents"]],
                rules2.loc[:, ["antecedents", "consequents"]],
            )
        )
    )
    print(
        "Number of same rows among top 10 most interesting rules : "
        + str(
            dataframe_difference(
                rules1.sort_values(
                    by=["confidence", "interestingness"], ascending=False
                )
                .loc[:, ["antecedents", "consequents"]]
                .head(10),
                rules2.sort_values(
                    by=["confidence", "interestingness"], ascending=False
                )
                .loc[:, ["antecedents", "consequents"]]
                .head(10),
            )
        )
    )
    print(
        "Number of same rows among top 20 most interesting rules : "
        + str(
            dataframe_difference(
                rules1.sort_values(
                    by=["confidence", "interestingness"], ascending=False
                )
                .loc[:, ["antecedents", "consequents"]]
                .head(20),
                rules2.sort_values(
                    by=["confidence", "interestingness"], ascending=False
                )
                .loc[:, ["antecedents", "consequents"]]
                .head(20),
            )
        )
    )
