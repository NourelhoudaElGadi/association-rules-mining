from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth


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
def fp_growth_per_cluster(one_hot, group, index, max_len, min_confidence):
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
        rules_fp_clustering.extend(
            extract_rules_for_group(one_hot_cluster, max_len, min_confidence)
        )
    return rules_fp_clustering


@timeit
def fp_growth_per_community(one_hot, communities, max_len, min_confidence):
    """
    Apply FP-Growth algorithm and generate rules for selected clusters
    index = the groups with more than 50 articles (see elbow method)
    """

    rules_fp_clustering = []

    for i in range(len(communities)):
        one_hot_cluster = one_hot.T[one_hot.columns.isin(communities[i])].T
        rules_fp_clustering.extend(
            extract_rules_for_group(one_hot_cluster, max_len, min_confidence)
        )
    return rules_fp_clustering


def extract_rules_for_group(one_hot_cluster, max_len, min_confidence):
    rules_fp_clustering = []

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
