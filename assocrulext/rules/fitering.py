from assocrulext.utils.timing import timeit

from tqdm import tqdm


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
def delete_redundant_group(rules_clustering):
    """
    Apply delete_redundant to each cluster
    """

    rules_without_redundant = []
    for rules in tqdm(rules_clustering, desc="Deleting redundant rules"):
        rules = delete_redundant(rules)
        rules_without_redundant.append(rules)
    return rules_without_redundant


def dataframe_difference(df1, df2):
    """Find rows which are equal between two DataFrames."""
    comparison_df = df1.merge(df2, indicator=True, how="outer")
    diff_df = comparison_df[comparison_df["_merge"] == "both"]

    return diff_df.shape[0]


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
