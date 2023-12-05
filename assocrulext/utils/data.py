import pickle
import pandas as pd
from assocrulext.utils.timing import timeit
from assocrulext.rules.fitering import find_symmetric

import pyarrow as pa


def delete_Label_number(label):
    k = 0
    for i in label:
        if i.isdigit():
            k = k + 1
    return k == len(label)


def df_from_redis_if_exists(redis_client, key):
    """
    Return a DataFrame from Redis if it exists, else return None.
    """
    if redis_client.exists(key):
        return pickle.loads(redis_client.get(key))
    else:
        return None


def df_to_redis(redis_client, key, df):
    """
    Serialize a DataFrame and store it in Redis.
    """
    redis_client.set(key, pickle.dumps(df).to_buffer().to_pybytes())


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


def create_rules_df_group(rules_fp_clustering, interestingness):
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


def dataframe_difference(df1, df2):
    """Find rows which are equal between two DataFrames."""
    comparison_df = df1.merge(df2, indicator=True, how="outer")
    diff_df = comparison_df[comparison_df["_merge"] == "both"]

    return diff_df.shape[0]


def list_to_string(df):
    # transform lists into strings to use in drop_duplicates
    df["antecedents"] = [",".join(map(str, l)) for l in df["antecedents"]]
    df["consequents"] = [",".join(map(str, l)) for l in df["consequents"]]


def string_to_list(df):
    df["antecedents"] = [x.split(",") for x in df["antecedents"]]
    df["consequents"] = [x.split(",") for x in df["consequents"]]
