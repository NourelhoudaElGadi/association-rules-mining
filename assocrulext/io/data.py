import json
from pathlib import Path

from urllib.request import urlopen

import pandas as pd
import urllib3

from utils import timeit


def fetch_data(url):
    try:
        response = urlopen(url)
        return json.loads(response.read())
    except Exception:
        print("Error in ", url)
        return None


def fetch_crobora_data(datasets, endpoint, append, data_path: Path):
    url = datasets[endpoint]["labels"]

    datafile = data_path / f"input_data_{endpoint}.csv"
    keysfile = data_path / "fetched_keys_crobora.csv"

    data_json = []
    if isinstance(url, list):
        for u in url:
            # response = urlopen(u) # store the response of URL
            data_json += fetch_data(u)

    data = []

    fetched_urls = []
    if (
        append and keysfile.exists()
    ):  # in case of network issues, it resumes from where it started
        csv_file = pd.read_csv(datafile)
        # Create a multiline json
        data = json.loads(csv_file.to_json(orient="records"))

        urls = pd.read_csv(keysfile)
        fetched_urls = list(urls["url"])

    for value in data_json:
        category = value["type"]
        keyword = value["value"]

        params = {"categories": category, "keywords": keyword}
        params = urllib3.parse.urlencode(params)  # verifier !

        data_url = datasets[endpoint]["images"] % (params)
        data_url = data_url.replace(" ", "%20")

        if data_url in fetched_urls:
            continue

        # response = urlopen(data_url)
        value_json = fetch_data(data_url)
        if value_json is None:
            continue

        for document in value_json:
            for record in document["records"]:
                data.append(
                    {
                        "article": record["image_title"],
                        "label": category + "--" + keyword,
                    }
                )

        data_df = pd.DataFrame(data)
        data_df.to_csv(
            datafile, sep=",", index=False, header=list(data_df.columns), mode="w"
        )

        fetched_urls.append(data_url)
        fetched_df = pd.DataFrame(fetched_urls, columns=["url"])
        fetched_df.to_csv(
            keysfile, sep=",", index=False, header=list(fetched_df.columns), mode="w"
        )

    return data_df


@timeit
def compute_cooccurrence_matrix(df_article_sort):
    # Convert the article column to string and specify that label and year are categories for the one-hot-encoding
    df_article_sort[["label"]].drop_duplicates()

    train = df_article_sort.astype({"article": str, "label": str})
    train["label"] = train["label"].astype("category")

    ##One hot encoding train set (5000 by 5000 articles) + Sparse type to reduce the memory
    one_hot = (
        pd.get_dummies(
            train[train["article"].isin(train["article"].unique()[:5000])]
            .drop_duplicates()
            .set_index("article")
        )
        .sum(level=0)
        .apply(lambda y: y.apply(lambda x: 1 if x >= 1 else 0))
        .astype("Sparse[int]")
    )
    i = 5000
    while one_hot.shape[0] < len(train["article"].unique()):
        one_hot = one_hot.append(
            pd.get_dummies(
                train[train["article"].isin(train["article"].unique()[i : i + 5000])]
                .drop_duplicates()
                .set_index("article")
            )
            .sum(level=0)
            .apply(lambda y: y.apply(lambda x: 1 if x >= 1 else 0))
            .astype("Sparse[int]")
        )
        i = i + 5000

    # Replace NaN by 0 and delete the rows with only 0
    one_hot = one_hot.fillna(0)
    one_hot = one_hot.loc[:, (one_hot != 0).any(axis=0)]

    # Convert the one-hot-encoded matrix to sparse type to reduce the memory usage
    one_hot = one_hot.astype("Sparse[int]")

    # Delete the year columns if we don't want to take them into account in the analysis
    drop = [x for x in one_hot.columns if not x.startswith("label_")]
    one_hot_label = one_hot.drop(drop, axis=1)
    one_hot_label.columns = list(
        pd.DataFrame(one_hot_label.columns)[0].apply(lambda x: x.split("_")[-1])
    )

    return one_hot_label
