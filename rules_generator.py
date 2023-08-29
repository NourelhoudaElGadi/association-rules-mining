from base64 import encode
from pathlib import Path
import warnings
from assocrulext.clustering.community import walk_trap
from assocrulext.clustering.hac import clustering_CAH
from assocrulext.eval import (
    interestingness_measure,
    interestingness_measure_clustering,
    interestingness_measure_community,
    score_rules,
)
from assocrulext.io.data import fetch_crobora_data

from assocrulext.io.querying import sparql_service_to_dataframe
from assocrulext.ml.dimred import dimensionality_reduction
from assocrulext.rules.fitering import delete_redundant, delete_redundant_group
from assocrulext.rules.fpgrowth_algo import fp_growth_association_rules, fp_growth_per_cluster, fp_growth_per_community
from assocrulext.utils.data import (
    create_rules_df,
    create_rules_df_group,
    list_to_string,
    string_to_list,
    transform_data,
)

warnings.filterwarnings("ignore")

import torch
import pandas as pd

import pandas as pd

import pykeen
import rdflib
import numpy as np

import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import re


# import urllib library
from urllib.request import urlopen
import urllib.parse

import umap.umap_ as umap
from utils import *
from datasets import *
from assocrulext.ml.autoencoder import *
import argparse, sys
from os.path import exists

parser = argparse.ArgumentParser()

parser.add_argument(
    "--endpoint",
    help="The endpoint from where retrieve the data (identified through codes: issa, covid). Include this information in queries.json if not available.",
    required=False,
)
parser.add_argument(
    "--input",
    help="If available, path to the CSV file containing the input data",
    required=False,
)  # to reduce the data import time when using the same data
parser.add_argument(
    "--graph",
    help="In case there is a graph where to get the data from in the endpoint, provide (valid for ISSA: agrovoc, geonames, wikidata, dbpedia)",
    required=False,
)
# parser.add_argument('--lang', help='The language of the labels. Default is English (en). Provide the acronym (e.g. en, fr, pt, etc.)', default='en', required=False) # language of labels
parser.add_argument(
    "--filename",
    help="The output file name. If not provided, it will be automatically generated based on the input information.",
    required=False,
)
parser.add_argument(
    "--conf",
    help="Minimum confidence of rules. Default is .7, rules with less than x confidence are filtered out.",
    default=0.7,
    required=False,
)
parser.add_argument(
    "--int",
    help="Minimum interestingness (serendipity, rarity) of rules. Default is .3, rules with less than x interestingess are filtered out.",
    default=0.3,
    required=False,
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Chemin vers le modèle d'embedding de graphe",
)

parser.add_argument(
    "--occurrence",
    help="Keep only terms co-occurring more than x times. Default is 5",
    default=5,
    required=False,
)  # keep only terms co-occurring more than x times, default is 5

parser.add_argument(
    "--nocluster",
    action=argparse.BooleanOptionalAction,
    help="Compute rules without any kind of clustering. Default is False.",
    default=False,
    required=False,
)
parser.add_argument(
    "--community",
    help="Compute rules using a Community Detection method to cluster transactions. Default is False.",
    action=argparse.BooleanOptionalAction,
    default=False,
    required=False,
)
parser.add_argument(
    "--hac",
    help="Compute rules using a hierarchial agglomerative (i.e., bottom-up) clustering (HAC) of transactions. Default is False.",
    action=argparse.BooleanOptionalAction,
    default=False,
    required=False,
)
parser.add_argument(
    "--clustercombo",
    help="Compute rules by combining both clustering methods (i.e. HAC and Community Detection). Default is False.",
    action=argparse.BooleanOptionalAction,
    default=False,
    required=False,
)
parser.add_argument(
    "--method",
    help="The method of dimensionality reduction. Available options: PCA, AutoEncoder,ICA,Isomap,MDS,UMAP,TSNE,LLE",
    default="pca",
    required=False,
)
parser.add_argument(
    "--n_components",
    type=int,
    help="The n_components for the method of dimensionality reduction. Default is 128",
    default=128,
    required=False,
)

parser.add_argument("--n_clusters", type=int, default=10, required=False)

parser.add_argument("--append", default=False, required=False)

parser.add_argument(
    "--combined",
    help="Compute rules using both HAC and Community Detection clustering methods. Default is False.",
    action=argparse.BooleanOptionalAction,
    default=False,
    required=False,
)

args = parser.parse_args()

# app = Flask(__name__)


def query():
    """
    Query the data according to the provided SPARQL query (see datasets.py)

    Returns :
        df_total : DataFrame
            The DataFrame with all the request responses
    """

    offset = 0
    if args.graph != None:
        query = datasets[args.endpoint][args.graph]

    print("query = ", query)

    complete_query = query % (offset)
    df_query = sparql_service_to_dataframe(
        datasets[args.endpoint]["url"], complete_query
    )

    ## List with all the request responses ##
    list_total = [df_query]
    ## Get all data by set the offset at each round ##
    while df_query.shape[0] > 0:
        print("offset = ", offset)
        offset += 10000
        complete_query = query % (offset)
        df_query = sparql_service_to_dataframe(
            datasets[args.endpoint]["url"], complete_query
        )

        list_total.append(df_query)

    ## Concatenate all the dataframes from the list ##
    df_total = pd.concat(list_total)

    data_path = Path("data")
    if not data_path.exists():
        data_path.mkdir()

    datafile = (
        "data/input_data_"
        + args.endpoint
        + ("_" + args.graph if args.graph else "")
        + ".csv"
    )
    df_total.to_csv(
        datafile, sep=",", index=False, header=list(df_total.columns), mode="w"
    )

    return df_total


def fetch_data(url):
    try:
        response = urlopen(url)
        return json.loads(response.read())
    except Exception:
        print("Error in ", url)
        return None


# Creation of the co-occurrence matrix, which is the dataset for clustering


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


@timeit
def extract_rules_no_clustering(one_hot_matrix):
    rules_fp =fp_growth_association_rules(one_hot_matrix, 3, float(args.conf))

    print(
        f"No clustering | Number of rules before filtering = {str(rules_fp.shape[0])}"
    )

    ### POST-PROCESSING : interestingness + removal of redundant rules ###
    rules_fp = interestingness_measure(rules_fp, one_hot_matrix)
    rules_fp = delete_redundant(rules_fp)
    print(
        "No clustering | Number of rules after redundancy filter = "
        + str(rules_fp.shape[0])
    )
    rules = create_rules_df(rules_fp, float(args.int))
    print(
        "No clustering | Number of rules after interestingness filter = "
        + str(rules_fp.shape[0])
    )

    rules["cluster"] = "no_clustering"

    print(f"No clustering | Number of rules : {str(rules.shape[0])}")

    return rules


@timeit
def extract_rules_from_communities(one_hot_label, communities_wt):
    rules_communities_wt = fp_growth_per_community(
        one_hot_label, communities_wt, 3, float(args.conf)
    )
    print(
        "Communities clustering | Number of rules before filtering = "
        + str(pd.concat(rules_communities_wt).shape[0])
    )
    rules_communities_wt = interestingness_measure_community(
        rules_communities_wt, one_hot_label, communities_wt
    )
    rules_communities_wt = delete_redundant_group(rules_communities_wt)
    print(
        "Communities clustering | Number of rules after redundancy filter = "
        + str(pd.concat(rules_communities_wt).shape[0])
    )
    rules_wt = create_rules_df_group(rules_communities_wt, float(args.int))
    print(
        "Communities clustering | Number of rules after interestingness filter = "
        + str(pd.concat(rules_communities_wt).shape[0])
    )

    for i in range(len(rules_wt)):
        rules_wt[i]["cluster"] = "wt" + "_community" + str(i + 1)

    all_rules_wt = pd.concat(rules_wt)
    # Number of rules
    print(f"Communities clustering | Number of rules = {str(all_rules_wt.shape[0])}")

    return all_rules_wt


@timeit
def rules_clustering(one_hot_label, groupe, index, new_cluster, index_of_cluster):
    rules_fp_clustering = fp_growth_per_cluster(
        one_hot_label, groupe, index, 3, float(args.conf)
    )

    print(
        f"Clustering | Number of rules before filtering = {str(pd.concat(rules_fp_clustering).shape[0])}"
    )

    ### POST_PROCESSING ###

    rules_fp_clustering = interestingness_measure_clustering(
        rules_fp_clustering, one_hot_label, groupe, index
    )
    rules_fp_clustering = delete_redundant_group(rules_fp_clustering)

    # Concatenate DataFrames in the list
    rules_clustering = pd.concat(rules_fp_clustering)

    print(
        "Clustering | Number of rules after redundancy filter = "
        + str(rules_clustering.shape[0])
    )
    rules_clustering_final = create_rules_df(rules_clustering, float(args.int))
    print(
        "Clustering | Number of rules after interestingness filter = "
        + str(rules_clustering_final.shape[0])
    )

    # Associate each rule to the cluster it belongs to

    rules_clustering_final["cluster"] = [f"clust{str(i + 1)}" for i in range(len(rules_clustering_final))]

    # Number of rules
    print(f"Clustering | Number of rules = {str(rules_clustering_final.shape[0])}")

    return rules_clustering_final

@timeit
def rules_new_cluster(one_hot_label, new_cluster, index_of_cluster):
    # If we repeated the clustering to decrease the number of articles in some classes then we apply on these new classes
    rules_fp_clustering_reclustered = pd.DataFrame()
    for i in range(len(new_cluster)):
        if len(new_cluster[i][0]) != 0:
            rules = fp_growth_per_cluster(
                one_hot_label, new_cluster[i][1], new_cluster[i][2], 4, float(args.conf)
            )
            print(
                "Clustering "
                + str(i)
                + " | Number of rules = "
                + str(pd.concat(rules).shape[0])
            )
            rules = interestingness_measure_clustering(
                rules, one_hot_label, new_cluster[i][1], new_cluster[i][2]
            )
            rules = delete_redundant_group(rules)
            print(
                "Clustering "
                + str(i)
                + " | Number of rules after redundancy filter = "
                + str(pd.concat(rules).shape[0])
            )
        else:
            rules = pd.DataFrame([])

        rules_fp_clustering_reclustered.append(rules)

    ### POST PROCESSING ###
    rules_reclustering = pd.DataFrame()
    for i in range(len(rules_fp_clustering_reclustered)):
        rules_reclustering.append(
            create_rules_df(rules_fp_clustering_reclustered[i], float(args.int))
        )
        print(f"Clustering | Post-processing step {str(i)}")

    # Associate each rule to the cluster it belongs to. Caution: we have two clusters: the first one and the one we got after reclustering
    # (e.g : clust1_clust1 + clust1_clust2 + clust1_clust3)

    rules_reclustering_final = pd.DataFrame()
    for i in range(len(rules_reclustering)):
        if len(rules_reclustering[i]) != 0:
            for j in range(len(rules_reclustering[i])):
                rules_reclustering[i][j][
                    "cluster"
                ] = f"_clust{str(index_of_cluster[i] + 1)}_clust{str(j + 1)}"
                rules_reclustering_final.append(rules_reclustering[i][j])

    return rules_reclustering_final


def combine_cluster_rules(rules_clustering_final, rules_reclustering_final):
    # Gather all rules from all clusters and remove duplicates across clusters

    rules_clustering = rules_clustering_final.append(rules_reclustering_final)
    rules_clustering.reset_index(inplace=True, drop=True)
    print(f"Clustering | Total number of rules = {str(rules_clustering.shape[0])}")

    # transform lists into strings to use in drop_duplicates
    list_to_string(rules_clustering)

    # remove duplicates, keeping only the duplicate with highest confidence
    rules_clustering = (
        rules_clustering.sort_values("confidence")
        .drop_duplicates(subset=["antecedents", "consequents"], keep="last")
        .sort_index()
    )

    # transform strings back into lists for exporting
    string_to_list(rules_clustering)

    print(
        "Clustering | Total number of rules after duplicate filter = "
        + str(rules_clustering.shape[0])
    )

    return rules_clustering


# Application of Community detection + Clustering (we group article and label)
def rules_community_cluster(one_hot, communities_wt, encoding_dim, method):
    all_rules_clustering_wt = rules_HAC_communities(
        one_hot,
        communities_wt,
        20,
        "cosine",
        3,
        float(args.conf),
        float(args.int),
        encoding_dim,
        method,
    )

    # transform lists into strings to use in drop_duplicates
    list_to_string(all_rules_clustering_wt)

    # remove duplicates, keeping only the duplicate with highest confidence
    all_rules_clustering_wt = (
        all_rules_clustering_wt.sort_values("confidence")
        .drop_duplicates(subset=["antecedents", "consequents"], keep="last")
        .sort_index()
    )

    # transform strings back into lists for exporting
    string_to_list(all_rules_clustering_wt)

    print(
        "Clustering article/label | Number of rules = "
        + str(all_rules_clustering_wt.shape[0])
    )

    return all_rules_clustering_wt

def rules_HAC_communities(communities_wt, matrix_one_hot, encoded_data):
    # Effectuer le clustering HAC
    groupe, new_cluster, index, index_of_cluster = clustering_CAH(
        encoded_data, number_of_clusters=10, metric="cosine"
    )
    rules_clustering_hac = rules_clustering(
        matrix_one_hot, groupe, index, new_cluster, index_of_cluster
    )
    rules_reclustering_hac = rules_new_cluster(
        matrix_one_hot, new_cluster, index_of_cluster
    )
    rules_clustering_hac = combine_cluster_rules(
        rules_clustering_hac, rules_reclustering_hac
    )
    export_rules(rules_clustering_hac, "clustering_hac")
    # Effectuer le clustering de détection de communautés

    communities_clustering = extract_rules_from_communities(
        matrix_one_hot, communities_wt
    )
    export_rules(communities_clustering, "community")
    # Combiner les résultats des deux méthodes
    combined_rules = rules_clustering_hac.append(communities_clustering)
    combined_rules.reset_index(inplace=True, drop=True)


    return combined_rules

def filename(cluster):
    if args.filename:
        return args.filename + "_" + cluster + ".json"

    if args.input:
        return args.input.split(".")[0] + "_" + cluster + ".json"

    graph = "_" + args.graph if args.graph else ""
    dataset = "_" + args.endpoint if args.endpoint else ""
    return "data/rules" + dataset + graph + "_" + cluster + ".json"


def export_rules(rules_df, cluster,rule_scores=None):
    if args.graph:
        rules_df["graph"] = args.graph

    rules_df["source"] = rules_df["antecedents"]
    rules_df["target"] = rules_df["consequents"]


    if rule_scores is not None:
        rules_df["score"] = rule_scores
        
    cluster_filename = filename(cluster)
    print("Filename: " + cluster_filename)
    
    rules_df.to_json(path_or_buf=cluster_filename, orient="records")


if __name__ == "__main__":
    print("Running algorithm with parameters:")
    print(
        (
            "SPARQL endpoint = "
            + (
                "None"
                if args.endpoint is None
                else datasets[args.endpoint]["url"] + " (" + args.endpoint + ")"
            )
        )
    )
    print(f"Graph = {str(args.graph)}")
    print(f"Minimum confidence = {str(args.conf)}")
    print(f"Minimum interestingness = {str(args.int)}")
    print(f"Minimum occurrence = {str(args.occurrence)}")
    print("Input data path = ", args.input)
    print("Output data file = ", args.filename)

    args.append = args.append == "True"

    if args.endpoint is None and args.input is None:
        print(
            "You must provide either an endpoint name (e.g. issa, covid) or an input file."
        )
        sys.exit(0)

    if args.endpoint is not None and args.endpoint not in datasets:
        print(
            "Please provide a valid endpoint. The endpoint "
            + args.endpoint
            + " is not registered."
        )
        sys.exit(0)

    path_suffix = (
        f"_{args.graph}_{args.endpoint}_{args.occurrence}_{args.conf}_{args.int}"
    )

    if args.input is not None:
        df_total = pd.read_csv(args.input)
    elif args.endpoint is not None and datasets[args.endpoint]["type"] == "rdf":
        ## retrieve the data from SPARQL endpoint
        df_total = query()
    elif args.endpoint == "crobora":
        df_total = fetch_crobora_data()

    print(f"Input size = {str(df_total.shape[0])} lines")

    # DATA PREPARATION : keep articles with at least one label associated, sort articles by alphabetic order, put labels all in lower case, etc.

    matrix_path = Path(f"data/matrix_one_hot{path_suffix}.csv")

    if matrix_path.exists():
        matrix_one_hot = pd.read_csv(matrix_path)
        print(matrix_one_hot.shape)

    else:
        df_article_sort = transform_data(df_total, int(args.occurrence))
        print(
            "Number of unique items (articles) : "
            + str(len(df_article_sort["article"].unique()))
        )
        print(
            "Number of unique labels (e.g. named entities) : "
            + str(len(df_article_sort["label"].unique()))
        )
        matrix_one_hot = compute_cooccurrence_matrix(df_article_sort)
        matrix_one_hot.to_csv(matrix_path, index=False)

    encoded_path = Path(
        f"data/encoded_data{path_suffix}_NC{args.n_components}_{args.method}.csv"
    )
    if encoded_path.exists():
        encoded_data = pd.read_csv(encoded_path)
    else:
 
        my_pykeen_model = torch.load(args.model_path, map_location=torch.device("cpu"))
        entity_embeddings = my_pykeen_model.entity_representations[0]._embeddings.weight
        relation_embeddings = my_pykeen_model.relation_representations[0]._embeddings.weight
        entity_embeddings_cpu = entity_embeddings.cpu()
        relation_embeddings_cpu = relation_embeddings.cpu()
        entity_embeddings_df = pd.DataFrame(entity_embeddings_cpu.detach().numpy())
        relation_embeddings_df = pd.DataFrame(relation_embeddings_cpu.detach().numpy())
        rule_score=score_one_hot_matrices(matrix_one_hot,entity_embeddings_df, relation_embeddings_df,my_pykeen_model)
        rule_score_df = pd.DataFrame(rule_score)
        rule_score_df.to_json(f'rule_scores1.json', orient="records")
        encoded_data = dimensionality_reduction(
            matrix_one_hot, args.n_components, args.method,rule_score
        )
        encoded_data.to_csv(encoded_path, index=False)

    rules_no_clustering = pd.DataFrame()
    if args.nocluster:
        rules_no_clustering = extract_rules_no_clustering(matrix_one_hot)
        export_rules(rules_no_clustering, "no_cluster")

    rules_communities = pd.DataFrame()
    if args.community:
        communities_wt = walk_trap(matrix_one_hot)
        rules_communities = extract_rules_from_communities(
            matrix_one_hot, communities_wt
        )
        export_rules(rules_communities, "community")

    rules_clustering_CAH = pd.DataFrame()
    if args.hac:
        ## generate clusters from labels
        groupe, new_cluster, index, index_of_cluster = clustering_CAH(
            encoded_data, number_of_clusters=10, metric="cosine"
        )
        ## generate rules from clusters
        rules_clustering_h = rules_clustering(
            matrix_one_hot, groupe, index, new_cluster, index_of_cluster
        )

        ## find sub-clusters, if any, and generate rules from them
        rules_reclustering = rules_new_cluster(
            matrix_one_hot, new_cluster, index_of_cluster
        )

        ## combine all rules generated from clustering and remove duplicates (possible rules find in several clusters), keeping only the most relevant
        rules_clustering_CAH = combine_cluster_rules(
            rules_clustering_h, rules_reclustering
        )
        export_rules(rules_clustering_CAH, "clustering_hac")

    rules_combined = pd.DataFrame()
    if args.clustercombo:
        communities_wt = walk_trap(matrix_one_hot)
        rules_combined = rules_HAC_communities(
            communities_wt,
            matrix_one_hot,
            encoded_data
        )
        export_rules(rules_combined, "combining_clustering_community")
    # all_rules_clustering_wt = rulesCommunityCluster(matrix_one_hot, communities_wt)

    # exportRules(all_rules_clustering_wt, 'communities_clustering')
    all_rules = rules_no_clustering.append(rules_clustering_CAH).append(
        rules_communities
    ).append(rules_combined)
    all_rules.reset_index(inplace=True, drop=True)

    print("All rules | Number of rules = ", all_rules.shape[0])
    list_to_string(all_rules)
    all_rules = all_rules.drop_duplicates(
        subset=["antecedents", "consequents", "isSymmetric"]
    )
    string_to_list(all_rules)

    print(
        "All rules | Number of rules after symmetric duplicate filter = ",
        all_rules.shape[0],
    )
    export_rules(all_rules, "all_rules")
    all_rules_with_scores = pd.DataFrame()
    if args.model_path:
        my_pykeen_model = torch.load(args.model_path, map_location=torch.device("cpu"))
        entity_embeddings = my_pykeen_model.entity_representations[0]._embeddings.weight
        relation_embeddings = my_pykeen_model.relation_representations[0]._embeddings.weight
        entity_embeddings_cpu = entity_embeddings.cpu()
        relation_embeddings_cpu = relation_embeddings.cpu()
        # Convertir les embeddings des entités en DataFrame
        entity_embeddings_df = pd.DataFrame(entity_embeddings_cpu.detach().numpy())
        relation_embeddings_df = pd.DataFrame(relation_embeddings_cpu.detach().numpy())
        with open("data/rules_issa_agrovoc_all_rules.json", "r") as file:
            rules = json.load(file)

        # Appeler la fonction score_rules pour obtenir les scores et classer les règles en catègories
        (
            rule_scores,
            unknown_rules,
            partial_known_rules,
            known_rules,
            unknown_scores,
            partial_known_scores,
            known_scores,
            ) = score_rules(
            rules, entity_embeddings_df, relation_embeddings_df, my_pykeen_model
        )
        rule_scores_df = pd.DataFrame(rule_scores)
        rule_scores_df.to_json(f'rule_scores.json', orient="records")
        #export_rules(rules, "all_rules_with_scores",rule_scores=rule_scores)
        # Convertir les règles classifiées en DataFrames
        unknown_df = pd.DataFrame(unknown_rules)
        export_rules(unknown_df, "all_rules_with_scores",rule_scores=unknown_scores)
    filename = Path(f"data/config_{str(args.endpoint)}.json")
    # verify if config file exists before
    if filename.exists():
        with open(filename, "r") as f:
            config = json.load(f)
    else:
        config = {
            "lang": [],
            "graph": [],
            "min_interestingness": float(args.int),
            "min_confidence": float(args.conf),
            "methods": [
                {"label": "No clustering method", "key": "no_clustering"},
                {"label": "Clusters of labels", "key": "clust_"},
                {"label": "Communities of articles", "key": "wt_community"},
                {
                    "label": "Combination of clusters and communities",
                    "key": "communities",
                },
            ],
        }
    if args.graph is not None:
        config["graph"].append(args.graph)

    with open(filename, "w") as outfile:
        json.dump(config, outfile, indent=4, sort_keys=False)
