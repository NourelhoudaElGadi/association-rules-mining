import SPARQLWrapper
import json
from SPARQLWrapper import SPARQLWrapper, JSON

import pandas as pd

from utils import timeit


@timeit
def sparql_service_to_dataframe(service, query):
    """
    Helper function to convert SPARQL results into a Pandas DataFrame.

    Credit to Ted Lawless https://lawlesst.github.io/notebook/sparql-dataframe.html
    """
    sparql = SPARQLWrapper(service)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query()

    processed_results = json.load(result.response)
    cols = processed_results["head"]["vars"]

    out = []
    for row in processed_results["results"]["bindings"]:
        item = [row.get(c, {}).get("value") for c in cols]
        out.append(item)

    return pd.DataFrame(out, columns=cols)


def query(datasets, graph, endpoint):
    """
    Query the data according to the provided SPARQL query (see datasets.py)

    Returns :
        df_total : DataFrame
            The DataFrame with all the request responses
    """

    offset = 0
    endpoint_data = datasets[endpoint]
    if graph != None:
        query = endpoint_data[graph]

    print("query = ", query)

    complete_query = query % (offset)
    endpoint_data_url = endpoint_data["url"]
    df_query = sparql_service_to_dataframe(endpoint_data_url, complete_query)

    ## List with all the request responses ##
    list_total = [df_query]
    ## Get all data by set the offset at each round ##
    while df_query.shape[0] > 0:
        print("offset = ", offset)
        offset += 10000
        complete_query = query % (offset)
        df_query = sparql_service_to_dataframe(endpoint_data_url, complete_query)
        list_total.append(df_query)

    return pd.concat(list_total)
