import re
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


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


def normalize_scores(scores):
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(np.array(scores).reshape(-1, 1))
    normalized_scores = normalized_scores.flatten()
    return normalized_scores


import torch


def score_rules(rules, entity_embeddings, relation_embeddings, model, prefix="all"):
    rule_scores = []

    for rule in tqdm(rules, desc=f"Scoring {prefix} rules for noverly"):
        antecedents = [
            int(a) if a.isdigit() else re.search(r"\d+", a).group()
            for a in rule["antecedents"]
        ]
        consequents = [
            int(c) if c.isdigit() else re.search(r"\d+", c).group()
            for c in rule["consequents"]
        ]

        scores = []

        for antecedent in antecedents:
            if antecedent in entity_embeddings.index:
                for consequent in consequents:
                    if consequent in entity_embeddings.index:
                        pair_scores = []
                        for relation in relation_embeddings.index:
                            hrt_batch = torch.tensor(
                                [[antecedent, relation, consequent]]
                            )
                            triple_scores = model.score_hrt(hrt_batch)
                            pair_scores.append(torch.max(triple_scores).item())
                            # print("pair",pair_scores)
                        max_score = max(pair_scores, default=0.0)
                        # print("max",max_score)
                        scores.append(float(max_score))
                        # print("score",scores)

        rule_score = (
            np.mean(scores) if scores else 0.0
        )  # Score de nouveauté de la règle
        rule_scores.append(rule_score)
        # print("rule_scores",rule_scores)

    normalized_scores = normalize_scores(rule_scores)
    mean_score = np.mean(
        normalized_scores
    )  # Score moyen de nouveauté pour l'ensemble des règles
    # print("mean_score",mean_score)
    # Classer les règles en fonction du score de nouveauté
    unknown_rules = [
        rule for rule, score in zip(rules, normalized_scores) if score > mean_score
    ]  # nouvelles connaissance
    partial_known_rules = [
        rule
        for rule, score in zip(rules, normalized_scores)
        if score <= mean_score and score > 0
    ]
    known_rules = [rule for rule, score in zip(rules, normalized_scores) if score == 0]
    unknown_scores = [score for score in normalized_scores if score > mean_score]
    partial_known_scores = [
        score for score in normalized_scores if 0 < score <= mean_score
    ]
    known_scores = [score for score in normalized_scores if score == 0]

    return (
        rule_scores,
        unknown_rules,
        partial_known_rules,
        known_rules,
        unknown_scores,
        partial_known_scores,
        known_scores,
    )
