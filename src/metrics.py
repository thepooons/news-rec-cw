import numpy as np


class Metrics(object):
    """Metrics class
    has:
        - ARHR
        - Precision@k
        - Recall@k
    """

    @staticmethod
    def ARHR(
        recommendation_list: np.array,
        positive_item_list: np.array,
        negative_item_list: np.array,
    ):
        arhr_ = []
        for item in positive_item_list:
            arhr_.append(1 / (list(recommendation_list).index(item) + 1))
        for item in negative_item_list:
            arhr_.append(
                list(recommendation_list).index(item) / len(recommendation_list)
            )
        return np.mean(arhr_)

    @staticmethod
    def precision_at_k(
        recommendation_list: np.array,
        positive_item_list: np.array,
        negative_item_list: np.array,
    ):
        # k = 10 because we can only recommend 10 items
        top_k_reccomendations = set(recommendation_list[:10])
        relevant_items = set(positive_item_list)
        return len(top_k_reccomendations.intersection(relevant_items)) / len(
            top_k_reccomendations
        )

    @staticmethod
    def recall_at_k(
        recommendation_list: np.array,
        positive_item_list: np.array,
        negative_item_list: np.array,
    ):
        # k = 10 because we can only recommend 10 items
        top_k_reccomendations = set(recommendation_list[:10])
        relevant_items = set(positive_item_list)
        return len(top_k_reccomendations.intersection(relevant_items)) / len(
            relevant_items
        )
