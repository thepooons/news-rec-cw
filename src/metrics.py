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
        recommendation_list,
        user_data_train,
        user_data_test,
    ):
        user_average_time_spent = np.mean(user_data_train.loc[:, "time_spent"])

        arhr_ = []
        for _, item in user_data_test.iterrows():
            max_time_spent = max(user_data_train.loc[:, "time_spent"])
            left_slope = (0 - 1) / (user_average_time_spent -
                                    0 + 0.000000001)  # always negative
            # always positive
            right_slope = (1 - 0) / (max_time_spent -
                                     user_average_time_spent + 0.000000001)

            if item["time_spent"] > user_average_time_spent:
                weight = right_slope * \
                    (item["time_spent"] - user_average_time_spent)
                arhr_.append(
                    (1 /
                     (list(recommendation_list).index(item["article_id"]) + 1)) * weight
                )
            else:
                weight = -left_slope * \
                    (user_average_time_spent - item["time_spent"])
                arhr_.append(
                    (list(recommendation_list).index(
                        item["article_id"]) / len(recommendation_list)) * weight
                )

        return np.mean(arhr_)

    @staticmethod
    def precision_at_k(
        recommendation_list,
        user_data_train,
        user_data_test,
    ):
        user_average_time_spent = np.mean(user_data_train.loc[:, "time_spent"])

        positive_item_list = user_data_test.loc[
            user_data_test.loc[:,
                               "time_spent"] >= user_average_time_spent, "article_id"
        ].values

        # k = 10 because we can only recommend 10 items
        top_k_recommendations = set(recommendation_list[:10])
        relevant_items = set(positive_item_list)
        if len(relevant_items) == 0:
            return None
        return len(top_k_recommendations.intersection(relevant_items)) / len(
            top_k_recommendations
        )

    @staticmethod
    def recall_at_k(
        recommendation_list,
        user_data_train,
        user_data_test,
    ):
        user_average_time_spent = np.mean(user_data_train.loc[:, "time_spent"])

        positive_item_list = user_data_test.loc[
            user_data_test.loc[:,
                               "time_spent"] >= user_average_time_spent, "article_id"
        ].values

        # k = 10 because we can only recommend 10 items
        top_k_recommendations = set(recommendation_list[:10])
        relevant_items = set(positive_item_list)
        if len(relevant_items) == 0:
            return None
        return len(top_k_recommendations.intersection(relevant_items)) / len(
            relevant_items
        )
