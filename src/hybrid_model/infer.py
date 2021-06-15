import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


def infer(model, all_ids_data, train_data, user_id):
    """
    Make predictions on all the data 

    Returns
    _______

    1. All predictions ranking 
    2. Prediction to give to the user by seperating the already watched articles
    """
    # Make the columns
    cols_article_data = ["heading_%d" % i for i in range(50)]

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices(({"input_1": np.array([user_id] * len(all_ids_data)),
                                                   "input_2": all_ids_data[cols_article_data].values})).batch(128)

    # Pass the dataset and fetch predictions
    prediction = model.predict(dataset)

    # Sort the predictions
    mapped_data = pd.DataFrame({
        "article_id": all_ids_data["id"],
        "preds": prediction.tolist()
    })

    # sort the data
    sorted_data = mapped_data.sort_values(by="preds", ascending=False)

    # Collect the data that has not been watched
    watched_articles = np.unique(
        train_data[train_data["user_id"] == user_id]["article_id"])
    not_watched = list(set(all_ids_data["id"]) - set(watched_articles))

    list_articles_not_watched = []
    for index, row in sorted_data.iterrows():
        if row["article_id"] in not_watched:
            list_articles_not_watched.append(row["article_id"])

    # Reuturn both the list
    return sorted_data, list_articles_not_watched
