import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


def infer(model, all_ids_data, train_data, user_id):
    """
    Makes predictions on all of the data

    Args:
        model (tf.keras.model.Model): The pretrained model to make predictions

        all_ids_data (pd.DataFrame): The dataframe containing all the articles with their vector mapping

        train_data (pd.DataFrame): The training data that was used to train the model

        user_id (int): The user for whom predictions are being made


    Returns:
        [tuple]: tuple of all predictions and api predictions
    """
    # Make the columns
    cols_article_data = ["heading_%d" % i for i in range(100)]

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_1": np.array([user_id - 1] * len(all_ids_data)),
                "input_2": all_ids_data[cols_article_data].values,
            }
        )
    ).batch(128)

    # Pass the dataset and fetch predictions
    prediction = model.predict(dataset)

    # Sort the predictions
    mapped_data = pd.DataFrame(
        {"article_id": all_ids_data["article_id"],
            "preds": prediction.reshape(-1).tolist()}
    )

    # sort the data
    sorted_data = mapped_data.sort_values(by="preds", ascending=False)

    # Collect the data that has not been watched
    watched_articles = np.unique(
        train_data[train_data["user_id"] == int(user_id - 1)]["article_id"]).astype(int)
    not_watched = list(
        set(all_ids_data["article_id"].astype(int)) - set(watched_articles))

    list_articles_not_watched = []
    for index, row in sorted_data.iterrows():
        if int(row["article_id"]) in not_watched:
            list_articles_not_watched.append(int(row["article_id"]))

    # Reuturn both the list
    return sorted_data, list_articles_not_watched
