import numpy as np
import pandas as pd
import surprise


class SurpriseDataset(object):
    """
    Create a surprise dataset classes
    """

    def __init__(self, is_train):
        # Initialize the instance variables
        self.is_train = is_train

    def get_surprise_dataset(self, scale, data):
        """
        SRPRISE COMPATIABLE DATASET
        """
        # Make check assertions
        assert isinstance(data, pd.DataFrame), \
            "data object not of type dataframe"
        assert 'user_id' in data.columns, "user_id not found"
        assert 'item_id' in data.columns, "article_id not found"
        assert 'rating' in data.columns, "click/time not found"

        # Create the reader scale
        reader = surprise.Reader(rating_scale=scale)

        # Load from dataframe
        dataset = surprise.Dataset.load_from_df(data[['user_id', 'item_id', 'rating']],
                                                reader=reader)

        # Convert to train dataset
        dataset = dataset.build_full_trainset()

        # Check if test or not
        if self.is_train == False:
            # Create the test set and return
            dataset = dataset.build_testset()

        # return the dataset
        return dataset
