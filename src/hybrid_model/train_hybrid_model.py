import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


class TrainHybridModel(object):
    """
    Train for both API and evaluation
    """

    def __init__(self, model_obj, train_data, test_data, task_type="evaluation"):
        # Initialize the instance variables
        self.task_type = task_type
        self.train_data = train_data
        self.test_data = test_data
        self.model_obj = model_obj

    def train_model(self, batch_size, epochs, verbose=False):
        # Fetch the model beforehand
        hybrid_model = self.model_obj
        path = "weights/model_hybrid_%s.h5" % (self.task_type)

        # Check type of task
        if self.task_type == "api":
            # Specify the columns to take as features
            cols_article_data = ["heading_%d" % i for i in range(50)]
            cols_user_data = "user_id"

            # Start training on full data
            if self.test_data is not None:
                merged_data = pd.concat(
                    [self.train_data[cols_article_data + ["click", "time_spent", "user_id"]], self.test_data[cols_article_data + ["click", "time_spent", "user_id"]]], axis=0)
            else:
                merged_data = self.train_data

            # Collect the data
            article_data = merged_data[cols_article_data]
            user_data = merged_data[cols_user_data] - 1
            target_data = merged_data["time_spent"] * merged_data["click"]

            # Collect the required features and create the dataset
            train_dataset = tf.data.Dataset.from_tensor_slices(({"input_1": user_data,
                                                                 "input_2": article_data},
                                                                target_data.values)).batch(batch_size)

            # Train the model
            history = hybrid_model.fit(train_dataset,
                                       epochs=epochs,
                                       verbose=verbose)

            # Save the model
            hybrid_model.save(path)
            print("Model Saved.....")

        elif self.task_type == "evaluation":
            # Specify the columns to take as features
            cols_article_data = ["heading_%d" % i for i in range(50)]
            cols_user_data = "user_id"

            # Collect the data
            article_data_train = self.train_data[cols_article_data]
            user_data_train = self.train_data[cols_user_data] - 1
            target_data_train = self.train_data["time_spent"] * \
                self.train_data["click"]

            article_data_test = self.test_data[cols_article_data]
            user_data_test = self.test_data[cols_user_data] - 1
            target_data_test = self.test_data["time_spent"] * \
                self.test_data["click"]

            # Collect the required features and create the dataset
            train_dataset = tf.data.Dataset.from_tensor_slices(({"input_1": user_data_train,
                                                                 "input_2": article_data_train},
                                                                target_data_train.values)).batch(batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices(({"input_1": user_data_test,
                                                                "input_2": article_data_test},
                                                               target_data_test.values)).batch(batch_size)

            # Train the model
            history = hybrid_model.fit(train_dataset,
                                       validation_data=test_dataset,
                                       epochs=epochs,
                                       verbose=verbose)

            # Save the model
            hybrid_model.save(path)
            print("Model Saved.....")

        else:
            raise Exception("Invalid task type options ==> [evaluation, api]")

        return hybrid_model
