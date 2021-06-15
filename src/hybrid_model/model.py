import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


class MfHybridModel(object):
    """
    Returns a model based
    """

    def __init__(self, num_user, item_dim=50, comb_type="concat", embed_dim=100, lr=0.0001):
        # Initialize the instance variables
        self.num_user = num_user
        self.item_dim = item_dim
        self.comb_type = comb_type
        self.embed_dim = embed_dim
        self.lr = lr

    def get_model(self):
        # Return the model
        input_user_id = keras.layers.Input(shape=(1,), name="input_1")
        input_item_id = keras.layers.Input(
            shape=(self.item_dim,), name="input_2")

        # Create the embedding layers
        embedding_user_gmf = keras.layers.Embedding(input_dim=self.num_user, output_dim=self.embed_dim,
                                                    embeddings_initializer="he_normal",
                                                    embeddings_regularizer=tf.keras.regularizers.l2(1e-6))(input_user_id)

        embedding_user_mlp = keras.layers.Embedding(input_dim=self.num_user, output_dim=self.embed_dim,
                                                    embeddings_initializer="he_normal",
                                                    embeddings_regularizer=tf.keras.regularizers.l2(1e-6))(input_user_id)

        # GMF and its optimal shape
        flatten_user_gmf = keras.layers.Flatten()(embedding_user_gmf)
        flatten_item_gmf = keras.layers.Flatten()(input_item_id)
        flatten_item_gmf = keras.layers.Dense(units=self.embed_dim, activation="relu",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-6))(flatten_item_gmf)
        gmf_embed = keras.layers.Multiply()(
            [flatten_user_gmf, flatten_item_gmf])

        # MLP and option available
        flatten_user_mlp = keras.layers.Flatten()(embedding_user_mlp)
        flatten_item_mlp = keras.layers.Flatten()(input_item_id)
        flatten_item_mlp = keras.layers.Dense(units=self.embed_dim, activation="relu",
                                              kernel_regularizer=tf.keras.regularizers.l2(1e-6))(flatten_item_mlp)

        if self.comb_type == "concat":
            mlp_embed = keras.layers.Concatenate()(
                [flatten_user_mlp, flatten_item_mlp])
        elif self.comb_type == "add":
            mlp_embed = keras.layers.Add()(
                [flatten_user_mlp, flatten_item_mlp])
        else:
            raise Exception(
                "Invalid comb type ==> %s | options ==> [concat, add]" % (self.comb_type))

        # MLP Dense layers
        mlp_x = keras.layers.Dense(units=512, activation="relu",
                                   kernel_regularizer=keras.regularizers.l1(1e-6))(mlp_embed)
        mlp_x = keras.layers.BatchNormalization()(mlp_x)
        mlp_x = keras.layers.Dropout(0.3)(mlp_x)

        mlp_x = keras.layers.Dense(units=256, activation="relu",
                                   kernel_regularizer=keras.regularizers.l1(1e-6))(mlp_x)
        mlp_x = keras.layers.BatchNormalization()(mlp_x)
        mlp_x = keras.layers.Dropout(0.2)(mlp_x)

        mlp_x = keras.layers.Dense(units=128, activation="relu",
                                   kernel_regularizer=keras.regularizers.l1(1e-6))(mlp_x)
        mlp_x = keras.layers.BatchNormalization()(mlp_x)
        mlp_x = keras.layers.Dropout(0.1)(mlp_x)

        # Final merge
        merged = keras.layers.Concatenate()([gmf_embed, mlp_x])

        # Create the dense net
        x = keras.layers.Dense(
            units=1, kernel_initializer="lecun_uniform")(merged)

        # Create the model
        model = keras.models.Model(
            inputs=[input_user_id, input_item_id], outputs=[x])
        model.compile(optimizer=keras.optimizers.Adam(self.lr),
                      loss=keras.losses.MeanSquaredError(),
                      metrics=keras.metrics.RootMeanSquaredError())

        # Returnt the model
        return model