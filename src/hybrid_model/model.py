import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras


class MfHybridModel(object):
    """
    Class for hybrid model object

    Args:
        num_user (int): The total number of users in the full data

        item_dim (int): The dimension of item representation. Default is 100

        comb_type (string): The type of combination layer to user add | concat. Default is concat

        embed_dim (int): The size of embedding layers. Defaut is 100

        lr (float): The learning for the model
    """

    def __init__(
        self,
        num_user,
        item_dim=100,
        comb_type="concat",
        embed_dim=100,
        lr=0.0001,
        user_pretrained=None,
    ):
        # Initialize the instance variables
        self.num_user = num_user
        self.item_dim = item_dim
        self.comb_type = comb_type
        self.embed_dim = embed_dim
        self.user_pretrained = user_pretrained
        self.lr = lr

    def get_model(self):
        # Return the model
        input_user_id = keras.layers.Input(shape=(1,), name="input_1")
        input_item_id = keras.layers.Input(
            shape=(self.item_dim,), name="input_2")

        if self.user_pretrained == None:
            # Create the embedding layers
            embedding_user_gmf = keras.layers.Embedding(
                input_dim=self.num_user,
                output_dim=self.embed_dim,
                embeddings_initializer="he_normal",
                embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            )(input_user_id)

            embedding_user_mlp = keras.layers.Embedding(
                input_dim=self.num_user,
                output_dim=self.embed_dim,
                embeddings_initializer="he_normal",
                embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            )(input_user_id)
        else:
            # Create the embedding layers
            embedding_user_gmf = keras.layers.Embedding(
                input_dim=self.num_user,
                output_dim=self.embed_dim,
                weights=[self.user_pretrained[0]],
                embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            )(input_user_id)

            embedding_user_mlp = keras.layers.Embedding(
                input_dim=self.num_user,
                output_dim=self.embed_dim,
                weights=[self.user_pretrained[1]],
                embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
            )(input_user_id)

        # GMF and its optimal shape
        flatten_user_gmf = keras.layers.Flatten()(embedding_user_gmf)
        flatten_item_gmf = keras.layers.Flatten()(input_item_id)
        flatten_item_gmf = keras.layers.Dense(
            units=self.embed_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        )(flatten_item_gmf)

        flatten_item_gmf = keras.layers.Dense(
            units=self.embed_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        )(flatten_item_gmf)

        gmf_embed = keras.layers.Multiply()(
            [flatten_user_gmf, flatten_item_gmf])

        # MLP and option available
        flatten_user_mlp = keras.layers.Flatten()(embedding_user_mlp)
        flatten_item_mlp = keras.layers.Flatten()(input_item_id)
        flatten_item_mlp = keras.layers.Dense(
            units=self.embed_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        )(flatten_item_mlp)

        flatten_item_mlp = keras.layers.Dense(
            units=self.embed_dim,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
        )(flatten_item_mlp)

        if self.comb_type == "concat":
            mlp_embed = keras.layers.Concatenate()(
                [flatten_user_mlp, flatten_item_mlp])
        elif self.comb_type == "add":
            mlp_embed = keras.layers.Add()(
                [flatten_user_mlp, flatten_item_mlp])
        else:
            raise Exception(
                "Invalid comb type ==> %s | options ==> [concat, add]"
                % (self.comb_type)
            )

        # MLP Dense layers
        mlp_x = keras.layers.Dense(
            units=512, activation="relu", kernel_regularizer=keras.regularizers.l1(1e-6)
        )(mlp_embed)
        mlp_x = keras.layers.BatchNormalization()(mlp_x)
        mlp_x = keras.layers.Dropout(0.3)(mlp_x)

        mlp_x = keras.layers.Dense(
            units=256, activation="relu", kernel_regularizer=keras.regularizers.l1(1e-6)
        )(mlp_x)
        mlp_x = keras.layers.BatchNormalization()(mlp_x)
        mlp_x = keras.layers.Dropout(0.2)(mlp_x)

        mlp_x = keras.layers.Dense(
            units=128, activation="relu", kernel_regularizer=keras.regularizers.l1(1e-6)
        )(mlp_x)
        mlp_x = keras.layers.BatchNormalization()(mlp_x)
        mlp_x = keras.layers.Dropout(0.1)(mlp_x)

        # Final merge
        merged = keras.layers.Concatenate()([gmf_embed, mlp_x])

        # Create the dense net
        x = keras.layers.Dense(
            units=1, kernel_initializer="lecun_uniform", activation="relu"
        )(merged)

        # Create the model
        model = keras.models.Model(
            inputs=[input_user_id, input_item_id], outputs=[x])
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss=keras.losses.MeanSquaredError(),
            metrics=keras.metrics.RootMeanSquaredError(),
        )

        # Returnt the model
        return model
