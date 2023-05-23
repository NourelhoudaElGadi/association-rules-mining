from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.base import TransformerMixin

import tensorflow as tf


def softplus(x):
    return tf.math.log(tf.math.exp(x) + 1)


def mish(x):
    return x * tf.math.tanh(softplus(x))


class AutoEncoderDimensionReduction(TransformerMixin):
    def __init__(self, encoding_dim, epochs, batch_size, lr=1e5):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        encoding_dim = self.encoding_dim

        # `input_layer` is defining the input layer of the autoencoder neural network. It is a
        # placeholder for the input data, with the shape of `(input_dim,)`, where `input_dim` is the
        # number of features in the input data.
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(512, activation="tanh")(input_layer)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.1)(encoder)
        encoder = Dense(256, activation="relu")(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.1)(encoder)
        encoder = Dense(128, activation="relu")(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.1)(encoder)
        encoder = Dense(encoding_dim, activation="relu")(encoder)
        encoder = Model(inputs=input_layer, outputs=encoder)

        # Create the decoder
        decoder = Dense(128, activation=mish)(encoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.1)(decoder)
        decoder = Dense(256, activation=mish)(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.1)(decoder)
        decoder = Dense(512, activation=mish)(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.1)(decoder)
        decoder = Dense(input_dim, activation=mish)(decoder)
        decoder = Model(inputs=encoder, outputs=decoder)

        # Combine the encoder and decoder to create the autoencoder
        autoencoder = Model(inputs=input_layer, outputs=decoder(encoder(input_layer)))

        optimizer = Adam(lr=self.lr)
        # Compile the model
        autoencoder.compile(optimizer=optimizer, loss="mean_squared_error")

        # Train the model
        autoencoder.fit(
            X, X, epochs=self.epochs, batch_size=self.batch_size, shuffle=True
        )

        self.encoder = encoder

        # Return the transformed data
        return self.encoder.predict(X)

    def transform(self, X):
        return self.encoder.predict(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
