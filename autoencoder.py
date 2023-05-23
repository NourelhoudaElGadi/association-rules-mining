from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from sklearn.base import TransformerMixin


class AutoEncoderDimensionReduction(TransformerMixin):
    def __init__(self, encoding_dim, epochs, batch_size, lr=1e5):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        encoding_dim = self.encoding_dim

        input_layer = Input(shape=(input_dim,))
        encoder_layer_1 = Dense(512, activation="tanh")(input_layer)
        encoder_layer_2 = Dense(256, activation="relu")(encoder_layer_1)
        encoder_layer_3 = Dense(128, activation="relu")(encoder_layer_2)
        encoder_layer_4 = Dense(encoding_dim, activation="relu")(encoder_layer_3)
        encoder = Model(inputs=input_layer, outputs=encoder_layer_4)

        # Create the decoder
        decoder_layer_1 = Dense(128, activation="tanh")(encoder_layer_4)
        decoder_layer_2 = Dense(256, activation="relu")(decoder_layer_1)
        decoder_layer_3 = Dense(512, activation="relu")(decoder_layer_2)
        decoder_layer_4 = Dense(input_dim, activation="tanh")(decoder_layer_3)
        decoder = Model(inputs=encoder_layer_4, outputs=decoder_layer_4)

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
