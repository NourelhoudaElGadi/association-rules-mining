from keras.layers import Input, Dense
from keras.models import Model
from sklearn.base import TransformerMixin

class AutoEncoderDimentionReduction(TransformerMixin):
    def __init__(self, encoding_dim, epochs, batch_size, activation, activation_output):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.activation_output = activation_output

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        encoding_dim = self.encoding_dim

        input_layer = Input(shape=(input_dim,))
        encoder_layer_1 = Dense(512, activation=self.activation)(input_layer)
        encoder_layer_2 = Dense(256, activation=self.activation)(encoder_layer_1)
        encoder_layer_3 = Dense(128, activation=self.activation)(encoder_layer_2)
        encoder_layer_4 = Dense(encoding_dim, activation=self.activation)(encoder_layer_3)
        encoder = Model(inputs=input_layer, outputs=encoder_layer_4)

        # Create the decoder
        decoder_layer_1 = Dense(128, activation=self.activation_output)(encoder_layer_4)
        decoder_layer_2 = Dense(256, activation=self.activation_output)(decoder_layer_1)
        decoder_layer_3 = Dense(512, activation=self.activation_output)(decoder_layer_2)
        decoder_layer_4 = Dense(input_dim, activation=self.activation_output)(decoder_layer_3)
        decoder = Model(inputs=encoder_layer_4, outputs=decoder_layer_4)

        # Combine the encoder and decoder to create the autoencoder
        autoencoder = Model(inputs=input_layer, outputs=decoder(encoder(input_layer)))

        # Compile the model
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        autoencoder.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, shuffle=True)

        self.encoder = encoder

        # Return the transformed data
        return self.encoder.predict(X)

    def transform(self, X):
        return self.encoder.predict(X)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y)




