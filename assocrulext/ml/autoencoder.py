from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.base import TransformerMixin
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.losses import mean_squared_error
class AutoEncoderDimensionReduction(TransformerMixin):
    def __init__(self, encoding_dim, epochs, batch_size, lr=1e-2,novelty_score=None):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.novelty_score=novelty_score
        

    def loss_function(self, X, Y):
        X = tf.cast(X, tf.float32)
        Y = tf.cast(Y, tf.float32)
        #reconstruction_loss = tf.reduce_mean(tf.square(X - Y))
        reconstruction_loss = mean_squared_error(X, Y)
        average_novelty_score = tf.reduce_mean(tf.cast(self.novelty_score, tf.float32))
        total_loss = average_novelty_score * reconstruction_loss
        return total_loss

    # def loss_function(self, y_true, y_pred):
        #y_true = tf.cast(y_true, tf.float32)
        #reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Flatten the support values and take the mean
        #support_values = tf.concat(self.support, axis=0)
        #support_values = tf.cast(support_values, tf.float32)
        #support_penalty = tf.reduce_mean(support_values)

        #total_loss = self.novelty_score + support_penalty + reconstruction_loss
        #return total_loss

    
    def fit(self, X, y=None):
        input_dim = X.shape[1]
        encoding_dim = self.encoding_dim

        # `input_layer` is defining the input layer of the autoencoder neural network. It is a
        # placeholder for the input data, with the shape of `(input_dim,)`, where `input_dim` is the
        # number of features in the input data.
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(1024, activation="tanh")(input_layer)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(512, activation="tanh")(input_layer)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(256, activation="tanh")(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(128, activation="tanh")(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(encoding_dim, activation="tanh")(encoder)
        encoder_model = Model(inputs=input_layer, outputs=encoder)

        # Create the decoder
        decoder = Dense(128, activation="tanh")(encoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.2)(decoder)
        decoder = Dense(256, activation="tanh")(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.2)(decoder)
        decoder = Dense(512, activation="tanh")(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.2)(decoder)
        decoder = Dense(1024, activation="tanh")(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(0.2)(decoder)
        decoder = Dense(input_dim, activation="sigmoid")(decoder)
        decoder_model = Model(inputs=encoder, outputs=decoder)

        # Combine the encoder and decoder to create the autoencoder
        autoencoder = Model(
            inputs=input_layer, outputs=decoder_model(encoder_model(input_layer))
        )

        optimizer = Adam(lr=self.lr)
        # Compile the model
        autoencoder.compile(optimizer=optimizer, loss=self.loss_function)

        # Train the model
        autoencoder.fit(
            X, X, epochs=self.epochs, batch_size=self.batch_size, shuffle=True 
        )

        self.encoder_model = encoder_model

        # Return the transformed data
        return self.encoder_model.predict(X)

    def transform(self, X):
        return self.encoder_model.predict(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
