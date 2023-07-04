from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


class DR:
    def __init__(self):
       pass

    def create_pca_model(self, data, n_components=0, random_state=0):
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(data)

    def create_autoencoder_model(self,data,outputs):
        """
        :param data:
        :param outputs:
        :return:
        Create an autoencoder model
        """
        input_layer = Input(shape=(data.shape[1],))
        encoder = Dense(10, activation="relu")(input_layer)
        encoder = Dense(7, activation="relu")(encoder)
        encoder = Dense(5, activation="relu")(encoder)
        encoder = Dense(3, activation="relu")(encoder)
        encoder = Dense(outputs, activation="relu")(encoder)
        decoder = Dense(3, activation="relu")(encoder)
        decoder = Dense(5, activation="relu")(decoder)
        decoder = Dense(7, activation="relu")(decoder)
        decoder = Dense(10, activation="relu")(decoder)
        decoder = Dense(data.shape[1], activation="sigmoid")(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer="adam", loss="mse")
        autoencoder.fit(data, data, epochs=10, batch_size=32, verbose=0)
        encoder = Model(inputs=input_layer, outputs=encoder)
        return encoder.predict(data)
