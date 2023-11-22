import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

def load_data(path):
    data = pd.read_csv(path)
    features = data.drop(["quality", "Id"], axis=1)
    target = data["quality"]
    return train_test_split(features, target, test_size=0.2, random_state=42)


class Model(tf.keras.Model):

    def __init__(self, data_path):
        super(Model, self).__init__()
        self.data = pd.read_csv(data_path)
        features = self.data.drop(["quality", "Id"], axis=1)
        target = self.data["quality"]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.values)
        # X_test_scaled = self.scaler.transform(X_test)

        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dense(10),
            layers.Dense(1)  # Output layer for regression task
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(1,)),
            layers.Dense(32),
            layers.Dense(X_train_scaled.shape[1])
        ])

        # Compile the model
        # self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        # self.decoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        self.compile(optimizer='adam', loss="mean_squared_error", metrics=["mse"])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train(self, path):
        X_train, X_test, y_train, y_test = load_data(path)
        X_train_scaled = self.scaler.transform(X_train.values)
        # Train the model
        history = self.fit(X_train_scaled, X_train, epochs=20, batch_size=8, validation_split=0.1)
        y_pred = self.encoder.predict(X_train_scaled)
        mse = mean_squared_error(y_train, y_pred)
        print(f"Encoder mse: {mse}")
        y_pred = self.decoder.predict(y_train)
        mse = mean_squared_error(X_train, y_pred)
        print(f"Decoder mse: {mse}")
        return mse

    def decode(self, value):
        value = np.array([[value]])
        return self.decoder.predict(value)

    def save(self, path):
        self.encoder.save(path)

    def load(self, path):
        self.encoder = tf.keras.models.load_model(path)

    def predict(self, wine):
        df = pd.Series(json.loads(wine.json()))
        X = df.drop("quality").values
        values = self.scaler.transform([X])
        res = self.encoder.predict(values, verbose=0)[0]
        return res

    def to_json(self):
        model_config = self.encoder.get_config()
        model_weights = [w.tolist() for w in self.encoder.get_weights()]
        return json.dumps({'config': model_config, 'weights': model_weights})


if __name__ == "__main__":
    wine = {
        "fixed_acidity": -111.3040542602539,
        "volatile_acidity": -6.910439491271973,
        "citric_acid": -3.291638135910034,
        "residual_sugar": -31.249847412109375,
        "chlorides": -1.4252290725708008,
        "free_sulfur_dioxide": -27,
        "total_sulfur_dioxide": -78,
        "density": -15.060705184936523,
        "ph": -44.447513580322266,
        "sulphates": -8.248477935791016,
        "alcohol": -136.79942321777344,
    }

    model = Model("app/model/data.csv")
