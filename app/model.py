import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

class Model:

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        features = self.data.drop(["quality", "Id"], axis=1)
        target = self.data["quality"]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        # X_test_scaled = self.scaler.transform(X_test)

        self.model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            layers.Dense(10),
            layers.Dense(1)  # Output layer for regression task
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.transform(X_train)
        # Train the model
        history = self.model.fit(X_train_scaled, y_train, epochs=50, batch_size=2, validation_split=0.2)
        y_pred = self.model.predict(X_train_scaled)
        mse = mean_squared_error(y_train, y_pred)
        return mse

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, wine):
        df = pd.Series(json.loads(wine.json()))
        values = df.drop("quality").array
        res = self.model.predict(np.array([values]), verbose=0)[0]
        return res

    def to_json(self):
        model_config = self.model.get_config()
        model_weights = [w.tolist() for w in self.model.get_weights()]
        return json.dumps({'config': model_config, 'weights': model_weights})
