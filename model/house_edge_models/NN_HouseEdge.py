import pickle
import torch
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Add
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

class NN_HouseEdge:
    def __init__(self, training_data_filepath):
        self.X,self.y = self.get_data(training_data_filepath)
        self.nn_models = self.fit(self.X, self.y, folds=10)


    def get_data(self, training_data_filepath):
        train = pd.read_csv(training_data_filepath)

        label_column = 'ev'
        X = train.drop([label_column, "id"], axis=1).copy()
        y = train[label_column].copy()
        return X,y

    def fit(self, X, y, folds=10):
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        oof_pred_nn = np.zeros(len(X))
        fold_mse_nn = []
        nn_collection = []
        nn_weights_collection = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            print(f"Training fold {fold} ...")

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            nn_val_preds_list = []

            for rep in range(3):
                seed = 42 + rep
                np.random.seed(seed)
                tf.random.set_seed(seed)

                nn_model = Sequential()
                nn_model.add(Dense(256, activation='relu', input_dim=X_train_scaled.shape[1]))
                nn_model.add(Dense(256, activation='relu'))
                nn_model.add(Dense(256, activation='relu'))
                nn_model.add(Dense(256, activation='relu'))
                nn_model.add(Dense(128, activation='relu'))
                nn_model.add(Dense(128, activation='relu'))
                nn_model.add(Dense(1, activation='linear'))
                nn_model.compile(optimizer='adam', loss='mean_squared_error')

                early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=0)
                nn_model.fit(X_train_scaled, y_train,
                             validation_data=(X_val_scaled, y_val),
                             epochs=1000,
                             batch_size=32,
                             callbacks=[early_stop],
                             verbose=1)
                nn_collection.append(nn_model)
                nn_weights_collection.append(nn_model.get_weights())

                nn_val_pred_rep = nn_model.predict(X_val_scaled).flatten()
                nn_val_preds_list.append(nn_val_pred_rep)

            nn_val_pred = np.mean(nn_val_preds_list, axis=0)

            mse_nn = mean_squared_error(y_val, nn_val_pred)
            fold_mse_nn.append(mse_nn)
            oof_pred_nn[val_idx] = nn_val_pred

            print(f"Fold {fold} MSE (NN): {mse_nn:.8f}")


        overall_nn_mse = mean_squared_error(y, oof_pred_nn)

        filepath_weights = "nn_house_edge_weights.pkl"
        with open(filepath_weights, "wb") as file:
            pickle.dump(nn_weights_collection, file)

        filepath_models = "nn_house_edge_models.pkl"
        with open(filepath_models, "wb") as file:
            pickle.dump(nn_collection, file)

        return nn_collection

if __name__ == "__main__":
    models=NN_HouseEdge()

    filepath_models = "nn_house_edge_models.pkl"
    with open(filepath_models, "rb") as file:
            models = pickle.load(file)
    remaining_card_count = pd.DataFrame([{1: 15, 2: 3, 3:5, 4:10, 5:9, 6:12, 7:8, 8:8, 9:5, 10:20}])
    predictions = []
    for nn in models:
        predictions.append(nn.predict(remaining_card_count))
    print(np.mean(predictions))