import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from Hit_By_Hit_Transformer import Hit_By_Hit_Transformer
import numpy as np
from dataset import GameSimulator
from dataset.dataset import CSVDataset
from eval_model import evaluate_model


class NN_BlackJack:
    """
    Class for a Deep Neural Network Hit/Stay classifier.
    """
    def __init__(self, hidden_layer_sizes=(256, 128, 64, 32, 16), activation='relu', solver='adam', max_iter=1000):
        """
        Creates a pipeline for the hit/stay classifier
        """
        self.le = LabelEncoder()
        self.pipeline = Pipeline([
            ("transformed_data", Hit_By_Hit_Transformer()),
            ("standardized_data", ColumnTransformer([
                ("DealerUpCard", OneHotEncoder(), ["DealerUpCard"]),
                ("PlayerScore", StandardScaler(), ["PlayerScore"]),
                #("HiLoScore", StandardScaler(), ["HiLoScore"]),
                ("NumOfSoftAces", StandardScaler(), ["NumOfSoftAces"]),
                ("CardsRemaining", StandardScaler(), ["CardsRemaining"]),
                ("AceFrequency", "passthrough", ["AceFrequency"]),
                ("TwoFrequency", "passthrough", ["TwoFrequency"]),
                ("ThreeFrequency", "passthrough", ["ThreeFrequency"]),
                ("FourFrequency", "passthrough", ["FourFrequency"]),
                ("FiveFrequency", "passthrough", ["FiveFrequency"]),
                ("SixFrequency", "passthrough", ["SixFrequency"]),
                ("SevenFrequency", "passthrough", ["SevenFrequency"]),
                ("EightFrequency", "passthrough", ["EightFrequency"]),
                ("NineFrequency", "passthrough", ["NineFrequency"]),
                ("TenFrequency", "passthrough", ["TenFrequency"]),
            ])),
            ("mlp", MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                  activation=activation,
                                  solver=solver,
                                  max_iter=max_iter,
                                  random_state=42)),
        ])
        self.fitted = False

    def fit(self, X, y):
        """
        Trains a deep neural network on the Kaggle Dataset with EVs
        :param X: The dataset from Kaggle with computed EVs
        :param y: The labels of the highest EV
        """
        encoded_y = self.le.fit_transform(y)
        self.pipeline.fit(X, encoded_y)
        self.fitted = True

    def predict(self, player_hand, dealer_up_card, remaining_cards):
        """
        Prediction used in game_simulator to recieve the player's hand, the dealer's up card, and the remaining cards in the shoe
        """
        if not self.fitted:
            raise ValueError("This model has not been fitted yet.")

        sample = pd.DataFrame({"dealer_up": [dealer_up_card],
                               "initial_hand": [player_hand],
                               "cards_remaining": [sum(remaining_cards.values())],
                              "remaining_card_counts": [remaining_cards]
        })
        prediction = np.argmax(self.pipeline.predict_proba(sample), axis=1)
        label = self.le.inverse_transform(prediction)
        return label[0]

    def predict_proba(self, X):
        """
        Prediction function for testing the ML model. Uses the softmax values for multi-class classification on a list of samples
        :param X: Dataset from Kaggle with EVs computed
        :return: Probabilities of each class for each sample
        """
        if not self.fitted:
            raise ValueError("This model has not been fitted yet.")
        return self.pipeline.predict_proba(X)

    def score(self, X, y):
        """
        Compute Accuracy score of the ML model
        :param X: Dataset from Kaggle with EVs computed
        :param y: The labels of the highest EV
        :return: The accuracy score
        """
        encoded_y = self.le.transform(y)
        return self.pipeline.score(X, encoded_y)

    def get_bet_size(self, remaining_cards):
        """Always bets the table minimum."""
        return 1.0

if __name__ == "__main__":
    dataset = CSVDataset()
    train_X = dataset.get_split("train")
    train_y = train_X["best_action_by_ev"]
    test_X = dataset.get_split("test")
    test_y = test_X["best_action_by_ev"]

    print("Beginning Training")
    nn = NN_BlackJack()
    nn.fit(train_X, train_y)
    print("Done Training")

    filepath = "nn_2000sim_3000000samples.pkl"
    with open(filepath, "wb") as file:
        pickle.dump(nn, file)

    game_simulator = GameSimulator()
    evaluate_model(nn, dataset, game_simulator, num_simulations=1000)