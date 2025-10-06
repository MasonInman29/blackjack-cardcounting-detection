import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from Hit_By_Hit_Transformer import Hit_By_Hit_Transformer
import numpy as np
from dataset import BlackjackDataset, GameSimulator
from eval_model import evaluate_model


class SVM_BlackJack:
    """
    Class for an SVM Hit/Stay classifier.
    Ensemble learning used.
    """
    def __init__(self, kernel="rbf"):
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
            ("ensemble_models", CalibratedClassifierCV(SVC(kernel=kernel))),
        ])
        self.fitted = False

    def fit(self, X, y):
        """
        Trains an ensemble SVM on the Kaggle Dataset with EVs
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
    dataset = BlackjackDataset(csv_path='../dataset/blackjack_simulator.csv', num_simulations=100, num_data_limit=5000)

    train_X = dataset.get_split('train')
    train_y = train_X["best_action_by_ev"]

    svm = SVM_BlackJack()
    svm.fit(train_X, train_y)

    game_simulator = GameSimulator()
    evaluate_model(svm, dataset, game_simulator, num_simulations=100)

    filepath = "svm_2000sim_5000000samples.pkl"
    with open(filepath, "wb") as file:
        # Use pickle.dump() to serialize the object and write it to the file
        pickle.dump(svm, file)

    print(f"Model saved to {filepath}")
