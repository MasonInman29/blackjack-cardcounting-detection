import pickle
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from model.Hit_By_Hit_Transformer import Hit_By_Hit_Transformer
from dataset import GameSimulator
from dataset.dataset import CSVDataset
from eval_model import evaluate_model
from datetime import datetime
import eval_model
from sklearn.model_selection import cross_val_score

run_dir = os.path.join('figures', 'LogisticRegression_BlackJack', datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
os.makedirs(run_dir, exist_ok=True)
eval_model.OUTPUT_DIR = run_dir

class LogisticRegression_BlackJack:

    def __init__(self, C: float = 0.1, max_iter: int = 2000):
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
            ("lr", LogisticRegression(C=C,
                                      max_iter=max_iter,
                                      multi_class="multinomial",
                                      solver="lbfgs",
                                      n_jobs=None,
                                      class_weight="balanced", # Added balanced class weights
                                      penalty='l2',
            )),
        ])

        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the logistic regression on the Kaggle dataset with EV labels.
        :param X: DataFrame
        :param y: Action labels ('H', 'S', 'D', 'R')
        """
        encoded_y = self.le.fit_transform(y)
        self.pipeline.fit(X, encoded_y)
        self.fitted = True

    def predict(self, player_hand, dealer_up_card, remaining_cards):
        """
        Predict a single action for the simulator interface.
        """
        if not self.fitted:
            raise ValueError("This model has not been fitted yet.")

        sample = pd.DataFrame({
            "dealer_up": [dealer_up_card],
            "initial_hand": [player_hand],
            "cards_remaining": [sum(remaining_cards.values())],
            "remaining_card_counts": [remaining_cards],
        })

        encoded_pred = self.pipeline.predict(sample)
        label = self.le.inverse_transform(encoded_pred)
        return label[0]

    def predict_proba(self, X: pd.DataFrame):
        if not self.fitted:
            raise ValueError("This model has not been fitted yet.")
        return self.pipeline.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        if not self.fitted:
            raise ValueError("This model has not been fitted yet.")
        encoded_y = self.le.transform(y)
        return self.pipeline.score(X, encoded_y)

    def get_bet_size(self, remaining_cards) -> float:
        return 1.0


if __name__ == "__main__":
    dataset = CSVDataset()
    train_X = dataset.get_split("train")
    train_y = train_X["best_action_by_ev"]
    test_X = dataset.get_split("test")
    test_y = test_X["best_action_by_ev"]
    
    print(f"Training set size: {len(train_X)}")
    print(f"Test set size: {len(test_X)}")

    best_c = 1.0
    best_acc = 0.0

    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression_BlackJack(C=c)
        model.fit(train_X, train_y)
        acc = model.score(test_X, test_y)
        print(f"C={c}: test accuracy = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_c = c
    print(f"\nBest C value: {best_c} with accuracy {best_acc:.4f}")


    print("Beginning Training (Logistic Regression)")
    
    logreg = LogisticRegression_BlackJack(C=best_c)
    logreg.fit(train_X, train_y)
    print("Done Training")
    
    # Test accuracy
    train_acc = logreg.score(train_X, train_y)
    test_acc = logreg.score(test_X, test_y)
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", "logreg_model.pkl")
    with open(filepath, "wb") as file:
        pickle.dump(logreg, file)
    print(f"Model saved to {filepath}")
    
    # Evaluate with game simulator
    print("\nRunning game simulations...")
    game_simulator = GameSimulator()
    evaluate_model(logreg, dataset, game_simulator, num_simulations=10000)


