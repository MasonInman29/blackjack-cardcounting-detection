import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from helper import get_hand_value_and_soft_aces_count

class Hit_By_Hit_Transformer(BaseEstimator, TransformerMixin):
    """
    Transformer class to transform the Kaggle dataset of hands into features
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        """
        No fitting is required since the EVs are precomputed.
        """
        return self
    def transform(self, X):
        """
        Transforms the Kaggle Dataset with EV into features for ML models
        The features for each sample are:
            - DealerUpCard: The card the player can see the dealer has
            - PlayerScore: The maximum score the player starts with
            - HiLoScore: The precomputed HiLo score from the HiLo algorithm
            - NumOfSoftAces: The number of soft aces (value=11)
            - CardsRemaining: Amount of cards left in the deck
            - _Frequency: The percent of remaining cards with value _

        :param X: Kaggle dataset with EV
        :return: Feature vector for ML training
        """

        return pd.DataFrame({
            "DealerUpCard": X["dealer_up"],
            "PlayerScore": [get_hand_value_and_soft_aces_count(sample["initial_hand"])[0]
                            for _, sample in X.iterrows()],
            #"HiLoScore": X["run_count"],
            "NumOfSoftAces": [get_hand_value_and_soft_aces_count(sample["initial_hand"])[1]
                              for _, sample in X.iterrows()],
            "CardsRemaining": X["cards_remaining"],
            "AceFrequency": [sample["remaining_card_counts"][11] / sample["cards_remaining"]
                             for _, sample in X.iterrows()],
            "TwoFrequency": [sample["remaining_card_counts"][2] / sample["cards_remaining"]
                             for _, sample in X.iterrows()],
            "ThreeFrequency": [sample["remaining_card_counts"][3] / sample["cards_remaining"]
                               for _, sample in X.iterrows()],
            "FourFrequency": [sample["remaining_card_counts"][4] / sample["cards_remaining"]
                              for _, sample in X.iterrows()],
            "FiveFrequency": [sample["remaining_card_counts"][5] / sample["cards_remaining"]
                              for _, sample in X.iterrows()],
            "SixFrequency": [sample["remaining_card_counts"][6] / sample["cards_remaining"]
                             for _, sample in X.iterrows()],
            "SevenFrequency": [sample["remaining_card_counts"][7] / sample["cards_remaining"]
                               for _, sample in X.iterrows()],
            "EightFrequency": [sample["remaining_card_counts"][8] / sample["cards_remaining"]
                               for _, sample in X.iterrows()],
            "NineFrequency": [sample["remaining_card_counts"][9] / sample["cards_remaining"]
                              for _, sample in X.iterrows()],
            "TenFrequency": [sample["remaining_card_counts"][10] / sample["cards_remaining"]
                             for _, sample in X.iterrows()],
        })