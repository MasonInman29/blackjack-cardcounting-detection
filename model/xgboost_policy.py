# model/xgb_blackjack.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, Booster, DMatrix

# Classes used when training
ACTION_MAP = {'H':0,'S':1,'D':2,'P':3,'R':4}
IDX2ACTION = {v:k for k,v in ACTION_MAP.items()}

# Fallback feature order (matches training snippet)
FEATURES = [
    'player_total','is_soft','is_pair','dealer_up',
    'cards_remaining','run_count','true_count',
    'p_ace','p_ten','p_low','hi_lo_density'
]

def _card_to_val(x):
    if x in ['J','Q','K','10',10]: return 10
    if x in ['A','11',11]: return 11
    return int(x)

def _hand_features(hand):
    vals = [_card_to_val(c) for c in hand]
    total = sum(11 if v==11 else v for v in vals)
    soft = sum(1 for v in vals if v==11)
    while total > 21 and soft:
        total -= 10; soft -= 1
    is_soft = int(soft > 0)
    is_pair = int(len(vals) == 2 and vals[0] == vals[1])
    return total, is_soft, is_pair

def _derive_hilo_from_remaining(remaining_cards: dict, num_decks: int):
    """
    Given remaining counts {2..10, 11(A)}, compute Hi-Lo running & true counts
    using initial composition of num_decks. 10 represents 10/J/Q/K pooled.
    """
    # initial per-deck: 2..9 = 4 each, 10-bin = 16 (10,J,Q,K), Ace(11) = 4
    init = {**{r: 4*num_decks for r in range(2,10)}, 10: 16*num_decks, 11: 4*num_decks}
    seen = {r: init[r] - int(remaining_cards.get(r, 0)) for r in init}
    # Hi-Lo: 2-6 = +1, 7-9 = 0, T(=10/J/Q/K) = -1, A = -1
    run = sum(seen[r] for r in [2,3,4,5,6]) - (seen[10] + seen[11])
    cards_left = sum(int(remaining_cards.get(r, 0)) for r in init)
    decks_left = max(cards_left / 52.0, 0.01)
    true = run / decks_left
    return run, true, cards_left

def _row_from_obs(*,
                  player_hand, dealer_up, cards_remaining,
                  remaining_card_counts, run_count=None, true_count=None,
                  num_decks=8):
    # Normalize remaining_card_counts (dict or list)
    if not isinstance(remaining_card_counts, dict):
        # list case: [2..10, A(=11)]
        remaining_card_counts = {2+i: int(remaining_card_counts[i]) for i in range(0,9)}
        remaining_card_counts[11] = int(remaining_card_counts[9])

    if run_count is None or true_count is None:
        run_count, true_count, cards_left = _derive_hilo_from_remaining(remaining_card_counts, num_decks)
        cards_remaining = float(cards_left)

    ten = int(remaining_card_counts.get(10, 0))
    ace = int(remaining_card_counts.get(11, 0))
    lows = sum(int(remaining_card_counts.get(k, 0)) for k in range(2,10)) - ten  # 2..9 (exclude 10-bin)

    remaining = max(int(cards_remaining), 1)
    p_ace = ace / remaining
    p_ten = ten / remaining
    p_low = lows / remaining
    hi_lo_density = (lows - (ten + ace)) / remaining

    pt, is_soft, is_pair = _hand_features(player_hand)
    dealer_up_val = _card_to_val(dealer_up)

    row = pd.DataFrame([{
        'player_total': pt,
        'is_soft': is_soft,
        'is_pair': is_pair,
        'dealer_up': dealer_up_val,
        'cards_remaining': float(cards_remaining),
        'run_count': float(run_count if run_count is not None else 0.0),
        'true_count': float(true_count if true_count is not None else 0.0),
        'p_ace': p_ace,
        'p_ten': p_ten,
        'p_low': p_low,
        'hi_lo_density': hi_lo_density,
    }]).astype(float)
    return row


class XGB_BlackJack:
    def __init__(self, model_path: str, num_decks: int = 8):
        self.num_decks = num_decks
        self.booster = Booster()
        self.booster.load_model(model_path)   # loads 'models/xgboost_model.model'
        # prefer names embedded in model; else fallback
        self.features = self.booster.feature_names if self.booster.feature_names else FEATURES

    def _predict_df(self, df: pd.DataFrame):
        # ensure exact feature order / presence
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.features]
        dmat = DMatrix(df, feature_names=list(df.columns))
        probs = self.booster.predict(dmat)    # shape (1, num_class)
        idx = int(np.argmax(probs[0]))
        return IDX2ACTION[idx]

    # simulator-style signature
    def predict(self, player_hand=None, dealer_up_card=None, remaining_cards=None, **kwargs):
        if player_hand is not None and dealer_up_card is not None and remaining_cards is not None:
            row = _row_from_obs(
                player_hand=player_hand,
                dealer_up=dealer_up_card,
                cards_remaining=sum(remaining_cards.values()),
                remaining_card_counts=remaining_cards,
                run_count=None, true_count=None,
                num_decks=self.num_decks
            )
            return self._predict_df(row)
        # evaluator-style kwargs
        if 'player_hand' in kwargs and ('dealer_up' in kwargs or 'dealer_up_card' in kwargs):
            row = _row_from_obs(
                player_hand=kwargs['player_hand'],
                dealer_up=kwargs.get('dealer_up', kwargs.get('dealer_up_card')),
                cards_remaining=kwargs.get('cards_remaining', 0),
                remaining_card_counts=kwargs.get('remaining_card_counts', remaining_cards or {}),
                run_count=kwargs.get('run_count', None),
                true_count=kwargs.get('true_count', None),
                num_decks=self.num_decks
            )
            return self._predict_df(row)
        raise TypeError("XGB_BlackJack.predict called with unsupported arguments.")

    def get_action(self, obs_dict):
        return self.predict(**obs_dict)

    def predict_proba_row(self, **obs_kwargs):
        row = _row_from_obs(num_decks=self.num_decks, **obs_kwargs)
        for col in self.features:
            if col not in row.columns: row[col] = 0.0
        row = row[self.features]
        dmat = DMatrix(row, feature_names=list(row.columns))
        probs = self.booster.predict(dmat)[0]
        return {IDX2ACTION[i]: float(p) for i, p in enumerate(probs)}

    def get_bet_size(self, remaining_cards):
        return 1.0