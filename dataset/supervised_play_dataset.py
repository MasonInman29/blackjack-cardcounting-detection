import torch
from torch.utils.data import Dataset
import numpy as np
from helper import get_hand_value

ACTION_TO_IDX = {'H': 0, 'S': 1, 'D': 2, 'R': 3, 'P': 4}

CARD_KEYS = [2,3,4,5,6,7,8,9,10,11]  # must match RLModelDQN


# ---- One-hot encoding for total (4..21 -> 18 dims) ----
def one_hot_total(total):
    total = int(max(4, min(21, total)))
    vec = np.zeros(18, dtype=np.float32)
    vec[total - 4] = 1.0
    return vec


# ---- One-hot encoding for dealer up-card (2..11 → 10 dims) ----
def one_hot_dealer(dealer):
    dealer = int(max(2, min(11, dealer)))
    idx = dealer - 2
    vec = np.zeros(10, dtype=np.float32)
    vec[idx] = 1.0
    return vec


# ---- Count ratios ----
def get_ratio_vector(remaining):
    total = sum(remaining.values())
    ratios = [remaining.get(k, 0) / total for k in CARD_KEYS]
    return ratios, total


# ---- True count (same formula as RL model) ----
def compute_true_count(remaining, num_decks):
    current_low = sum(remaining.get(c, 0) for c in [2,3,4,5,6])
    current_high = sum(remaining.get(c, 0) for c in [10,11])

    initial_low = 20 * num_decks
    initial_high = 20 * num_decks

    low_played = initial_low - current_low
    high_played = initial_high - current_high

    running = low_played - high_played
    decks_remaining = sum(remaining.values()) / 52.0

    if decks_remaining < 0.25:
        decks_remaining = 0.25

    tc = running / decks_remaining
    return max(-1.0, min(1.0, tc / 10.0))


# ---- Preprocess one row ----
def preprocess_row(row, num_decks=8):
    hand = row["initial_hand"]
    dealer = row["dealer_up"]
    remaining = row["remaining_card_counts"]

    # Convert J/Q/K → 10
    hand_t = [c if c <= 20 else 10 for c in hand]
    total = get_hand_value(hand_t)

    is_soft = 11 in hand_t and sum(hand_t) - 10 < 21
    can_split = len(hand_t) == 2 and hand_t[0] == hand_t[1]
    can_double = len(hand_t) == 2

    # --- base flags (3 dims) ---
    base_flags = [
        float(is_soft),
        float(can_split),
        float(can_double)
    ]

    # --- 18: one-hot total ---
    total_onehot = one_hot_total(total)

    # --- 10: one-hot dealer ---
    dealer_onehot = one_hot_dealer(dealer)

    # --- 10 ratios + 1 penetration ---
    ratios, total_cards = get_ratio_vector(remaining)
    penetration = total_cards / (52 * num_decks)

    # --- 1 true count ---
    true_count = compute_true_count(remaining, num_decks)

    # ------- FINAL INPUT VECTOR (43 dims) -------
    state_vector = np.concatenate([
        total_onehot,       # 18
        dealer_onehot,      # 10
        base_flags,         # 3
        ratios,             # 10
        [penetration],      # 1
        [true_count],       # 1
    ]).astype(np.float32)

    # --- soft labels from EVs ---
    evs = row["action_evs"]
    actions = ['H','S','D','R','P']

    # replace None with large negative
    ev_vec = np.array([evs.get(a, -9999) if evs.get(a) is not None else -9999
                       for a in actions], dtype=np.float32)

    # convert EVs → soft probabilities
    T = 0.1
    probs = np.exp(ev_vec / T)
    probs /= probs.sum()

    return state_vector, probs


class PlaySupervisedDataset(Dataset):
    def __init__(self, df, num_decks=8):
        self.df = df
        self.num_decks = num_decks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y = preprocess_row(row, self.num_decks)
        return torch.tensor(x), torch.tensor(y)
