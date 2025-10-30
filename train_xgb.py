# pip install xgboost
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

ACTION_MAP = {'H':0,'S':1,'D':2,'P':3,'R':4}

def card_to_val(x):
    if x in ['J','Q','K','10',10]: return 10
    if x in ['A', '11', 11]: return 11
    return int(x)

def hand_features(hand):
    # hand like ['A','6'] or [8,8]; return total, is_soft, is_pair
    vals = [card_to_val(c) for c in hand]
    # count aces as 11, reduce by 10 while busting
    total = sum(11 if v==11 else v for v in vals)
    soft_aces = sum(1 for v in vals if v==11)
    while total > 21 and soft_aces:
        total -= 10
        soft_aces -= 1
    is_soft = (soft_aces > 0)
    is_pair = (len(vals)==2 and vals[0]==vals[1])
    return total, int(is_soft), int(is_pair)

def build_frame(df):
    # Expand deck-composition
    rcc = df['remaining_card_counts']  # dict or list in order [2..10,11(A)]
    if isinstance(rcc.iloc[0], dict):
        get = lambda d,k: d.get(k,0)
        ten = rcc.apply(lambda d: get(d,10))
        ace = rcc.apply(lambda d: get(d,11))
        lows = [rcc.apply(lambda d: get(d,k)) for k in range(2,10)]
    else:
        # list case: index 0..8 => 2..10, index 9 => Ace(11)
        ten  = rcc.apply(lambda L: L[8])
        ace  = rcc.apply(lambda L: L[9])
        lows = [rcc.apply(lambda L, i=i: L[i]) for i in range(0,7)] + \
               [rcc.apply(lambda L: L[7])]  # 2..9 split

    remaining = df['cards_remaining'].clip(lower=1).astype(float)
    sum_low = sum(lows)
    p_ace = (ace / remaining)
    p_ten = (ten / remaining)
    p_low = (sum_low / remaining)
    # quick density proxy for Hi-Lo flavor
    hi_lo_density = ((sum_low) - (ten + ace)) / remaining

    # Hand features
    # initial_hand looks like ['A','6'] or similar
    h = df['initial_hand'].apply(hand_features)
    player_total = h.apply(lambda t: t[0])
    is_soft = h.apply(lambda t: t[1])
    is_pair = h.apply(lambda t: t[2])
    dealer_up_val = df['dealer_up'].apply(card_to_val)

    X = pd.DataFrame({
        'player_total': player_total,
        'is_soft': is_soft,
        'is_pair': is_pair,
        'dealer_up': dealer_up_val,
        'cards_remaining': df['cards_remaining'],
        'run_count': df['run_count'],
        'true_count': df['true_count'],
        'p_ace': p_ace,
        'p_ten': p_ten,
        'p_low': p_low,
        'hi_lo_density': hi_lo_density,
    }).astype(float)

    y = df['best_action_by_ev'].map(ACTION_MAP)
    return X, y

# ---- load ParquetDataset() ----
from dataset.dataset import ParquetDataset
dset = ParquetDataset()
train_df = dset.get_split('train')
test_df  = dset.get_split('test')
df = pd.concat([train_df, test_df], ignore_index=True)

# Drop any rows with missing labels
df = df[df['best_action_by_ev'].isin(ACTION_MAP)]

X, y = build_frame(df)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # 80/20 split

model = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=5,
    tree_method='hist',
    device='cuda',          # use 'gpu_hist' if CUDA is available, 'hist' otherwise
    eval_metric='mlogloss',
    early_stopping_rounds=50
)

print("fitting xgb model...")
model.fit(
    Xtrain, ytrain,
    eval_set=[(Xtest, ytest)],
    verbose=False
)

pred = model.predict(Xtest)
acc = accuracy_score(ytest, pred)
print(f"Policy accuracy (EV-optimal match): {acc:.3f}")

os.makedirs("models", exist_ok=True)
model_path = "models/xgb_policy.json"
model.save_model('models/xgboost_model.model')

# To use: .... do this dummy
# # Load the saved model
# loaded_model = xgb.XGBClassifier()
# loaded_model.load_model('xgboost_model.model')
