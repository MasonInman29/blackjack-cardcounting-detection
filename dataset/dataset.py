import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from helper import get_hand_value
import multiprocessing
import os

class BlackjackDataset:
    """
    A class to load, process, and split the blackjack hands dataset.

    This class calculates the Expected Value (EV) for each core action (Hit, Stand,
    Double, Surrender, Split) using Monte Carlo simulation. It then uses this EV to
    determine the optimal move for each hand, which serves as the ground truth
    for model training and evaluation.
    """

    def __init__(self, csv_path, train_split=0.8, val_split=0.1, test_split=0.1, num_simulations=100, num_data_limit=1000000):
        """
        Initializes the BlackjackDataset.
        (Same as before)
        """
        if not (0.99 < train_split + val_split + test_split < 1.01):
            raise ValueError("Train, validation, and test splits must sum to approximately 1.0.")

        self.csv_path = csv_path
        self.train_split_ratio = train_split
        self.val_split_ratio = val_split
        self.num_simulations = num_simulations
        self.df = self._load_and_process_data(num_data_limit)
        self.train_df, self.val_df, self.test_df = self._split_data()

        print("Dataset loaded and split successfully.")
        print(f"Total hands: {len(self.df)}")
        print(f"Training hands: {len(self.train_df)}")
        print(f"Validation hands: {len(self.val_df)}")
        print(f"Test hands: {len(self.test_df)}")

    @staticmethod
    def _calculate_true_count(sim_deck):
        """Calculates the Hi-Lo true count from a deck list."""
        count = 0
        for card in sim_deck:
            if 2 <= card <= 6:
                count += 1
            elif card == 10 or card == 11: # 10, J, Q, K, A
                count -= 1
        
        num_decks_remaining = len(sim_deck) / 52.0
        if num_decks_remaining < 0.25: 
            num_decks_remaining = 0.25
            
        return count / num_decks_remaining

    @staticmethod
    def _play_player_hand(hand, dealer_up_card, sim_deck):
        """
        Plays out a player hand according to an ADAPTIVE strategy
        based on the true count of the remaining sim_deck.
        """
        player_hand_sim = list(hand)
        
        true_count = BlackjackDataset._calculate_true_count(sim_deck)

        while get_hand_value(player_hand_sim) < 21:
            p_val = get_hand_value(player_hand_sim)
            is_soft = 11 in player_hand_sim and sum(player_hand_sim) - 10 < 21
            
            # --- Apply Adaptive Strategy Rules (Basic Strategy + Deviations) ---
            
            # Example Deviation: 16 vs. 10
            if not is_soft and p_val == 16 and dealer_up_card == 10:
                if true_count > 0: # Deviation: Stand if count is positive
                    break 
            
            # Example Deviation: 12 vs. 4
            if not is_soft and p_val == 12 and dealer_up_card == 4:
                if true_count < 0: # Deviation: Hit if count is negative
                    pass # Falls through to default hit logic
                else: # (count >= 0) Stand
                    break
            
            # --- Standard Basic Strategy Rules (for non-deviation cases) ---
            if is_soft:
                if p_val >= 19: break
                if p_val == 18 and dealer_up_card <= 8: break
            else: # Hard hand
                if p_val >= 17: break
                if 13 <= p_val <= 16 and dealer_up_card <= 6: break
                if p_val == 12 and 4 <= dealer_up_card <= 6: break
            
            if not sim_deck: break
            player_hand_sim.append(sim_deck.pop())

        return player_hand_sim

    @staticmethod
    def _play_dealer_hand(dealer_up_card, sim_deck):
        """
        Plays out the dealer's hand according to standard rules (Hit soft 17).
        Expects sim_deck to be a list starting with the hole card.
        """
        if not sim_deck: return [dealer_up_card]
        
        dealer_hand = [dealer_up_card, sim_deck.pop(0)] 
        
        while get_hand_value(dealer_hand) < 17 or (get_hand_value(dealer_hand) == 17 and 11 in dealer_hand):
            if not sim_deck: break
            dealer_hand.append(sim_deck.pop(0))
        return dealer_hand

    @staticmethod
    def _get_round_outcome(player_value, dealer_value, bet_multiplier=1):
        """
        Compares final hand values and returns the outcome.
        (Same as before)
        """
        if player_value > 21: return -bet_multiplier
        if dealer_value > 21 or player_value > dealer_value: return bet_multiplier
        if player_value < dealer_value: return -bet_multiplier
        return 0 # Push


    @staticmethod
    def _run_monte_carlo_simulation(player_hand, dealer_up_card, remaining_cards, num_simulations):
        """
        Runs simulations for each possible action to determine its Expected Value (EV).
        Includes "Dealer Peeks" logic.
        This is now a static method to be used by multiprocessing.
        """
        ev_results = {}
        
        deck = [card for card, count in remaining_cards.items() for _ in range(count)]
        if not deck: return {}

        player_blackjack = (len(player_hand) == 2 and get_hand_value(player_hand) == 21)
        dealer_needs_to_peek = (dealer_up_card == 11 or dealer_up_card == 10)

        # --- Action 1: Stand (S) ---
        total_outcome = 0
        for _ in range(num_simulations):
            np.random.shuffle(deck)
            sim_deck = list(deck)
            
            if not sim_deck: continue
            hole_card = sim_deck.pop(0)
            
            dealer_has_bj = (dealer_up_card == 11 and hole_card == 10) or \
                            (dealer_up_card == 10 and hole_card == 11)

            if dealer_needs_to_peek and dealer_has_bj:
                total_outcome += 0 if player_blackjack else -1.0
            else:
                player_value = get_hand_value(player_hand)
                dealer_sim_deck = [hole_card] + sim_deck # Re-add hole card for dealer play
                dealer_hand = BlackjackDataset._play_dealer_hand(dealer_up_card, dealer_sim_deck)
                dealer_value = get_hand_value(dealer_hand)
                total_outcome += BlackjackDataset._get_round_outcome(player_value, dealer_value)
                
        ev_results['S'] = total_outcome / num_simulations

        # --- Action 2: Hit (H) ---
        total_outcome = 0
        for _ in range(num_simulations):
            np.random.shuffle(deck)
            sim_deck = list(deck)
            
            if not sim_deck: continue
            hole_card = sim_deck.pop(0)
            
            dealer_has_bj = (dealer_up_card == 11 and hole_card == 10) or \
                            (dealer_up_card == 10 and hole_card == 11)

            if dealer_needs_to_peek and dealer_has_bj:
                total_outcome += 0 if player_blackjack else -1.0
            else:
                if not sim_deck: # Can't hit
                    total_outcome -= 1.0 # Bust (no card to draw)
                    continue
                    
                player_hand_after_hit = player_hand + [sim_deck.pop(0)]
                
                player_hand_final = BlackjackDataset._play_player_hand(player_hand_after_hit, dealer_up_card, sim_deck)
                player_value = get_hand_value(player_hand_final)

                if player_value > 21:
                    total_outcome -= 1.0
                    continue
                
                dealer_sim_deck = [hole_card] + sim_deck 
                dealer_hand = BlackjackDataset._play_dealer_hand(dealer_up_card, dealer_sim_deck)
                dealer_value = get_hand_value(dealer_hand)
                total_outcome += BlackjackDataset._get_round_outcome(player_value, dealer_value)

        ev_results['H'] = total_outcome / num_simulations
        
        # --- Action 3: Double Down (D) ---
        if len(player_hand) == 2:
            total_outcome = 0
            for _ in range(num_simulations):
                np.random.shuffle(deck)
                sim_deck = list(deck)
                
                if not sim_deck: continue
                hole_card = sim_deck.pop(0)

                dealer_has_bj = (dealer_up_card == 11 and hole_card == 10) or \
                                (dealer_up_card == 10 and hole_card == 11)

                if dealer_needs_to_peek and dealer_has_bj:
                    total_outcome += 0 if player_blackjack else -1.0
                else:
                    if not sim_deck: # Can't double
                        total_outcome -= 2.0
                        continue
                        
                    player_hand_after_double = player_hand + [sim_deck.pop(0)]
                    player_value = get_hand_value(player_hand_after_double)

                    if player_value > 21:
                        total_outcome -= 2.0
                        continue

                    dealer_sim_deck = [hole_card] + sim_deck
                    dealer_hand = BlackjackDataset._play_dealer_hand(dealer_up_card, dealer_sim_deck)
                    dealer_value = get_hand_value(dealer_hand)
                    total_outcome += BlackjackDataset._get_round_outcome(player_value, dealer_value, bet_multiplier=2)
                    
            ev_results['D'] = total_outcome / num_simulations

        # --- Action 4: Surrender (R) ---
        if len(player_hand) == 2:
            total_outcome = 0
            for _ in range(num_simulations):
                np.random.shuffle(deck)
                sim_deck = list(deck)
                
                if not sim_deck: continue
                hole_card = sim_deck.pop(0)
                
                dealer_has_bj = (dealer_up_card == 11 and hole_card == 10) or \
                                (dealer_up_card == 10 and hole_card == 11)

                if dealer_needs_to_peek and dealer_has_bj:
                    total_outcome += 0 if player_blackjack else -1.0
                else:
                    total_outcome += -0.5 # Successful surrender
                    
            ev_results['R'] = total_outcome / num_simulations

        # --- Action 5: Split (P) ---
        if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
            total_outcome = 0
            split_card = player_hand[0]
            
            for _ in range(num_simulations):
                np.random.shuffle(deck)
                sim_deck = list(deck)
                
                if not sim_deck: continue
                hole_card = sim_deck.pop(0)

                dealer_has_bj = (dealer_up_card == 11 and hole_card == 10) or \
                                (dealer_up_card == 10 and hole_card == 11)
                
                if dealer_needs_to_peek and dealer_has_bj:
                    total_outcome += 0 if player_blackjack else -2.0 
                else:
                    if len(sim_deck) < 2: continue 
                    
                    # --- Play Hand 1 ---
                    hand_1_start = [split_card, sim_deck.pop(0)]
                    if split_card == 11: 
                        player_hand_1_final = hand_1_start
                    else:
                        player_hand_1_final = BlackjackDataset._play_player_hand(hand_1_start, dealer_up_card, sim_deck)
                    player_value_1 = get_hand_value(player_hand_1_final)
                    
                    # --- Play Hand 2 ---
                    if not sim_deck: continue 
                    hand_2_start = [split_card, sim_deck.pop(0)]
                    if split_card == 11:
                        player_hand_2_final = hand_2_start
                    else:
                        player_hand_2_final = BlackjackDataset._play_player_hand(hand_2_start, dealer_up_card, sim_deck)
                    player_value_2 = get_hand_value(player_hand_2_final)

                    # --- Play Dealer Hand ---
                    dealer_sim_deck = [hole_card] + sim_deck
                    dealer_hand = BlackjackDataset._play_dealer_hand(dealer_up_card, dealer_sim_deck)
                    dealer_value = get_hand_value(dealer_hand)
                    
                    # --- Calculate Total Outcome ---
                    total_outcome += BlackjackDataset._get_round_outcome(player_value_1, dealer_value)
                    total_outcome += BlackjackDataset._get_round_outcome(player_value_2, dealer_value)

            ev_results['P'] = total_outcome / num_simulations

        return ev_results

    @staticmethod
    def _run_sim_wrapper(args_tuple):
        """Helper to unpack arguments for pool.imap_unordered."""
        # Unpack the tuple
        player_hand, dealer_up, remaining_cards, num_sims = args_tuple
        
        # Call the original static method
        return BlackjackDataset._run_monte_carlo_simulation(
            player_hand, dealer_up, remaining_cards, num_sims
        )

    def _load_and_process_data(self, num_data_limit):
        """
        Loads the data, processes it, and adds all annotations.
        Uses multiprocessing to parallelize EV calculations.
        """
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path, nrows=num_data_limit)
        
        list_like_cols = ['initial_hand', 'dealer_final', 'player_final', 'actions_taken']
        for col in list_like_cols:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and '[' in x else x)

        df = self._calculate_remaining_cards(df)
        
        print("Calculating Expected Value for each possible action...")

        tasks = [
            (row['initial_hand'], row['dealer_up'], row['remaining_card_counts'], self.num_simulations)
            for _, row in df.iterrows()
        ]
        
        num_workers = max(1, os.cpu_count() - 1) 
        print(f"Starting EV calculation on {num_workers} processes...")
        
        action_evs_list = []
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            action_evs_list = list(tqdm(
                pool.imap(
                    BlackjackDataset._run_sim_wrapper,
                    tasks,
                    chunksize=1
                ),
                total=len(tasks),
                desc="Calculating Action EVs"
            ))

        df['action_evs'] = action_evs_list
        df['best_action_by_ev'] = df['action_evs'].apply(lambda evs: max(evs, key=evs.get) if evs else None)
        
        # df['remaining_card_counts'] = df['remaining_card_counts'].apply(
        #     lambda counts: {str(k): v for k, v in counts.items()}
        # )
        # df.to_parquet("blackjack_simulations.parquet", engine="pyarrow")
        
        return df

    def _calculate_remaining_cards(self, df):
        print("Calculating remaining card counts for each shoe...")
        initial_shoe_counts = {**{i: 32 for i in range(2, 10)}, 10: 128, 11: 32}
        all_processed_shoes = []
        for shoe_id, shoe_df in df.groupby('shoe_id'):
            sorted_shoe_df = shoe_df.sort_values('cards_remaining', ascending=False).copy()
            remaining_counts_per_round = []
            running_counts = initial_shoe_counts.copy()
            for index, row in sorted_shoe_df.iterrows():
                remaining_counts_per_round.append(running_counts.copy())
                # Flatten nested lists in player_final before iterating
                flat_player_final = [card for hand in row['player_final'] for card in hand]
                cards_played = flat_player_final + row['dealer_final']
                for card in cards_played:
                    if card in running_counts and running_counts[card] > 0:
                        running_counts[card] -= 1
            sorted_shoe_df['remaining_card_counts'] = remaining_counts_per_round
            all_processed_shoes.append(sorted_shoe_df)
        return pd.concat(all_processed_shoes).sort_index()

    def _split_data(self):
        shoe_ids = self.df['shoe_id'].unique()
        np.random.seed(42)
        np.random.shuffle(shoe_ids)
        train_end_idx = int(len(shoe_ids) * self.train_split_ratio)
        val_end_idx = train_end_idx + int(len(shoe_ids) * self.val_split_ratio)
        train_shoe_ids, val_shoe_ids, test_shoe_ids = shoe_ids[:train_end_idx], shoe_ids[train_end_idx:val_end_idx], shoe_ids[val_end_idx:]
        train_df = self.df[self.df['shoe_id'].isin(train_shoe_ids)].copy()
        val_df = self.df[self.df['shoe_id'].isin(val_shoe_ids)].copy()
        test_df = self.df[self.df['shoe_id'].isin(test_shoe_ids)].copy()
        return train_df, val_df, test_df

    def get_split(self, split_name='train'):
        if split_name == 'train': 
            return self.train_df
        elif split_name == 'val': 
            return self.val_df
        elif split_name == 'test': 
            return self.test_df
        else: 
            raise ValueError("Invalid split name. Choose from 'train', 'val', or 'test'.")


class CSVDataset:
    def __init__(self, nrows=3000000):
        print("Loading Dataset")
        schema = {
            "shoe_id": int,
            "cards_remaining": int,
            "dealer_up": int,
            "run_count": int,
            "true_count": int,
            "dealer_final_value": lambda x: 21 if x == "BJ" else int(x),
            "win": float,
            "initial_hand": lambda x: [int(i) for i in ast.literal_eval(x)],
            "dealer_final": lambda x: [int(i) for i in ast.literal_eval(x)],
            "player_final": lambda x: [[int(j) for j in i] for i in ast.literal_eval(x)],
            "player_final_value": lambda x: [
                21 if v == "'BJ'" or v == "BJ" else int(v)
                for v in ast.literal_eval(x)
            ],
            "actions_taken": lambda x: [[str(j) for j in i] for i in ast.literal_eval(x)],
            "action_evs": lambda x: {str(k): float(v) for k, v in ast.literal_eval(x).items()},
            "best_action_by_ev": str,
            "remaining_card_counts": lambda x: {int(k): int(v) for k, v in ast.literal_eval(x).items()},
        }

        self.df = pd.read_csv("blackjack_labeled_simulations.csv", dtype=str, nrows=nrows)

        for col, parser in schema.items():
            self.df[col] = self.df[col].apply(parser)

        print("Dataset loaded")

    def get_split(self, split_name="train"):
        if split_name == "train":
            return train_test_split(self.df, train_size=.8, random_state=42)[0]
        elif split_name == "test":
            return train_test_split(self.df, train_size=.8, random_state=42)[1]
        else:
            raise ValueError("Invalid split name. Must be 'train' or 'test'.")
        
        
class ParquetDataset:
    def __init__(self, nrows=None):
        print("Loading Dataset from Parquet...")
        
        df = pd.read_parquet("blackjack_simulations.parquet", engine="pyarrow")
        # print(df.columns)
        if nrows is not None:
            self.df = df.iloc[:nrows]
        else:
            self.df = df
            
        self.df['remaining_card_counts'] = self.df['remaining_card_counts'].apply(
            lambda string_key_dict: {int(k): v for k, v in string_key_dict.items()}
        )

        print("Dataset loaded.")
        
    def get_split(self, split_name="train"):
        if split_name == "train":
            return train_test_split(self.df, train_size=.8, random_state=42)[0]
        elif split_name == "test":
            return train_test_split(self.df, train_size=.8, random_state=42)[1]
        else:
            raise ValueError("Invalid split name. Must be 'train' or 'test'.")


if __name__== "__main__":
    # Example usage
    dataset = BlackjackDataset(csv_path='blackjack_simulator.csv', num_simulations=100, num_data_limit=1000) 
    # download 'blackjack_simulator.csv' from https://www.kaggle.com/datasets/dennisho/blackjack-hands/data
    # use a small num_data_limit for debugging.
    
    
    train_data = dataset.get_split('train')
    val_data = dataset.get_split('val')
    test_data = dataset.get_split('test')
    print(f"Train data sample:\n{train_data.head()}")
    print(f"Validation data sample:\n{val_data.head()}")
    print(f"Test data sample:\n{test_data.head()}")