import pandas as pd
import numpy as np
import ast # Used to safely evaluate string-formatted lists
from tqdm import tqdm
from helper import get_hand_value

tqdm.pandas(desc="Calculating Action EVs")

class BlackjackDataset:
    """
    A class to load, process, and split the blackjack hands dataset.

    This class calculates the Expected Value (EV) for each core action (Hit, Stand,
    Double, Surrender) using Monte Carlo simulation. It then uses this EV to
    determine the optimal move for each hand, which serves as the ground truth
    for model training and evaluation.
    """

    def __init__(self, csv_path, train_split=0.8, val_split=0.1, test_split=0.1, num_simulations=100, num_data_limit=1000000):
        """
        Initializes the BlackjackDataset.

        Args:
            csv_path (str): The file path to the blackjack CSV data.
            train_split (float): The proportion of the dataset to allocate for training.
            val_split (float): The proportion of the dataset to allocate for validation.
            test_split (float): The proportion of the dataset to allocate for testing.
            num_simulations (int): Number of Monte Carlo simulations to run for EV calculation.
                                   Higher is more accurate but slower.
            num_data_limit (int): Maximum number of rows to load from the CSV for processing. Choose a small number for debugging.
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

    def _run_monte_carlo_simulation(self, player_hand, dealer_up_card, remaining_cards):
        """
        Runs simulations for each possible action to determine its Expected Value (EV).
        """
        ev_results = {}
        
        deck = [card for card, count in remaining_cards.items() for _ in range(count)]
        if not deck: return {} # Cannot run simulation with an empty deck

        # --- Action 1: Stand (S) ---
        total_outcome = 0
        for _ in range(self.num_simulations):
            np.random.shuffle(deck)
            sim_deck = list(deck)
            
            dealer_hand = [dealer_up_card, sim_deck.pop()]
            while get_hand_value(dealer_hand) < 17 or (get_hand_value(dealer_hand) == 17 and 11 in dealer_hand):
                if not sim_deck: break
                dealer_hand.append(sim_deck.pop())
            
            player_value = get_hand_value(player_hand)
            dealer_value = get_hand_value(dealer_hand)
            
            if player_value > 21: total_outcome -= 1
            elif dealer_value > 21 or player_value > dealer_value: total_outcome += 1
            elif player_value < dealer_value: total_outcome -= 1
        ev_results['S'] = total_outcome / self.num_simulations

        # --- Action 2: Hit (H) ---
        total_outcome = 0
        for _ in range(self.num_simulations):
            np.random.shuffle(deck)
            sim_deck = list(deck)
            
            player_hand_sim = player_hand + [sim_deck.pop()]

            while get_hand_value(player_hand_sim) < 21:
                p_val = get_hand_value(player_hand_sim)
                is_soft = 11 in player_hand_sim and sum(player_hand_sim) - 10 < 21

                if is_soft:
                    if p_val >= 19: break
                    if p_val == 18 and dealer_up_card <= 8: break
                else:
                    if p_val >= 17: break
                    if 13 <= p_val <= 16 and dealer_up_card <= 6: break
                    if p_val == 12 and 4 <= dealer_up_card <= 6: break
                
                if not sim_deck: break
                player_hand_sim.append(sim_deck.pop())
            
            player_value = get_hand_value(player_hand_sim)

            if player_value > 21:
                total_outcome -= 1
                continue
            
            dealer_hand = [dealer_up_card, sim_deck.pop()]
            while get_hand_value(dealer_hand) < 17 or (get_hand_value(dealer_hand) == 17 and 11 in dealer_hand):
                if not sim_deck: break
                dealer_hand.append(sim_deck.pop())
            dealer_value = get_hand_value(dealer_hand)
            
            if dealer_value > 21 or player_value > dealer_value: total_outcome += 1
            elif player_value < dealer_value: total_outcome -= 1
        ev_results['H'] = total_outcome / self.num_simulations
        
        # --- Action 3: Double Down (D) ---
        if len(player_hand) == 2:
            total_outcome = 0
            for _ in range(self.num_simulations):
                np.random.shuffle(deck)
                sim_deck = list(deck)

                player_hand_after_double = player_hand + [sim_deck.pop()]
                player_value = get_hand_value(player_hand_after_double)

                if player_value > 21:
                    total_outcome -= 2
                    continue

                dealer_hand = [dealer_up_card, sim_deck.pop()]
                while get_hand_value(dealer_hand) < 17 or (get_hand_value(dealer_hand) == 17 and 11 in dealer_hand):
                    if not sim_deck: break
                    dealer_hand.append(sim_deck.pop())
                dealer_value = get_hand_value(dealer_hand)

                if dealer_value > 21 or player_value > dealer_value: total_outcome += 2
                elif player_value < dealer_value: total_outcome -= 2
            ev_results['D'] = total_outcome / self.num_simulations

        # --- Action 4: Surrender (R) ---
        if len(player_hand) == 2:
            ev_results['R'] = -0.5

        return ev_results

    def _load_and_process_data(self, num_data_limit):
        """
        Loads the data, processes it, and adds all annotations.
        """
        print(f"Loading data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path, nrows=num_data_limit)
        
        list_like_cols = ['initial_hand', 'dealer_final', 'player_final', 'actions_taken']
        for col in list_like_cols:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and '[' in x else x)

        df = self._calculate_remaining_cards(df)
        
        print("Calculating Expected Value for each possible action. This may take a while...")
        action_evs = df.progress_apply(
            lambda row: self._run_monte_carlo_simulation(
                row['initial_hand'], row['dealer_up'], row['remaining_card_counts']
            ),
            axis=1
        )
        df['action_evs'] = action_evs
        df['best_action_by_ev'] = df['action_evs'].apply(lambda evs: max(evs, key=evs.get) if evs else None)
        
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
        if split_name == 'train': return self.train_df
        elif split_name == 'val': return self.val_df
        elif split_name == 'test': return self.test_df
        else: raise ValueError("Invalid split name. Choose from 'train', 'val', or 'test'.")


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