import math
import pickle
import os

from helper import get_hand_value
import numpy as np


class RLModel:
    """
    A model that learns Blackjack strategy using Q-learning.
    It learns both a playing policy and a betting policy.
    """
    
    def __init__(self, num_decks=8, bet_spread=20, 
                 learning_rate=0.1, 
                 discount_factor=0.95, 
                 epsilon_start=1.0, 
                 epsilon_end=0.01, 
                 epsilon_decay=0.99999885):
        
        self.num_decks = num_decks
        self.bet_spread = float(bet_spread)
        
        # --- Hi-Lo Count State (copied from HILO for state representation) ---
        self.initial_low_cards = 20 * self.num_decks # (2, 3, 4, 5, 6)
        self.initial_high_cards = 20 * self.num_decks # (10, J, Q, K, A)

        # --- RL Parameters ---
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # --- Q-tables ---
        # 1. Q-table for playing decisions:
        # key = (player_total, dealer_up_card, is_soft, can_split, can_double, tc_bucket)
        # val = {action: q_value}
        self.q_table = {}

        # 2. Q-table for betting decisions:
        # key = tc_bucket
        # val = {bet_size: q_value}
        self.bet_q_table = {}
        
        # --- Bet Action Space ---
        self.bet_actions = [1.0]
        bet = 2.0
        while bet < self.bet_spread:
            self.bet_actions.append(bet)
            bet *= 2.0
        if self.bet_spread not in self.bet_actions:
             self.bet_actions.append(self.bet_spread)

    # --- State Representation Methods ---

    def _get_true_count(self, remaining_cards):
        """
        Calculates the running count and true count.
        """
        current_low = sum(remaining_cards.get(c, 0) for c in [2, 3, 4, 5, 6])
        current_high = sum(remaining_cards.get(c, 0) for c in [10, 11])
        
        low_cards_played = self.initial_low_cards - current_low
        high_cards_played = self.initial_high_cards - current_high
        
        running_count = low_cards_played - high_cards_played
        
        decks_remaining = sum(remaining_cards.values()) / 52.0
        
        return running_count / decks_remaining if decks_remaining > 0.25 else 0

    def _get_bet_state(self, remaining_cards):
        """Returns a hashable state for the betting decision."""
        tc = self._get_true_count(remaining_cards)
        # Discretize the true count into integer buckets
        return int(round(tc))
    
    def transform_hand(self, hand):
        """Converts J, Q, K (21, 22, 23) to 10 for value calculation."""
        return [card if card <= 20 else 10 for card in hand]

    def _get_play_state(self, player_hand, dealer_up_card, remaining_cards):
        """Returns a hashable state for the playing decision."""
        
        can_split = len(player_hand) == 2 and player_hand[0] == player_hand[1]
        
        player_hand = self.transform_hand(player_hand)
        
        player_total = get_hand_value(player_hand)
        # Use logic from HILO's basic strategy
        is_soft = 11 in player_hand and sum(player_hand) - 10 < 21
        
        can_double_surrender = len(player_hand) == 2
        
        tc_bucket = self._get_bet_state(remaining_cards) # Re-use the bet state
        
        return (player_total, dealer_up_card, is_soft, can_split, can_double_surrender, tc_bucket)

    def _get_valid_actions(self, state_tuple):
        """Gets all valid actions from a given state tuple."""
        (total, dealer, is_soft, can_split, can_double_surrender, tc) = state_tuple
        
        actions = ['H', 'S'] # Hit and Stand are always possible
        
        # Surrender, Double, and Split are only allowed on the first 2 cards
        if can_double_surrender:
            actions.append('D')
            actions.append('R')
        
        if can_split:
            actions.append('P')
            
        return actions

    def _get_q_values(self, table, state, valid_actions):
        """Helper to get Q-values from a table, initializing state if new."""
        if state not in table:
            # Initialize Q-values for all valid actions in this new state
            table[state] = {action: 0.0 for action in valid_actions}
        
        # In case the state was seen before, but not all actions were valid
        # (e.g., seen (10, 5, F, F, T, 0) from 3+ cards, then see it with 2 cards)
        for action in valid_actions:
            if action not in table[state]:
                table[state][action] = 0.0
                    
        return table[state]

    # --- Required API Methods (Policy) ---

    def predict(self, player_hand, dealer_up_card, remaining_cards):
        """
        Predicts the best action using the learned Q-table (policy).
        Uses epsilon-greedy for exploration during training.
        """
        state = self._get_play_state(player_hand, dealer_up_card, remaining_cards)
        player_hand = self.transform_hand(player_hand)
        valid_actions = self._get_valid_actions(state)
        
        # Get Q-values, initializing if state is new
        q_values = self._get_q_values(self.q_table, state, valid_actions)
        
        # Epsilon-greedy: Explore or Exploit
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploitation: Choose best-known action
        # Filter q_values to *only* currently valid actions
        valid_q_values = {a: q_values[a] for a in valid_actions}
        return max(valid_q_values, key=valid_q_values.get)

    def get_bet_size(self, remaining_cards):
        """
        Selects a bet size using the learned betting Q-table.
        Uses epsilon-greedy for exploration during training.
        """
        state = self._get_bet_state(remaining_cards)
        
        # Get Q-values, initializing if state is new
        q_values = self._get_q_values(self.bet_q_table, state, self.bet_actions)
        
        # Epsilon-greedy: Explore or Exploit
        if np.random.random() < self.epsilon:
            return np.random.choice(self.bet_actions)
        
        # Exploitation: Choose best-known bet
        return max(q_values, key=q_values.get)

    # --- Learning Methods (Must be called by your simulator) ---

    def update_play(self, state_tuple, action, reward, next_state_tuple, is_done):
        """
        Updates the Q-table for a (state, action, reward, next_state) transition.
        Your simulator must call this after each action (H, S, D, R, P).
        """
        # Ensure Q-values are initialized for the current state
        valid_actions = self._get_valid_actions(state_tuple)
        self._get_q_values(self.q_table, state_tuple, valid_actions)
        
        # Get the max Q-value for the *next* state
        if is_done:
            next_max_q = 0.0 # No future reward if the hand is over
        else:
            next_valid_actions = self._get_valid_actions(next_state_tuple)
            next_q_values = self._get_q_values(self.q_table, next_state_tuple, next_valid_actions)
            next_max_q = max(next_q_values.values())
        
        # Q-Learning update rule
        old_q = self.q_table[state_tuple][action]
        
        # new_q = old_q + lr * (reward + gamma * next_max_q - old_q)
        new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
        
        self.q_table[state_tuple][action] = new_q

    def update_bet(self, bet_state_tuple, bet_action, total_hand_reward):
        """
        Updates the betting Q-table with the final reward for the hand.
        Your simulator must call this ONCE at the end of the entire hand.
        
        'total_hand_reward' should be the total profit/loss for that hand, 
        e.g., +10, -10, +15.
        """
        # Ensure Q-values are initialized
        self._get_q_values(self.bet_q_table, bet_state_tuple, self.bet_actions)
        
        old_q = self.bet_q_table[bet_state_tuple][bet_action]
        
        # This is a Monte Carlo update: Q(s,a) = Q(s,a) + lr * (Reward - Q(s,a))
        # The Q-value will learn the *expected final profit* for making that bet.
        new_q = old_q + self.lr * (total_hand_reward - old_q)
        
        self.bet_q_table[bet_state_tuple][bet_action] = new_q

    def decay_epsilon(self):
        """Call this at the end of each episode (e.g., each shoe)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, file_path="rl_model.pkl"):
        """Saves the learned Q-tables to a file."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump((self.q_table, self.bet_q_table), f)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, file_path="rl_model.pkl"):
        """Loads learned Q-tables from a file and sets epsilon to 0 for exploitation."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    self.q_table, self.bet_q_table = pickle.load(f)
                # After loading a trained model, set epsilon low for exploitation
                self.epsilon = 0.0
                print(f"Model loaded from {file_path}. Epsilon set to {self.epsilon}.")
            except Exception as e:
                print(f"Error loading model: {e}. Starting fresh.")
        else:
            print(f"No model file found at {file_path}. Starting with empty Q-tables.")