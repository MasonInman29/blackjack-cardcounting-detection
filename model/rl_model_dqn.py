# model/rl_model_dqn.py

import math
import pickle
import os
import random
from collections import deque, namedtuple

from helper import get_hand_value
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# --- Neural Network Definitions ---

class PlayNetwork(nn.Module):
    def __init__(self, input_size=43, output_size=5, hidden_size=320):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Block 1
        x = torch.relu(self.norm1(self.fc1(x)))

        # Block 2 (residual)
        r1 = x
        x = torch.relu(self.norm2(self.fc2(x)))
        x = x + r1

        # Block 3 (residual)
        r2 = x
        x = torch.relu(self.norm3(self.fc3(x)))
        x = x + r2

        return self.out(x)



class BetNetwork(nn.Module):
    """
    Input: 10 features [normalized card counts for 2, 3, 4, 5, 6, 7, 8, 9, 10, A]
    Output: N Q-values [Q(bet_1), Q(bet_2), ..., Q(bet_N)]
    """
    def __init__(self, input_size=12, output_size=20, hidden_size=128):
        super(BetNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output_layer(x)


# --- Replay Buffer ---

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- Main RL Model Class ---

class RLModelDQN:
    """
    A model that learns Blackjack strategy using Deep Q-Learning (DQN).
    It learns both a playing policy and a betting policy.

    NEW / IMPROVED:
    - Supports 3-stage training:
        Stage 1: Supervised PlayNetwork (no RL updates).
        Stage 2: DQN fine-tuning of PlayNetwork (bet frozen, bet=1).
        Stage 3: DQN training of BetNetwork (play frozen).
      Call set_training_stage(stage) to enable staged behavior.
      If training_stage is left as None, behaves like legacy "joint RL training".
    - Double DQN for the play network for more stable targets.
    - Gradient clipping to prevent destructive updates.
    - Reward clipping for bet training to reduce variance.
    """

    def __init__(self,
                 num_decks=8,
                 bet_spread=20,
                 learning_rate=1e-4,
                 bet_learning_rate=1e-7,
                 discount_factor=0.95,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.999995,
                 memory_size=50000,
                 bet_memory_size=10000,
                 batch_size=1024,
                 target_update_freq=100,
                 bet_reward_clip=True,
                 freeze_play=False,
                 freeze_bet=False,
                 force_bet_one_warmup=True,
                 warmup_steps=50000,
                 max_grad_norm=1.0,
                 double_dqn=True):
        """
        Parameters largely match the original implementation, with some added safety knobs:
        - max_grad_norm: float or None; if set, gradients are clipped to this norm.
        - double_dqn: use Double DQN targets for play network.
        """
        self.num_decks = num_decks
        self.bet_spread = float(bet_spread)
        
        self.initial_low_cards = 20 * self.num_decks # (2, 3, 4, 5, 6)
        self.initial_high_cards = 20 * self.num_decks # (10, J, Q, K, A)

        # --- Device Detection (GPU/CPU) ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        # --- Initial Card Counts (for normalization) ---
        self.card_keys = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.initial_card_counts = {key: 4 * self.num_decks for key in self.card_keys}
        # 10s include 10,J,Q,K
        self.initial_card_counts[10] = 16 * self.num_decks

        # --- RL Parameters ---
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        # --- Safety / training control flags ---
        self.bet_reward_clip = bet_reward_clip
        self.freeze_play_flag = freeze_play
        self.freeze_bet_flag = freeze_bet
        self.force_bet_one_warmup = force_bet_one_warmup
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.double_dqn = double_dqn

        # Training stage:
        #   None = legacy joint RL training (both play & bet train as before)
        #   1 = supervised play (no RL updates)
        #   2 = DQN play fine-tune (bet frozen, bet=1)
        #   3 = DQN bet training (play frozen)
        self.training_stage = None

        # --- Action Mappings ---
        self.play_actions_list = ['H', 'S', 'D', 'R', 'P']
        self.action_to_idx = {action: i for i, action in enumerate(self.play_actions_list)}
        self.idx_to_action = {i: action for i, action in enumerate(self.play_actions_list)}

        self.bet_actions = [float(x) for x in range(1, int(round(self.bet_spread)) + 1)]
        self.bet_action_to_idx = {action: i for i, action in enumerate(self.bet_actions)}
        self.bet_idx_to_action = {i: action for i, action in enumerate(self.bet_actions)}

        # --- Define Network Input/Output Sizes ---
        self.play_input_size = 43
        self.play_output_size = len(self.play_actions_list)

        self.bet_input_size = 12
        self.bet_output_size = len(self.bet_actions)

        # --- Playing Model (Moved to device) ---
        self.play_policy_net = PlayNetwork(self.play_input_size, self.play_output_size).to(self.device)
        self.play_target_net = PlayNetwork(self.play_input_size, self.play_output_size).to(self.device)
        self.play_target_net.load_state_dict(self.play_policy_net.state_dict())
        self.play_target_net.eval()
        self.play_optimizer = optim.Adam(self.play_policy_net.parameters(), lr=learning_rate)
        self.play_memory = ReplayBuffer(memory_size)

        # --- Betting Model (Moved to device) ---
        self.bet_policy_net = BetNetwork(self.bet_input_size, self.bet_output_size).to(self.device)
        self.bet_target_net = BetNetwork(self.bet_input_size, self.bet_output_size).to(self.device)
        self.bet_target_net.load_state_dict(self.bet_policy_net.state_dict())
        self.bet_target_net.eval()
        self.bet_optimizer = optim.Adam(self.bet_policy_net.parameters(), lr=bet_learning_rate)
        self.bet_memory = ReplayBuffer(bet_memory_size)
        
        self.original_rl_model = None

    # --- Public controls for decoupled training ---

    def freeze_play(self):
        self.freeze_play_flag = True

    def unfreeze_play(self):
        self.freeze_play_flag = False

    def freeze_bet(self):
        self.freeze_bet_flag = True

    def unfreeze_bet(self):
        self.freeze_bet_flag = False

    def set_force_bet_one_warmup(self, enabled: bool, warmup_steps: int = None):
        self.force_bet_one_warmup = enabled
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps

    def set_training_stage(self, stage: int):
        """
        stage 1: supervised play (no RL updates, bet frozen, bet=1)
        stage 2: RL fine-tuning of play (bet frozen, bet=1)
        stage 3: RL training of bet (play frozen)
        If you never call this, training_stage stays None and model behaves like legacy joint RL.
        """
        if stage not in (1, 2, 3):
            raise ValueError("training_stage must be 1, 2, 3, or None")

        self.training_stage = stage

        if stage == 1:
            # Supervised only; RL buffers should not be used for play or bet
            self.freeze_play_flag = False      # we WANT to train play, but via external supervised code
            self.freeze_bet_flag = True        # bet always off during stage 1
            self.force_bet_one_warmup = True   # always bet 1
            self.warmup_steps = 10**9          # effectively never un-warmup
            print("[RLModelDQN] Training stage set to 1 (supervised play). Bet network frozen.")

        elif stage == 2:
            # Fine-tune play with RL; bet still frozen, bet=1
            self.freeze_play_flag = False
            self.freeze_bet_flag = True
            self.force_bet_one_warmup = True   # always bet 1 in stage 2
            print("[RLModelDQN] Training stage set to 2 (DQN play fine-tune, bet frozen).")

        elif stage == 3:
            # Train bet; freeze play
            self.freeze_play_flag = True
            self.freeze_bet_flag = False
            self.force_bet_one_warmup = False  # allow full bet policy
            print("[RLModelDQN] Training stage set to 3 (DQN bet training, play frozen).")

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
        if decks_remaining < 0.25:
            decks_remaining = 0.25
        
        return running_count / decks_remaining

    def _get_card_count_vector(self, remaining_cards):
        vector = []
        total = sum(remaining_cards.values())

        for key in self.card_keys:
            count = remaining_cards.get(key, 0)
            ratio = count / total
            vector.append(ratio)

        vector.append(total / (52 * self.num_decks))
        
        true_count = self._get_true_count(remaining_cards)
        # Normalize true count to [-1, 1] based on practical min/max
        normalized_true_count = max(-1.0, min(1.0, true_count / 10.0))
        vector.append(normalized_true_count)
        
        return vector


    def _get_bet_state(self, remaining_cards):
        # keep same representation as before; simulator expects dict
        return remaining_cards

    def transform_hand(self, hand):
        """Converts J, Q, K (21, 22, 23) to 10 for value calculation."""
        return [card if card <= 20 else 10 for card in hand]

    def _get_play_state(self, player_hand, dealer_up_card, remaining_cards):
        """Returns the state tuple. Note: player_hand is raw (can contain 21, 22, 23)."""
        can_split = len(player_hand) == 2 and player_hand[0] == player_hand[1]

        player_hand_transformed = self.transform_hand(player_hand)
        player_total = get_hand_value(player_hand_transformed)
        is_soft = 11 in player_hand_transformed and sum(player_hand_transformed) - 10 < 21

        can_double_surrender = len(player_hand) == 2

        return (player_total, dealer_up_card, is_soft, can_split, can_double_surrender, remaining_cards)

    def _get_valid_actions(self, state_tuple):
        (total, dealer, is_soft, can_split, can_double_surrender, _) = state_tuple

        actions = ['H', 'S']

        if can_double_surrender:
            actions.append('D')
            actions.append('R')

        if can_split:
            actions.append('P')

        return actions

    def _one_hot_total(self, total):
        total = int(max(4, min(21, total)))
        vec = np.zeros(18, dtype=np.float32)
        vec[total - 4] = 1.0
        return vec
    
    def _one_hot_dealer(self, dealer):
        """
        Dealer up-card ∈ {2..11}. One-hot → 10 dims.
        Index: card-2 (2→0 ... 11→9)
        """
        idx = max(0, min(9, dealer - 2))   # safety clamp
        vec = np.zeros(10, dtype=np.float32)
        vec[idx] = 1.0
        return vec

    def _preprocess_play_state(self, state_tuple):
        """
        New play state vector:
        - 18 dims: one-hot player total (4..21)
        - 10 dims: one-hot dealer up-card (2..11)
        - 3 dims: is_soft, can_split, can_double
        - 12 dims: shoe composition (10 ratios + penetration + true count)
        = 43 total
        """
        (total, dealer, is_soft, can_split, can_double, remaining_cards) = state_tuple

        # 1. One-hot player total (18)
        total_onehot = self._one_hot_total(total)

        # 2. One-hot dealer up-card (10)
        dealer_onehot = self._one_hot_dealer(dealer)

        # 3. Action flags (3)
        base_flags = np.array([
            float(is_soft),
            float(can_split),
            float(can_double),
        ], dtype=np.float32)

        # 4. Shoe composition (12 = 10 ratios + penetration + normalized TC)
        card_count_features = np.array(
            self._get_card_count_vector(remaining_cards),
            dtype=np.float32
        )

        # Final concatenation
        state_vector = np.concatenate([
            total_onehot,        # 18
            dealer_onehot,       # 10
            base_flags,          # 3
            card_count_features  # 12
        ]).astype(np.float32)

        return torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)


    def _preprocess_bet_state(self, state_tuple):
        """Convert remaining_cards dict into 12-dim feature vector for BetNetwork."""
        remaining_cards = state_tuple

        bet_features = np.array(self._get_card_count_vector(remaining_cards), dtype=np.float32)

        return torch.tensor(bet_features, dtype=torch.float32, device=self.device).unsqueeze(0)


    # --- Policy Methods ---

    def predict(self, player_hand, dealer_up_card, remaining_cards):
        """
        Returns a play action (H, S, D, R, P) using epsilon-greedy policy.
        Respects per-hand valid action constraints.
        """
        state_tuple = self._get_play_state(player_hand, dealer_up_card, remaining_cards)
        valid_actions = self._get_valid_actions(state_tuple)

        # In stage 1, you typically won't be calling this during supervised training;
        # but if you do, it still behaves like an epsilon-greedy policy.
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        state_tensor = self._preprocess_play_state(state_tuple)
        with torch.no_grad():
            q_values = self.play_policy_net(state_tensor)[0]

            # Vectorized masking
            mask = torch.full(q_values.shape, -float('inf'), device=self.device)
            valid_indices = [self.action_to_idx[action] for action in valid_actions]
            mask[valid_indices] = 0.0
            masked_q_values = q_values + mask

            best_action_idx = torch.argmax(masked_q_values).item()
            return self.idx_to_action[best_action_idx]

    def get_bet_size(self, remaining_cards):
        """
        Returns the selected bet size.
        Stage-aware behavior:
         - Stage 1: always 1.0
         - Stage 2: always 1.0
         - Stage 3: learned bet sizing (unless freeze_bet_flag is True)
        Legacy mode (training_stage=None) behaves like original bet policy with warmup gating.
        """
        if self.original_rl_model is not None:
            return self.original_rl_model.get_bet_size(remaining_cards)
        
        # Stage 1 or 2: always return minimum bet
        if self.training_stage in (1, 2):
            return 1.0

        # Legacy or Stage 3 behavior below
        if self.freeze_bet_flag:
            return 1.0
        if self.force_bet_one_warmup and self.steps_done < self.warmup_steps:
            return 1.0

        if np.random.random() < self.epsilon:
            return np.random.choice(self.bet_actions)

        state_tensor = self._preprocess_bet_state(self._get_bet_state(remaining_cards))
        with torch.no_grad():
            q_values = self.bet_policy_net(state_tensor)[0]
            best_bet_idx = torch.argmax(q_values).item()
            return self.bet_idx_to_action[best_bet_idx]

    # --- Experience Storage (Replay) ---

    def remember_play(self, state_tuple, action, reward, next_state_tuple, is_done):
        """
        Stores a play transition in the replay buffer (all tensors on device).
        In Stage 1, this is disabled (supervised only).
        """
        # Stage 1: supervised only, skip RL memory
        if self.training_stage == 1:
            return

        if self.freeze_play_flag:
            return

        state_tensor = self._preprocess_play_state(state_tuple)

        action_idx = torch.tensor([[self.action_to_idx[action]]], dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([is_done], dtype=torch.float32, device=self.device)

        if not is_done:
            next_state_tensor = self._preprocess_play_state(next_state_tuple)
        else:
            next_state_tensor = None

        self.play_memory.push(state_tensor, action_idx, reward_tensor, next_state_tensor, done_tensor)

    def remember_bet(self, bet_state_tuple, bet_action, total_hand_reward):
        """
        Stores a bet transition in the replay buffer (all tensors on device).
        In Stage 1 and 2, this is disabled; bet learning only happens in Stage 3
        (or legacy mode with training_stage=None).
        """
        # Stage 1 & 2: bet network is frozen
        if self.training_stage in (1, 2):
            return

        if self.freeze_bet_flag:
            return

        state_tensor = self._preprocess_bet_state(bet_state_tuple)
        action_idx = torch.tensor([[self.bet_action_to_idx[bet_action]]], dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor([total_hand_reward], dtype=torch.float32, device=self.device)

        # For bets we store next_state=None and done=True (episodic per hand)
        self.bet_memory.push(state_tensor, action_idx, reward_tensor, None, torch.tensor([True], device=self.device))

    # --- Training Methods ---

    def train_play_model(self):
        """
        Trains the play_policy_net on a batch from memory.
        Uses Double DQN for more stable targets if self.double_dqn is True.
        Disabled in Stage 1 (supervised only).
        """
        # Stage 1: no RL training for play
        if self.training_stage == 1:
            return 0.0

        if self.freeze_play_flag:
            return 0.0
        if len(self.play_memory) < self.batch_size:
            return 0.0

        transitions = self.play_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            dtype=torch.bool,
            device=self.device
        )

        if non_final_mask.any():
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)  # [B,1]
        reward_batch = torch.cat(batch.reward)  # [B]
        done_batch = torch.cat(batch.done)      # [B]

        # Q(s,a)
        q_sa = self.play_policy_net(state_batch).gather(1, action_batch).squeeze(1)  # [B]

        # Next-state values
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        if non_final_mask.any():
            with torch.no_grad():
                if self.double_dqn:
                    # Double DQN: action selection from policy net, evaluation from target net
                    next_policy_q = self.play_policy_net(non_final_next_states)
                    next_actions = next_policy_q.argmax(dim=1, keepdim=True)  # [N,1]

                    next_target_q = self.play_target_net(non_final_next_states)
                    next_q_vals = next_target_q.gather(1, next_actions).squeeze(1)  # [N]

                    next_state_values[non_final_mask] = next_q_vals
                else:
                    # Vanilla DQN
                    next_state_values[non_final_mask] = self.play_target_net(non_final_next_states).max(1)[0]

        # Bellman target
        target_q_sa = reward_batch + (self.gamma * next_state_values * (1.0 - done_batch))

        loss = nn.SmoothL1Loss()(q_sa, target_q_sa)

        self.play_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.play_policy_net.parameters(), self.max_grad_norm)

        self.play_optimizer.step()

        return loss.item()

    def train_bet_model(self):
        """
        Trains the bet_policy_net.
        Active only in Stage 3 (or legacy mode with training_stage=None).
        """
        # Stage 1 & 2: bet network is frozen by stage design
        if self.training_stage in (1, 2):
            return 0.0

        if self.freeze_bet_flag:
            return 0.0
        if len(self.bet_memory) < self.batch_size:
            return 0.0

        transitions = self.bet_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Clip rewards to reduce variance & prevent single huge wins from dominating training
        if self.bet_reward_clip:
            reward_batch = torch.clamp(reward_batch, -self.bet_spread, self.bet_spread)

        # Q(s,a)
        q_sa = self.bet_policy_net(state_batch).gather(1, action_batch).squeeze(1)
        target_q_sa = reward_batch  # episodic, next_state is None, done=True

        loss = nn.SmoothL1Loss()(q_sa, target_q_sa)

        self.bet_optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.bet_policy_net.parameters(), self.max_grad_norm)

        self.bet_optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """Call this at the end of each episode (e.g., each shoe)."""
        # Exponential decay, but enforce minimum
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # increment steps (used for warmup gating)
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.play_target_net.load_state_dict(self.play_policy_net.state_dict())
            self.bet_target_net.load_state_dict(self.bet_policy_net.state_dict())

    # --- Save / Load ---

    def save_model(self, file_path="rl_model.pth"):
        """Saves the learned model weights to CPU (for portability)."""
        try:
            # Move to cpu for saving portability
            self.play_policy_net.to('cpu')
            self.bet_policy_net.to('cpu')

            torch.save({
                'play_policy_state_dict': self.play_policy_net.state_dict(),
                'bet_policy_state_dict': self.bet_policy_net.state_dict(),
                'play_optimizer_state_dict': self.play_optimizer.state_dict(),
                'bet_optimizer_state_dict': self.bet_optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done
            }, file_path)

            # Move back to device
            self.play_policy_net.to(self.device)
            self.bet_policy_net.to(self.device)

            print(f"Model weights saved to {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, file_path="rl_model.pth"):
        """Loads learned model weights onto the correct device."""
        if os.path.exists(file_path):
            try:
                checkpoint = torch.load(file_path, map_location=self.device)

                self.play_policy_net.load_state_dict(checkpoint['play_policy_state_dict'])
                self.bet_policy_net.load_state_dict(checkpoint['bet_policy_state_dict'])

                # Attempt to load optimizer state if shapes align
                try:
                    self.play_optimizer.load_state_dict(checkpoint['play_optimizer_state_dict'])
                    self.bet_optimizer.load_state_dict(checkpoint['bet_optimizer_state_dict'])
                except Exception:
                    # optimizer shapes may differ depending on device or PyTorch version - ignore if incompatible
                    pass

                # restore epsilon/steps_done (useful for continuing training)
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.steps_done = checkpoint.get('steps_done', self.steps_done)

                self.play_target_net.load_state_dict(self.play_policy_net.state_dict())
                self.bet_target_net.load_state_dict(self.bet_policy_net.state_dict())

                # ensure optimizer tensors moved to device if necessary
                for state in self.play_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

                for state in self.bet_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

                print(f"Model loaded from {file_path}. Epsilon set to {self.epsilon}.")
            except Exception as e:
                print(f"Error loading model: {e}. Starting fresh.")
        else:
            print(f"No model file found at {file_path}. Starting fresh.")
