from model import RLModel, HILO
import numpy as np
from tqdm import tqdm
from helper import get_hand_value
import matplotlib.pyplot as plt


class GameSimulator:
    """
    Simulates full games of blackjack using a provided model to make decisions.
    Can be run in 'training_mode' to feed back rewards to an RL model.
    """

    def __init__(self, num_decks=8, deck_penetration=6.5, max_bet_size=20):
        self.card_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 11] # 21=J, 22=Q, 23=K, 11=A
        self.num_decks = num_decks
        self.deck_penetration = deck_penetration
        self.model = None
        self.max_bet_size = max_bet_size

    def _create_shoe(self):
        """Creates and shuffles a shoe with the specified number of decks."""
        shoe = self.card_values * 4 * self.num_decks
        np.random.shuffle(shoe)
        return shoe
    
    def set_model(self, model):
        self.model = model
        
    def set_deck_info(self, num_decks, deck_penetration):
        self.num_decks = num_decks
        self.deck_penetration = deck_penetration
        
    def transform_hand(self, hand):
        """Converts J, Q, K (21, 22, 23) to 10 for value calculation."""
        return [card if card <= 20 else 10 for card in hand]
    
    def get_remaining_cards(self, shoe):
        """Gets the count of remaining cards, grouping 10, J, Q, K."""
        remaining_cards = {val: 0 for val in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
        for card in shoe:
            if card == 21 or card == 22 or card == 23:
                remaining_cards[10] += 1
            else:
                remaining_cards[card] += 1
        return remaining_cards

    def simulate_one_game(self, training_mode=False):
        """
        Simulates one full shoe of blackjack.
        If training_mode is True, it will call update methods on the model.
        """
        shoe = self._create_shoe()
        total_winnings = 0
        
        cards_in_shoe = self.num_decks * 52
        cut_off = cards_in_shoe - int(self.deck_penetration * 52)

        while len(shoe) > cut_off:
            # Not enough cards for a full hand
            if len(shoe) < 4: break
            
            # --- 1. Get Bet Size and State (for RL) ---
            remaining_cards = self.get_remaining_cards(shoe)
            
            # Get the state-action pair for the betting decision
            bet_state = None
            if training_mode:
                # We must get the state *before* the model predicts
                bet_state = self.model._get_bet_state(remaining_cards)
            
            bet_size = self.model.get_bet_size(remaining_cards)
            bet_size = min(bet_size, self.max_bet_size)
            
            # This will track all profits/losses for this one bet decision
            hand_profit_for_bet_update = 0.0

            # --- Dealing ---
            player_hand = [shoe.pop(), shoe.pop()]
            dealer_hand = [shoe.pop(), shoe.pop()]
            dealer_up_card_val = dealer_hand[0] if dealer_hand[0] <= 20 else 10

            # --- Blackjack Check ---
            player_bj = get_hand_value(self.transform_hand(player_hand)) == 21
            dealer_bj = get_hand_value(self.transform_hand(dealer_hand)) == 21
            
            # print(f"Player Hand: {player_hand}, Dealer Hand: {dealer_hand}, Bet Size: {bet_size}, Total Winnings: {total_winnings}")

            if player_bj and dealer_bj:
                # Push. No profit, no loss.
                if training_mode:
                    self.model.update_bet(bet_state, bet_size, 0.0)
                continue
            elif player_bj:
                profit = 1.5 * bet_size
                total_winnings += profit
                if training_mode:
                    self.model.update_bet(bet_state, bet_size, profit)
                continue
            elif dealer_bj:
                profit = -1.0 * bet_size
                total_winnings += profit
                if training_mode:
                    self.model.update_bet(bet_state, bet_size, profit)
                continue

            # --- Player's Turn ---
            # Use a list of dicts to manage hands, bets, and RL state
            player_hands = [{
                'hand': player_hand, 
                'bet': bet_size, 
                'final_sa': None # To store (state, action) for deferred update (Stand/Double)
            }]
            hand_index = 0

            while hand_index < len(player_hands):
                current_hand = player_hands[hand_index]['hand']
                current_bet = player_hands[hand_index]['bet']
                
                # --- Action Loop for this hand ---
                while True:
                    if not shoe: break # Out of cards
                    
                    # --- 2. Get Play State and Action (for RL) ---
                    remaining_cards = self.get_remaining_cards(shoe)
                    transformed_hand = self.transform_hand(current_hand)
                    
                    play_state = None
                    if training_mode:
                        play_state = self.model._get_play_state(
                            current_hand, dealer_up_card_val, remaining_cards
                        )
                        
                    action = self.model.predict(
                        current_hand, dealer_up_card_val, remaining_cards
                    )
                    
                    # print(f"Current Hand: {current_hand}, Action: {action}")
                    
                    # --- Handle Split Action ('P') ---
                    is_pair = len(current_hand) == 2 and current_hand[0] == current_hand[1]
                    can_split = len(player_hands) < 4 # Max 4 hands
                    
                    assert action in ['H', 'S', 'D', 'P', 'R'], f"Invalid action '{action}' predicted."
                    
                    if action == 'P':
                        if is_pair and can_split and len(shoe) >= 2:
                            card1 = current_hand[0]
                            card2 = current_hand[1]
                            
                            # Create two new hands, update lists
                            hand1 = [card1, shoe.pop()]
                            hand2 = [card2, shoe.pop()]
                            
                            player_hands[hand_index] = {'hand': hand1, 'bet': current_bet, 'final_sa': None}
                            player_hands.insert(hand_index + 1, {'hand': hand2, 'bet': current_bet, 'final_sa': None})
                            
                            # if training_mode:
                            #     # Update Q-table for the 'P' action.
                            #     # Reward is 0. Next state is the state of the first new hand.
                            #     # This is an approximation, but a common one.
                            #     next_state = self.model._get_play_state(
                            #         hand1, dealer_up_card_val, self.get_remaining_cards(shoe)
                            #     )
                            #     self.model.update_play(play_state, action, 0.0, next_state, False)
                            
                            # Restart action loop for the newly formed hand
                            current_hand = player_hands[hand_index]['hand']
                            continue
                        else:
                            action = 'S'

                    if action == 'S': # Stand
                        if training_mode:
                            # Defer update until after dealer plays
                            player_hands[hand_index]['final_sa'] = (play_state, action)
                        break
                    
                    if action == 'R': # Surrender
                        if len(current_hand) == 2:
                            reward = -0.5 # Normalized reward
                            profit = reward * current_bet
                            total_winnings += profit
                            hand_profit_for_bet_update += profit
                            
                            # if training_mode:
                            #     self.model.update_play(play_state, action, reward, None, True)
                            
                            current_hand.clear() # Signal that hand is over
                            break
                        else: # Cannot surrender, treat as Hit
                            action = 'H'

                    if action == 'D': # Double Down
                        if len(current_hand) == 2 and len(shoe) >= 1:
                            player_hands[hand_index]['bet'] *= 2.0
                            current_hand.append(shoe.pop())
                            player_value = get_hand_value(self.transform_hand(current_hand))
                            
                            if player_value > 21:
                                reward = -1.0
                                profit = reward * player_hands[hand_index]['bet']
                                total_winnings += profit
                                hand_profit_for_bet_update += profit
                                
                                # if training_mode:
                                #     self.model.update_play(play_state, action, profit / bet_size, None, True)
                            
                            else:
                                if training_mode:
                                    # Defer update until after dealer plays
                                    player_hands[hand_index]['final_sa'] = (play_state, action)
                            break # Doubling down ends the turn
                        else: # Cannot double, treat as Hit
                            action = 'H'

                    if action == 'H': # Hit
                        if not shoe: break
                        current_hand.append(shoe.pop())
                        player_value = get_hand_value(self.transform_hand(current_hand))
                        
                        if player_value > 21:
                            # --- Player Busts ---
                            reward = -1.0 # Normalized reward
                            profit = reward * current_bet
                            total_winnings += profit
                            hand_profit_for_bet_update += profit
                            
                            # if training_mode:
                            #     self.model.update_play(play_state, action, reward, None, True)
                            break # Hand is over
                        # else:
                            # --- Hit Successful ---
                            # if training_mode:
                            #     # Update with 0 reward, provide next state
                            #     next_state = self.model._get_play_state(
                            #         current_hand, dealer_up_card_val, self.get_remaining_cards(shoe)
                            #     )
                            #     self.model.update_play(play_state, action, 0.0, next_state, False)
                            # Continue 'while True' loop for next action
                
                hand_index += 1
            
            # --- Dealer's Turn ---
            dealer_busted = False
            dealer_value = get_hand_value(self.transform_hand(dealer_hand))
            
            while dealer_value < 17 or (dealer_value == 17 and 11 in self.transform_hand(dealer_hand)): # S17
                if not shoe: break
                dealer_hand.append(shoe.pop())
                dealer_value = get_hand_value(self.transform_hand(dealer_hand))

            if dealer_value > 21:
                dealer_busted = True
                
            # print(f"Dealer Final Hand: {dealer_hand}, Dealer Value: {dealer_value}")
            # print(f"Player Hands at Resolution: {[hand_data['hand'] for hand_data in player_hands]}")
            # print(f"Player Values at Resolution: {[get_hand_value(self.transform_hand(hand_data['hand'])) if hand_data['hand'] else 'Surrendered' for hand_data in player_hands]}")
            # print(f"Player Bets at Resolution: {[hand_data['bet'] for hand_data in player_hands]}")
            # print("total_winnings so far:", total_winnings)
            # print("action:", action)
            # breakpoint()

            # --- 3. Outcome Resolution and Final RL Updates ---
            for hand_data in player_hands:
                final_hand = hand_data['hand']
                if not final_hand: # Hand was surrendered (already updated)
                    continue
                
                player_value = get_hand_value(self.transform_hand(final_hand))
                
                if player_value > 21: # Player busted (already updated)
                    continue
                
                # Determine winner
                final_profit = 0.0
                hand_bet = hand_data['bet']
                
                if dealer_busted or player_value > dealer_value:
                    final_profit = hand_bet
                elif player_value < dealer_value:
                    final_profit = -hand_bet
                # else: Push, final_profit = 0.0
                
                total_winnings += final_profit
                hand_profit_for_bet_update += final_profit

                # --- Apply Deferred Updates for 'S' and 'D' ---
                if training_mode and hand_data['final_sa'] is not None:
                    (state, action) = hand_data['final_sa']
                    
                    # Normalize reward by the *initial* bet, not the doubled one
                    # e.g., Doubled and won: profit=2*bet_size. reward=2.0
                    # e.g., Stood and lost: profit=-bet_size. reward=-1.0
                    normalized_reward = final_profit / bet_size 
                    
                    # self.model.update_play(state, action, normalized_reward, None, True)
            
            # --- 4. Update Bet Q-Table (Once per initial hand) ---
            if training_mode:
                self.model.update_bet(bet_state, bet_size, hand_profit_for_bet_update)
        
        # --- 5. End of Shoe ---
        if training_mode:
            self.model.decay_epsilon()
            
        return total_winnings

    def run_multiple_simulations(self, num_games=100, training_mode=False):
        """
        Runs multiple game simulations.
        If training_mode=True, trains the model.
        If training_mode=False, evaluates the model with fixed seeds.
        """
        if self.model is None:
            raise ValueError("Model not set. Please set a model using set_model().")
        
        mode_desc = "Training" if training_mode else "Evaluating"
        print(f"\n--- Running {num_games} Simulations ({mode_desc}) ---")
        
        results = []
        for i in tqdm(range(num_games), desc=f"Simulating Games ({mode_desc})"):
            
            if not training_mode:
                # Use fixed seed for evaluation to get reproducible results
                np.random.seed(i)
            else:
                # For training, we want random, different shoes each time
                np.random.seed(None) 
            
            result = self.simulate_one_game(training_mode=training_mode)
            results.append(result)

        average_ev = np.mean(results)
        
        print(f"Total shoes simulated: {num_games}")
        print(f"Average Winnings/Losses (EV) per shoe: {average_ev:.4f} units")
        
        return average_ev


if __name__ == '__main__':
    # --- Hyperparameters ---
    STAGE = 1
    NUM_DECKS = 8
    DECK_PENETRATION = 6.5
    MAX_BET_SPREAD = 20
    MODEL_FILE_PATH = "blackjack_rl_model_bet_size_only_20x4_1e-6.pkl"
    plot_filename = 'training_progress_bet_size_20x4_1e-6.png'

    TRAINING_SHOES = 5000000
    EVALUATION_SHOES = 10000
    
    # How often to check progress during training
    PROGRESS_CHUNK_SIZE = 10000
    PROGRESS_EVAL_SHOES = 10000
    
    SAVE_INTERVAL = 100000

    # --- Initialize Models and Simulator ---
    rl_model = RLModel(
        num_decks=NUM_DECKS,
        bet_spread=MAX_BET_SPREAD,
        epsilon_decay=0.99999885
    )
    baseline_model = HILO(num_decks=NUM_DECKS, bet_spread=MAX_BET_SPREAD)
    rl_model.baseline_model = baseline_model
    rl_model.initialize_q_table_from_hilo()
    
    simulator = GameSimulator(
        num_decks=NUM_DECKS,
        deck_penetration=DECK_PENETRATION,
        max_bet_size=MAX_BET_SPREAD
    )

    # --- 1. Training Phase ---
    print("--- Starting Training Phase ---")
    training_progress = []
    shoes_axis = []
    
    # Optional: Load a previously trained model to continue training
    # rl_model.load_model(MODEL_FILE_PATH)

    simulator.set_model(rl_model)
    
    for i in range(TRAINING_SHOES):
        simulator.simulate_one_game(training_mode=True)
        
        if (i + 1) % PROGRESS_CHUNK_SIZE == 0:
            print(f"\n--- Training Progress Check at Shoe {i+1}/{TRAINING_SHOES} ---")
            print(f"Current Epsilon: {rl_model.epsilon:.4f}")
            
            bet_q_table_keys = sorted(list(rl_model.bet_q_table.keys()))
            print("Model bet table best actions:")
            for _idx, bet_q_table_key in enumerate(bet_q_table_keys):
                best_action = max(rl_model.bet_q_table[bet_q_table_key], key=rl_model.bet_q_table[bet_q_table_key].get)
                delta_ev = float(rl_model.bet_q_table[bet_q_table_key][20.0]) - float(rl_model.bet_q_table[bet_q_table_key][1.0])
                print(f"{bet_q_table_key}: {best_action}, {delta_ev:.7f}", end=';\t' if _idx % 5 != 4 else '\n')
            # Temporarily reduce epsilon for a more accurate evaluation of the current policy
            original_epsilon = rl_model.epsilon
            rl_model.epsilon = 0.0
            
            # Run a small evaluation
            avg_winnings = simulator.run_multiple_simulations(
                num_games=PROGRESS_EVAL_SHOES, 
                training_mode=False # Important: use eval mode
            )
            training_progress.append(avg_winnings)
            shoes_axis.append(i + 1)
            
            print(f"EV over last {PROGRESS_EVAL_SHOES} shoes: {avg_winnings:.4f} units")
            
            # Restore epsilon to continue training
            rl_model.epsilon = original_epsilon
            
            print("\n--- Generating Training Plot ---")
            plt.figure(figsize=(12, 7))
            plt.plot(shoes_axis, training_progress, marker='o', linestyle='-', label='RL Model EV')
            
            # Add a horizontal line for the baseline EV
            # plt.axhline(y=baseline_ev, color='r', linestyle='--', label=f'Baseline EV ({baseline_ev:.4f})')
            
            plt.title('RL Model Training Progress')
            plt.xlabel('Number of Training Shoes')
            plt.ylabel('Average Winnings (EV) per Shoe (units)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(plot_filename)
            print(f"Plot saved as {plot_filename}")
            
            if (i + 1) % SAVE_INTERVAL == 0:
                print(f"\n--- Saving Model at Shoe {i+1} ---")
                cur_file_path = f"blackjack_rl_model_bet_size_only_shoe_{i+1}.pkl"
                rl_model.save_model(cur_file_path)

    print("\n--- Training Complete ---")
    rl_model.save_model(MODEL_FILE_PATH)

    # --- 2. Evaluation Phase ---
    print("\n--- Starting Final Evaluation Phase ---")
    
    
    # A) Evaluate the Baseline Model for comparison
    # print("\nEvaluating Baseline (HILO) Model...")
    # simulator.set_model(baseline_model)
    # baseline_ev = simulator.run_multiple_simulations(
    #     num_games=EVALUATION_SHOES,
    #     training_mode=False
    # )
    # print(f"Baseline Model Final EV: {baseline_ev:.4f} units per shoe")
    
    # B) Evaluate the trained RL Model
    print("\nEvaluating Trained RL Model...")
    eval_rl_model = RLModel(num_decks=NUM_DECKS)
    eval_rl_model.load_model(MODEL_FILE_PATH) # Loads and sets epsilon low
    simulator.set_model(eval_rl_model)
    rl_ev = simulator.run_multiple_simulations(
        num_games=EVALUATION_SHOES, 
        training_mode=False
    )
    print(f"Trained RL Model Final EV: {rl_ev:.4f} units per shoe")


    # --- 3. Visualization ---
    if training_progress:
        print("\n--- Generating Training Plot ---")
        plt.figure(figsize=(12, 7))
        plt.plot(shoes_axis, training_progress, marker='o', linestyle='-', label='RL Model EV')
        
        # Add a horizontal line for the baseline EV
        # plt.axhline(y=baseline_ev, color='r', linestyle='--', label=f'Baseline EV ({baseline_ev:.4f})')
        
        plt.title('RL Model Training Progress')
        plt.xlabel('Number of Training Shoes')
        plt.ylabel('Average Winnings (EV) per Shoe (units)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.show()