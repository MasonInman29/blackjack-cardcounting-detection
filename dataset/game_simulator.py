import numpy as np
from tqdm import tqdm
from helper import get_hand_value

class GameSimulator:
    """
    Simulates full games of blackjack using a provided model to make decisions.
    """

    def __init__(self, num_decks=8, deck_penetration=6.5, max_bet_size=20):
        self.card_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 11] # 21=J, 22=Q, 23=K, 11=A
        self.num_decks = num_decks
        self.deck_penetration = deck_penetration
        self.model = None  # Model should be set using set_model method
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
        return [card if card <= 20 else 10 for card in hand]
    
    def get_remaining_cards(self, shoe):
        remaining_cards = {val: shoe.count(val) for val in self.card_values}
        remaining_cards[10] += remaining_cards.get(21, 0) + remaining_cards.get(22, 0) + remaining_cards.get(23, 0)
        del remaining_cards[21]
        del remaining_cards[22]
        del remaining_cards[23]
        return remaining_cards

    def simulate_one_game(self):
        """
        Simulates one full shoe of blackjack.
        """
        shoe = self._create_shoe()
        total_winnings = 0
        
        # Determine the cut-off point based on penetration
        cards_in_shoe = self.num_decks * 52
        cut_off = cards_in_shoe - int(self.deck_penetration * 52)

        while len(shoe) > cut_off:
            bet = 1.0
            
            # Not enough cards for a full hand
            if len(shoe) < 4: break
            
            # Before dealing, get bet size from model if available
            remaining_cards = self.get_remaining_cards(shoe)
            bet_size = self.model.get_bet_size(remaining_cards)
            bet_size = min(bet_size, self.max_bet_size)
            bet *= bet_size

            # --- Dealing ---
            player_hand = [shoe.pop(), shoe.pop()]
            dealer_hand = [shoe.pop(), shoe.pop()]
            dealer_up_card = dealer_hand[0] if dealer_hand[0] <= 20 else 10

            # --- Blackjack Check ---
            player_bj = get_hand_value(self.transform_hand(player_hand)) == 21
            dealer_bj = get_hand_value(self.transform_hand(dealer_hand)) == 21

            if player_bj and dealer_bj:
                # Push
                continue
            elif player_bj:
                total_winnings += 1.5 * bet
                continue
            elif dealer_bj:
                total_winnings -= bet
                continue

            # --- Player's Turn ---
            # Use a list to manage hands for splitting
            player_hands = [player_hand]
            bets = [bet]
            hand_index = 0

            while hand_index < len(player_hands):
                current_hand = player_hands[hand_index]
                
                # Inner loop for actions on the current hand
                while True:
                    remaining_cards = self.get_remaining_cards(shoe)
                    action = self.model.predict(self.transform_hand(current_hand), dealer_up_card, remaining_cards)
                    
                    # --- Handle Split Action ('P') ---
                    is_pair = len(current_hand) == 2 and current_hand[0] == current_hand[1]
                    can_split = len(player_hands) < 4 # Common rule: max 4 hands
                    
                    assert action in ['H', 'S', 'D', 'P', 'R'], f"Invalid action '{action}' predicted by model."
                    if action == 'P':
                        if is_pair and can_split:
                            card1 = current_hand[0]
                            card2 = current_hand[1]
                            
                            # Replace current hand with the first new hand
                            player_hands[hand_index] = [card1, shoe.pop()]
                            
                            # Insert the second new hand and its bet
                            player_hands.insert(hand_index + 1, [card2, shoe.pop()])
                            bets.insert(hand_index + 1, bets[hand_index])
                            
                            # Restart action loop for the newly formed hand
                            current_hand = player_hands[hand_index]
                            continue
                        else:
                            action = 'S'

                    if action == 'S': # Stand
                        break
                    
                    if action == 'R': # Surrender
                        if len(current_hand) == 2:
                            total_winnings -= 0.5 * bets[hand_index]
                            current_hand.clear() # Signal that hand is over
                            break
                        else: # Cannot surrender after hitting, treat as hit
                            action = 'H'

                    if action == 'D': # Double Down
                        if len(current_hand) == 2:
                            bets[hand_index] *= 2.0
                            current_hand.append(shoe.pop())
                            break # Doubling down ends the turn for this hand
                        else: # Cannot double after hitting, treat as hit
                            action = 'H'

                    if action == 'H': # Hit
                        current_hand.append(shoe.pop())
                        if get_hand_value(self.transform_hand(current_hand)) > 21:
                            break # Player busts
                
                hand_index += 1
            
            # --- Dealer's Turn ---
            dealer_busted = False
            dealer_value = get_hand_value(self.transform_hand(dealer_hand))
            
            while dealer_value < 17 or (dealer_value == 17 and 11 in dealer_hand):
                if not shoe: break
                dealer_hand.append(shoe.pop())
                dealer_value = get_hand_value(self.transform_hand(dealer_hand))

            if dealer_value > 21:
                dealer_busted = True

            # --- Outcome Resolution for all Player Hands ---
            for i, final_hand in enumerate(player_hands):
                if not final_hand: # Hand was surrendered
                    continue
                
                player_value = get_hand_value(self.transform_hand(final_hand))
                
                if player_value > 21: # Player busted
                    total_winnings -= bets[i]
                    continue
                
                if dealer_busted or player_value > dealer_value:
                    total_winnings += bets[i]
                elif player_value < dealer_value:
                    total_winnings -= bets[i]
                # If equal, it's a push, winnings don't change.

        return total_winnings

    def run_multiple_simulations(self, num_games=100):
        """
        Runs multiple game simulations and computes the average result.
        """
        if self.model is None:
            raise ValueError("Model not set. Please set a model using set_model() before running simulations.")
        
        print("\n--- Full Simulation ---")
        print(f"Running {num_games} simulations...")
        results = []
        for i in tqdm(range(num_games), desc="Simulating Games"):
            # Set a seed for each game to ensure the card order is the same
            # when testing different models on the same simulation number.
            np.random.seed(i)
            result = self.simulate_one_game()
            results.append(result)

        average_ev = np.mean(results)
        
        print(f"Total shoes simulated: {num_games}")
        print(f"Average Winnings/Losses (EV) per shoe: {average_ev:.4f} units")
        
        return average_ev
