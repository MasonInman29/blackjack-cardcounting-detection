import numpy as np
from tqdm import tqdm
from helper import get_hand_value

class GameSimulator:
    """
    Simulates full games of blackjack using a provided model to make decisions.
    """

    def __init__(self, num_decks=8, deck_penetration=6.5):
        self.card_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
        self.num_decks = num_decks
        self.deck_penetration = deck_penetration
        self.model = None  # Model should be set using set_model method

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
            remaining_cards = {val: shoe.count(val) for val in self.card_values}
            bet_size = self.model.get_bet_size(remaining_cards)
            bet *= bet_size

            # --- Dealing ---
            player_hand = [shoe.pop(), shoe.pop()]
            dealer_hand = [shoe.pop(), shoe.pop()]
            dealer_up_card = dealer_hand[0]

            # --- Blackjack Check ---
            player_bj = get_hand_value(player_hand) == 21
            dealer_bj = get_hand_value(dealer_hand) == 21

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
            player_busted = False
            while True:
                remaining_cards = {val: shoe.count(val) for val in self.card_values}
                action = self.model.predict(player_hand, dealer_up_card, remaining_cards)
                
                if action == 'S':
                    break
                
                if action == 'R':
                    if len(player_hand) == 2:
                        total_winnings -= 0.5 * bet
                        player_hand = [] # Signal that hand is over
                        break
                    else: # Cannot surrender after hitting, treat as hit
                        action = 'H'

                if action == 'D':
                    if len(player_hand) == 2:
                        bet *= 2.0
                        player_hand.append(shoe.pop())
                        if get_hand_value(player_hand) > 21:
                            player_busted = True
                            break
                    else: # Cannot double after hitting, treat as hit
                        action = 'H'

                if action == 'H':
                    player_hand.append(shoe.pop())
                    if get_hand_value(player_hand) > 21:
                        player_busted = True
                        break
            
            # --- Outcome Resolution ---
            if not player_hand: # Player surrendered
                continue

            if player_busted:
                total_winnings -= bet
                continue
            
            # --- Dealer's Turn ---
            while get_hand_value(dealer_hand) < 17 or (get_hand_value(dealer_hand) == 17 and 11 in dealer_hand):
                dealer_hand.append(shoe.pop())

            player_value = get_hand_value(player_hand)
            dealer_value = get_hand_value(dealer_hand)

            if dealer_value > 21 or player_value > dealer_value:
                total_winnings += bet
            elif player_value < dealer_value:
                total_winnings -= bet
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
