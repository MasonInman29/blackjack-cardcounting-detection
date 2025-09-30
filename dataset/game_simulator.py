import numpy as np
from tqdm import tqdm
from helper import get_hand_value

class GameSimulator:
    """
    Simulates full games of blackjack using a provided model to make decisions.
    """

    def __init__(self, model):
        """
        Initializes the GameSimulator.

        Args:
            model: A model object with a `predict(player_hand, dealer_up_card, remaining_cards)` method.
        """
        if not hasattr(model, 'predict'):
            raise TypeError("The provided model must have a 'predict' method.")
        self.model = model
        self.card_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

    def _create_shoe(self, num_decks=8):
        """Creates and shuffles a shoe with the specified number of decks."""
        shoe = self.card_values * 4 * num_decks
        np.random.shuffle(shoe)
        return shoe

    def simulate_one_game(self, num_decks=8, deck_penetration=6.5):
        """
        Simulates one full shoe of blackjack.

        Args:
            deck_penetration (float): The number of decks to be played before shuffling.

        Returns:
            float: The total winnings or losses for the simulated shoe.
        """
        shoe = self._create_shoe(num_decks)
        total_winnings = 0
        
        # Determine the cut-off point based on penetration
        cards_in_shoe = num_decks * 52
        cut_off = cards_in_shoe - int(deck_penetration * 52)

        while len(shoe) > cut_off:
            bet = 1.0
            
            # Not enough cards for a full hand
            if len(shoe) < 4: break

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

    def run_multiple_simulations(self, num_games=100, num_decks=8, deck_penetration=6.5):
        """
        Runs multiple game simulations and computes the average result.

        Args:
            num_games (int): The number of full shoes to simulate.
            deck_penetration (float): The deck penetration for each game.

        Returns:
            float: The average EV (winnings/losses) per shoe.
        """
        print(f"Running {num_games} simulations...")
        results = []
        for _ in tqdm(range(num_games), desc="Simulating Games"):
            result = self.simulate_one_game(num_decks, deck_penetration)
            results.append(result)

        average_ev = np.mean(results)
        
        print("\n--- Simulation Results ---")
        print(f"Total shoes simulated: {num_games}")
        print(f"Average Winnings/Losses (EV) per shoe: {average_ev:.4f} units")
        
        return average_ev
