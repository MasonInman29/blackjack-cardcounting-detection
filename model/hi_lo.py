from helper import get_hand_value

class BasicStrategyModel:
    """
    A simple model that mimics the basic strategy chart for making decisions.
    """
    def predict(self, player_hand, dealer_up_card, remaining_cards):
        player_total = get_hand_value(player_hand)
        # In blackjack, an Ace can be 1 or 11. A "soft" hand is one with an Ace
        # counted as 11, which doesn't bust the hand.
        is_soft = 11 in player_hand and sum(player_hand) - 10 < 21

        if is_soft:
            if player_total >= 19: return 'S'
            if player_total == 18:
                return 'S' if dealer_up_card <= 8 else 'H'
            return 'H'

        if player_total >= 17: return 'S'
        if 13 <= player_total <= 16 and dealer_up_card <= 6: return 'S'
        if player_total == 12 and 4 <= dealer_up_card <= 6: return 'S'

        if len(player_hand) == 2:
            if player_total == 11: return 'D'
            if player_total == 10 and dealer_up_card <= 9: return 'D'
            if player_total == 9 and 3 <= dealer_up_card <= 6: return 'D'
            if player_total == 16 and dealer_up_card >= 9: return 'R'
            if player_total == 15 and dealer_up_card == 10: return 'R'

        return 'H'

    def get_bet_size(self, remaining_cards):
        """Always bets the table minimum."""
        return 1.0

class HILO:
    """
    A model that implements the Hi-Lo card counting strategy.
    It deviates from basic strategy based on the true count.
    """
    def __init__(self, num_decks=8, bet_spread=10):
        self.num_decks = num_decks
        self.bet_spread = bet_spread
        # Define the initial counts for a full shoe
        self.initial_low_cards = 20 * self.num_decks # (2, 3, 4, 5, 6)
        self.initial_high_cards = 20 * self.num_decks # (10, J, Q, K, A)

    def _get_true_count(self, remaining_cards):
        """Calculates the running count and true count."""
        
        current_low = sum(remaining_cards.get(c, 0) for c in [2, 3, 4, 5, 6])
        current_high = sum(remaining_cards.get(c, 0) for c in [10, 11])
        
        low_cards_played = self.initial_low_cards - current_low
        high_cards_played = self.initial_high_cards - current_high
        
        running_count = low_cards_played - high_cards_played
        
        decks_remaining = sum(remaining_cards.values()) / 52.0
        
        # Avoid division by zero at the end of a shoe
        return running_count / decks_remaining if decks_remaining > 0.25 else 0

    def _get_basic_strategy_action(self, player_hand, dealer_up_card):
        """Returns the basic strategy action for a given hand."""
        player_total = get_hand_value(player_hand)
        is_soft = 11 in player_hand and sum(player_hand) - 10 < 21
        
        # --- Pair Splitting Logic ---
        if len(player_hand) == 2 and player_hand[0] == player_hand[1]:
            pair_val = player_hand[0]
            
            if pair_val == 11: return 'P' # Always split Aces
            if pair_val == 8: return 'P'  # Always split 8s
            
            if pair_val == 10: return 'S' # Never split 10s (Hard 20)
            
            if pair_val == 9: # 9s (Hard 18)
                if dealer_up_card in [2, 3, 4, 5, 6, 8, 9]:
                    return 'P'
                return 'S' # Stand vs 7, 10, A
            
            if pair_val == 7: # 7s (Hard 14)
                if dealer_up_card <= 7:
                    return 'P'
                # else, fall through to Hard 14 logic (Hit)
            
            if pair_val == 6: # 6s (Hard 12)
                if dealer_up_card <= 6:
                    return 'P'
                # else, fall through to Hard 12 logic (Hit)
            
            if pair_val == 5: # 5s (Hard 10)
                # Never split, fall through to Hard 10 logic (Double/Hit)
                pass 
            
            if pair_val == 4: # 4s (Hard 8)
                # Most strategies say Hit, some split vs 5, 6.
                # We will follow "Hit" to keep it simple.
                # Fall through to Hard 8 logic (Hit)
                pass
            
            if pair_val in [2, 3]: # 2s, 3s
                if dealer_up_card <= 7:
                    return 'P'
                # else, fall through to Hard 4/6 logic (Hit)
        
        # --- Standard Soft Hand Logic ---
        if is_soft:
            if player_total >= 19: return 'S'
            if player_total == 18:
                return 'S' if dealer_up_card <= 8 else 'H'
            return 'H'
        
        # --- Standard Hard Hand Logic ---
        if player_total >= 17: return 'S'
        if 13 <= player_total <= 16 and dealer_up_card <= 6: return 'S'
        if player_total == 12 and 4 <= dealer_up_card <= 6: return 'S'
        
        # --- Standard Double/Surrender Logic ---
        if len(player_hand) == 2:
            if player_total == 11: return 'D'
            if player_total == 10 and dealer_up_card <= 9: return 'D'
            if player_total == 9 and 3 <= dealer_up_card <= 6: return 'D'
            # Surrender (R) rules from original code
            if player_total == 16 and dealer_up_card >= 9: return 'R'
            if player_total == 15 and dealer_up_card == 10: return 'R'

        return 'H'
    
    def transform_hand(self, hand):
        """Converts J, Q, K (21, 22, 23) to 10 for value calculation."""
        return [card if card <= 20 else 10 for card in hand]
        
    def predict(self, player_hand, dealer_up_card, remaining_cards):
        """
        Predicts the best action using Hi-Lo strategy deviations.
        Falls back to basic strategy if no deviation is triggered.
        """
        true_count = self._get_true_count(remaining_cards)
        is_pair = len(player_hand) == 2 and player_hand[0] == player_hand[1]
        player_hand = self.transform_hand(player_hand)
        player_total = get_hand_value(player_hand)

        # --- Hi-Lo Deviations (based on "Illustrious 18" + Splits) ---

        # Pair-specific deviations (e.g., splitting 10s)
        if is_pair:
            pair_val = player_hand[0]
            if pair_val == 10:
                if dealer_up_card == 5 and true_count >= 5: return 'P'
                if dealer_up_card == 6 and true_count >= 4: return 'P'
                # Note: No other common split deviations in I18
                # Basic strategy for 10s is 'S', which will be
                # caught by the fallback logic.

        if player_total == 16 and dealer_up_card == 10 and true_count >= 0: return 'S'
        if player_total == 15 and dealer_up_card == 10 and true_count >= 4: return 'S'
        if player_total == 10 and dealer_up_card == 10 and true_count >= 4: return 'D'
        if player_total == 12 and dealer_up_card == 3 and true_count >= 2: return 'S'
        if player_total == 12 and dealer_up_card == 2 and true_count >= 3: return 'S'
        if player_total == 11 and dealer_up_card == 11 and true_count >= 1: return 'D'
        if player_total == 9 and dealer_up_card == 2 and true_count >= 1: return 'D'
        if player_total == 10 and dealer_up_card == 11 and true_count >= 4: return 'D'
        if player_total == 9 and dealer_up_card == 7 and true_count >= 3: return 'D'
        if player_total == 16 and dealer_up_card == 9 and true_count >= 5: return 'S'
        if player_total == 15 and dealer_up_card == 9 and true_count >= 2: return 'R'
        
        # --- Fallback to Basic Strategy ---
        return self._get_basic_strategy_action(player_hand, dealer_up_card)

    def get_bet_size(self, remaining_cards):
        true_count = self._get_true_count(remaining_cards)
        
        if true_count < 2:
            bet_size = 1.0
        elif true_count < 3:
            bet_size = 2.0
        elif true_count < 4:
            bet_size = 4.0
        elif true_count < 5:
            bet_size = 8.0
        else: # TC >= 5
            bet_size = self.bet_spread
        
        return max(1.0, min(self.bet_spread, bet_size))