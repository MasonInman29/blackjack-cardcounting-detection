from helper import get_hand_value

class NaiveStrategy:
    def __init__(self):
        pass

    def predict(self, player_hand, dealer_up_card, remaining_cards):
        hand_value = get_hand_value(player_hand)
        if hand_value >= 15:
            return 'S'
        else:
            return 'H'