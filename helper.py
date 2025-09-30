# Helper functions for a card game

def get_hand_value(hand):
    """Helper function to calculate the value of a hand, handling Aces correctly."""
    value = sum(hand)
    num_aces = hand.count(11)
    while value > 21 and num_aces > 0:
        value -= 10
        num_aces -= 1
    return value