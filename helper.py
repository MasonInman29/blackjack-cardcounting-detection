# Helper functions for a card game
import numpy as np

def get_hand_value(hand):
    """Helper function to calculate the value of a hand, handling Aces correctly."""
    if isinstance(hand, np.ndarray):
        hand = hand.tolist()
    value = sum(hand)
    num_aces = hand.count(11)
    while value > 21 and num_aces > 0:
        value -= 10
        num_aces -= 1
    return value

def get_hand_value_and_soft_aces_count(hand):
    """Helper function to calculate the value of a hand, handling and returning the number of Aces correctly."""
    if isinstance(hand, np.ndarray):
        hand = hand.tolist()
    value = sum(hand)
    num_aces = hand.count(11)
    while value > 21 and num_aces > 0:
        value -= 10
        num_aces -= 1
    return value, num_aces