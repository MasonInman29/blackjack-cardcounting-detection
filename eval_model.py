from dataset import BlackjackDataset, GameSimulator, game_simulator
from model import NaiveStrategy


def calculate_accuracy(dataset, model, split_name='test'):
    """
    Calculates the accuracy of a given model on a specified data split.

    The model is expected to have a `predict` method that takes the player's
    initial hand, the dealer's up card, and the remaining card counts as input,
    and returns a single action ('H', 'S', 'D', 'R').

    Args:
        dataset (BlackjackDataset): The dataset object containing the data splits.
        model: A model object with a `predict(player_hand, dealer_up_card, remaining_cards)` method.
        split_name (str): The data split to evaluate on ('train', 'val', or 'test').

    Returns:
        float: The accuracy of the model on the specified split (0.0 to 1.0).
    """
    df_split = dataset.get_split(split_name)
    if df_split.empty:
        print(f"Warning: {split_name} split is empty. Returning 0 accuracy.")
        return 0.0

    # Define a helper function to apply the model's prediction
    def get_prediction(row):
        return model.predict(
            player_hand=row['initial_hand'],
            dealer_up_card=row['dealer_up'],
            remaining_cards=row['remaining_card_counts']
        )

    print(f"Generating predictions for '{split_name}' split...")
    predictions = df_split.progress_apply(get_prediction, axis=1)

    # Compare predictions to the optimal action determined by EV
    correct_predictions = (predictions == df_split['best_action_by_ev']).sum()
    total_predictions = len(df_split)
    
    if total_predictions == 0:
        print("No predictions were made.")
        return 0.0

    accuracy = correct_predictions / total_predictions
    # print(f"Model accuracy on '{split_name}' split: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    return accuracy


if __name__ == "__main__":
    # Load dataset
    dataset = BlackjackDataset(csv_path='blackjack_simulator.csv', num_simulations=100, num_data_limit=1000) 
    # download 'blackjack_simulator.csv' from https://www.kaggle.com/datasets/dennisho/blackjack-hands/data
    # use a small num_data_limit for debugging.

    # Initialize model
    model = NaiveStrategy()
    
    # Calculate accuracy on test split
    test_accuracy = calculate_accuracy(dataset, model, split_name='test')
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    
    # Calculate average EV by running simulations
    game_simulator = GameSimulator(model=model)
    average_ev = game_simulator.run_multiple_simulations(num_games=10, num_decks=8, deck_penetration=6.5)