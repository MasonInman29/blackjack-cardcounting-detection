from dataset import BlackjackDataset, GameSimulator, CSVDataset, ParquetDataset
from model import NaiveStrategy, BasicStrategyModel, HILO, RLModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from helper import get_hand_value

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

# Create output directory for figures
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join('figures', TIMESTAMP)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, filename):
    """Helper to save figures into figures/<run_datetime>/<figurename>.png."""
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.png")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filepath}")
    plt.close(fig)
    return filepath


def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
    """Plot confusion matrix for action predictions."""
    from sklearn.metrics import confusion_matrix
    
    actions = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=actions)
    
    # Variable sizing based on number of actions
    size = max(6, len(actions) * 1.5)
    fig, ax = plt.subplots(figsize=(size, size))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=actions, yticklabels=actions, ax=ax,
               cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Action', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Action', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, f'confusion_matrix_{model_name.replace(" ", "_")}')


def plot_accuracy_by_player_total(dataset, model, predictions, split='test', model_name='Model'):
    """Plot accuracy breakdown by player total."""
    df = dataset.get_split(split)
    
    player_totals = df['initial_hand'].apply(get_hand_value)
    
    # Calculate accuracy by player total
    df_analysis = pd.DataFrame({
        'player_total': player_totals,
        'correct': (predictions == df['best_action_by_ev'])
    })
    
    accuracy_by_total = df_analysis.groupby('player_total')['correct'].agg(['mean', 'count'])
    accuracy_by_total = accuracy_by_total[accuracy_by_total['count'] >= 10]  # Filter low counts
    
    # Variable sizing based on number of unique totals
    num_totals = len(accuracy_by_total)
    width = max(10, num_totals * 0.6)
    
    fig, ax = plt.subplots(figsize=(width, 6))
    bars = ax.bar(accuracy_by_total.index, accuracy_by_total['mean'], 
                 color='skyblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Player Total', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - Accuracy by Player Total', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.axhline(y=accuracy_by_total['mean'].mean(), color='red', 
              linestyle='--', alpha=0.5, label='Overall Accuracy')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, f'accuracy_by_player_total_{model_name.replace(" ", "_")}')


def plot_simulation_results(results_per_shoe, model_name='Model'):
    """Plot running total and distribution of simulation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Running total
    cumulative = np.cumsum(results_per_shoe)
    ax1.plot(cumulative, linewidth=2, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Shoe Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Winnings (units)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name} - Cumulative Winnings', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Automatically adjust y-axis for cumulative plot
    y_margin = abs(max(cumulative) - min(cumulative)) * 0.1
    ax1.set_ylim([min(cumulative) - y_margin, max(cumulative) + y_margin])
    
    # Distribution histogram
    ax2.hist(results_per_shoe, bins=30, color='coral', alpha=0.7, edgecolor='black')
    mean_result = np.mean(results_per_shoe)
    ax2.axvline(x=mean_result, color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {mean_result:.2f}')
    ax2.set_xlabel('Winnings per Shoe (units)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name} - Result Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, f'simulation_results_{model_name.replace(" ", "_")}')


def plot_true_count_analysis(game_simulator, model, model_name='Model', num_games=100):
    """For card counting models, plot true count distribution during play."""
    if not hasattr(model, '_get_true_count'):
        return
    
    true_counts = []
    bet_sizes = []
    
    # Collect data from simulations
    for _ in range(num_games):
        shoe = game_simulator._create_shoe()
        cards_in_shoe = game_simulator.num_decks * 52
        cut_off = cards_in_shoe - int(game_simulator.deck_penetration * 52)
        
        while len(shoe) > cut_off and len(shoe) >= 4:
            remaining_cards = game_simulator.get_remaining_cards(shoe)
            tc = model._get_true_count(remaining_cards)
            bet = model.get_bet_size(remaining_cards)
            
            true_counts.append(tc)
            bet_sizes.append(bet)
            
            # Simulate some cards being dealt
            for _ in range(min(6, len(shoe))):
                shoe.pop()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # True count distribution
    ax1.hist(true_counts, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    mean_tc = np.mean(true_counts)
    ax1.axvline(x=mean_tc, color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {mean_tc:.2f}')
    ax1.set_xlabel('True Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name} - True Count Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Auto-adjust x-axis for true count
    tc_margin = (max(true_counts) - min(true_counts)) * 0.05
    ax1.set_xlim([min(true_counts) - tc_margin, max(true_counts) + tc_margin])
    
    # Bet size vs true count
    scatter = ax2.scatter(true_counts, bet_sizes, alpha=0.3, s=10, c=bet_sizes, 
                        cmap='viridis', edgecolors='none')
    ax2.set_xlabel('True Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Bet Size (units)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name} - Bet Sizing Strategy', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Bet Size')
    ax2.grid(alpha=0.3)
    
    # Auto-adjust axes for scatter plot
    ax2.set_xlim([min(true_counts) - tc_margin, max(true_counts) + tc_margin])
    bet_margin = (max(bet_sizes) - min(bet_sizes)) * 0.05
    ax2.set_ylim([min(bet_sizes) - bet_margin, max(bet_sizes) + bet_margin])
    
    plt.tight_layout()
    save_figure(fig, f'true_count_analysis_{model_name.replace(" ", "_")}')


def plot_model_comparison(results_dict):
    """Create a comparison chart of model accuracies and EVs."""
    # Variable width based on number of models
    num_models = len(results_dict)
    width = max(12, num_models * 4)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, 6))
    
    models = list(results_dict.keys())
    accuracies = [results_dict[m]['accuracy'] for m in models]
    evs = [results_dict[m]['ev'] for m in models]
    
    # Accuracy comparison
    colors = sns.color_palette("husl", len(models))
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=15)
    
    # EV comparison - auto-adjust y-axis
    bars2 = ax2.bar(models, evs, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Expected Value (units per shoe)', fontsize=12, fontweight='bold')
    ax2.set_title('Model EV Comparison', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Auto-adjust y-axis based on EV range
    min_ev, max_ev = min(evs), max(evs)
    ev_range = max_ev - min_ev
    margin = max(ev_range * 0.2, 1)  # At least 1 unit margin
    ax2.set_ylim([min_ev - margin, max_ev + margin])
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', 
                va='bottom' if height > 0 else 'top', fontweight='bold')
    
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    save_figure(fig, 'model_comparison')


def plot_action_distribution(dataset, split='test'):
    """Plot the distribution of optimal actions in the dataset."""
    df = dataset.get_split(split)
    action_counts = df['best_action_by_ev'].value_counts()
    
    # Variable width based on number of actions
    num_actions = len(action_counts)
    width = max(8, num_actions * 2)
    
    fig, ax = plt.subplots(figsize=(width, 6))
    colors = sns.color_palette("Set2", len(action_counts))
    bars = ax.bar(action_counts.index, action_counts.values, 
                 color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Action', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Optimal Action Distribution ({split.capitalize()} Set)', 
                fontsize=14, fontweight='bold')
    
    # Auto-adjust y-axis
    max_count = action_counts.max()
    ax.set_ylim([0, max_count * 1.15])
    
    # Add percentage labels
    total = action_counts.sum()
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}\n({percentage:.1f}%)', 
               ha='center', va='bottom', fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_figure(fig, f'action_distribution_{split}')


def plot_ev_by_action(dataset, split='test'):
    """Plot box plots of EV distributions for each action."""
    df = dataset.get_split(split)
    
    # Extract EVs for each action
    data_for_plot = []
    for _, row in df.iterrows():
        for action, ev in row['action_evs'].items():
            data_for_plot.append({'Action': action, 'EV': ev})
    
    df_plot = pd.DataFrame(data_for_plot)
    
    # Variable width based on number of unique actions
    num_actions = df_plot['Action'].nunique()
    width = max(8, num_actions * 2)
    
    fig, ax = plt.subplots(figsize=(width, 6))
    sns.boxplot(data=df_plot, x='Action', y='EV', ax=ax, palette='Set3')
    ax.set_xlabel('Action', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Value', fontsize=12, fontweight='bold')
    ax.set_title(f'EV Distribution by Action ({split.capitalize()} Set)', 
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Auto-adjust y-axis based on EV range
    min_ev, max_ev = df_plot['EV'].min(), df_plot['EV'].max()
    ev_range = max_ev - min_ev
    margin = ev_range * 0.1
    ax.set_ylim([min_ev - margin, max_ev + margin])
    
    plt.tight_layout()
    save_figure(fig, f'ev_by_action_{split}')


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

    print(f"\n--- Evaluation on {split_name} split ---")
    # print(f"Generating predictions for '{split_name}' split...")
    tqdm.pandas(desc=f"Generating predictions for '{split_name}' split...")
    predictions = df_split.progress_apply(get_prediction, axis=1)

    # Compare predictions to the optimal action determined by EV
    correct_predictions = (predictions == df_split['best_action_by_ev']).sum()
    total_predictions = len(df_split)
    
    if total_predictions == 0:
        print("No predictions were made.")
        return 0.0

    accuracy = correct_predictions / total_predictions
    print(f"Model accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    # Generate visualizations
    plot_confusion_matrix(df_split['best_action_by_ev'], predictions, 
                         model.__class__.__name__)
    plot_accuracy_by_player_total(dataset, model, predictions, split=split_name,
                                  model_name=model.__class__.__name__)
    
    return accuracy


def evaluate_model(model, dataset, game_simulator, num_simulations=100):
    # Calculate accuracy on the test split
    accuracy = calculate_accuracy(dataset, model, split_name='test')
    
    # Set the model in the game simulator
    game_simulator.set_model(model)
    
    # Run simulations to calculate average EV
    print(f"\nRunning {num_simulations} simulations...")
    simulation_results = []
    for i in tqdm(range(num_simulations), desc="Simulating Games"):
        np.random.seed(i)
        result = game_simulator.simulate_one_game()
        simulation_results.append(result)
    
    average_ev = np.mean(simulation_results)
    print(f"Average EV: {average_ev:.4f} units/shoe")
    
    # Generate simulation visualizations
    plot_simulation_results(simulation_results, model.__class__.__name__)
    
    # Generate card counting visualizations if applicable
    if hasattr(model, '_get_true_count'):
        print("Generating card counting analysis...")
        plot_true_count_analysis(game_simulator, model, model.__class__.__name__, 
                                num_games=100)
    
    return accuracy, average_ev


if __name__ == "__main__":
    # Load dataset
    dataset = ParquetDataset()
    
    # dataset = BlackjackDataset(csv_path='blackjack_simulator.csv', num_simulations=100, num_data_limit=1000) 
    # download 'blackjack_simulator.csv' from https://www.kaggle.com/datasets/dennisho/blackjack-hands/data
    # use a small num_data_limit for debugging.
    
    # Generate dataset visualizations
    print("\nGenerating dataset visualizations...")
    plot_action_distribution(dataset, split='train')
    plot_action_distribution(dataset, split='test')
    plot_ev_by_action(dataset, split='test')
    
    # Initialize game simulator, following the same settings as Kaggle dataset
    game_simulator = GameSimulator(num_decks=8, deck_penetration=6.5)

    # Store results for comparison
    results_dict = {}

    NUM_TEST_SIMS = 10000
    # Naive Strategy model
    model = NaiveStrategy()
    print("\n\n------------------------------")
    print("Evaluating Naive Strategy model...")
    accuracy, average_ev = evaluate_model(model, dataset, game_simulator, num_simulations=NUM_TEST_SIMS)
    results_dict['Naive Strategy'] = {'accuracy': accuracy, 'ev': average_ev}
    
    
    # # Basic Strategy model
    model = BasicStrategyModel()
    print("\n\n------------------------------")
    print("Evaluating Basic Strategy model...")
    accuracy, average_ev = evaluate_model(model, dataset, game_simulator, num_simulations=NUM_TEST_SIMS)
    results_dict['Basic Strategy'] = {'accuracy': accuracy, 'ev': average_ev}
    
    
    # # Hi-Lo model
    hilo_model = HILO(num_decks=8, bet_spread=20)
    print("\n\n------------------------------")
    print("Evaluating Hi-Lo model...")
    accuracy, average_ev = evaluate_model(hilo_model, dataset, game_simulator, num_simulations=NUM_TEST_SIMS)
    results_dict['Hi-Lo Strategy'] = {'accuracy': accuracy, 'ev': average_ev}
    
    # RL model
    # rl_model = RLModel(num_decks=8, bet_spread=20)
    # rl_model.load_model('blackjack_rl_model_bet_size_only_shoe_2300000.pkl')
    # rl_model.baseline_model = hilo_model
    # accuracy, average_ev = evaluate_model(rl_model, dataset, game_simulator, num_simulations=NUM_TEST_SIMS)
    # results_dict['RL Strategy'] = {'accuracy': accuracy, 'ev': average_ev}
            
    # Generate comparison plot
    print("\n\nGenerating model comparison plot...")
    plot_model_comparison(results_dict)
    
    print("\n" + "="*60)
    print(f"ALL FIGURES SAVED TO: {OUTPUT_DIR}/")
    print("="*60)
