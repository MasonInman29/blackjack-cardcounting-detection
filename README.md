## Card Counting Detection (COM S 573)

### Features
- Baselines: `BasicStrategyModel`, `HILO`, and `NaiveStrategy`
- Monte Carlo EVs
- Game simulator
- Evaluation
- MLP and SVM
- House-edge modeling


## Installation

```bash
pip install -r requirements.txt
```


## Data

- Kaggle “50 Million Blackjack Hands” (`https://www.kaggle.com/datasets/dennisho/blackjack-hands`).

## Repository structure (key modules)

- `eval_model.py`
  - Evaluation utilities and visualizations.
  - Saves figures under `figures/<timestamp>/`.

- `dataset/dataset.py`
  - `BlackjackDataset`

- `dataset/game_simulator.py`

- `model/hi_lo.py`

- `model/naive_strategy.py`

- `model/Hit_By_Hit_Transformer.py`

- `model/NeuralNet_BlackJack.py`

- `model/SVM_BlackJack.py`

- `model/house_edge_models/NN_HouseEdge.py`

- `helper.py`


1) Evaluate baselines and generate figures

```bash
python eval_model.py
```

2) Train and evaluate neural net

```bash
python model/NeuralNet_BlackJack.py
```

Notes:
- Uses `CSVDataset()` and takes `blackjack_labeled_simulations.csv`.
- Trains an `MLPClassifier`, then evaluates using  `GameSimulator`.

3) Train and evaluate SVM

```bash
python model/SVM_BlackJack.py
```

Notes:
- Uses `CSVDataset()`

4) Dataset processing

```bash
python dataset/dataset.py
```

Notes:
- Constructs `BlackjackDataset` from `blackjack_simulator.csv`, computing EVs, and creating splits.




- Actions modeled: Hit (H), Stand (S), Double (D), Surrender (R). Splits and Insurance are not simulated.
- `HILO` deviations reflect a subset of common index plays; scope is adjustable.
- Simulator uses standard dealer rules (hit soft-17 handled) and simple bet sizing APIs.

## Example output:
### LR
<img width="3203" height="1763" alt="accuracy_by_player_total_LogisticRegression_BlackJack" src="https://github.com/user-attachments/assets/54f65af5-4036-4f62-b3f8-9ea98f15f8ad" />
<img width="1809" height="1763" alt="confusion_matrix_LogisticRegression_BlackJack" src="https://github.com/user-attachments/assets/7bf93032-9da9-4c77-86c8-46ba1f168fb5" />
<img width="4162" height="1763" alt="simulation_results_LogisticRegression_BlackJack" src="https://github.com/user-attachments/assets/8a910db0-a8e5-4473-9cbb-5b9dea2e3953" />



- Data: Kaggle “50 Million Blackjack Hands” by `dennisho` — see dataset page for license/terms.

## TODO

- Add descriptions for functions
- Add other game mechanics (insurance)
- Add UI
- Automatic data download and usage
- Models
  - Logistic regression
  - XG boost
  - Reinforcement learning
- Labels
  - Outcome Labels
  - Advantage opportunity label
- Splits and insurance
- Add precision and recall
- Sequence classifer
- Counter probability
- Running true counts that persist for the session
- 
