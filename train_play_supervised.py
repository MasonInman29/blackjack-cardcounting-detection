import torch
from torch.utils.data import DataLoader
from model.rl_model_dqn import RLModelDQN
from dataset import PlaySupervisedDataset, ParquetDataset, GameSimulator
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-4
NUM_DECKS = 8

dataset = ParquetDataset()
train_df = dataset.get_split("train")
test_df = dataset.get_split("test")

train_ds = PlaySupervisedDataset(train_df, NUM_DECKS)
test_ds = PlaySupervisedDataset(test_df, NUM_DECKS)

from collections import Counter

# Compute class distribution
# label_counts = Counter()
# for _, y in train_ds:
#     label_counts[int(y)] += 1

# print("Class counts:", label_counts)


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
train_num_batches = len(train_loader)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Load RL model, but use STAGE 1
model = RLModelDQN(num_decks=NUM_DECKS)
model.set_training_stage(1)

play_net = model.play_policy_net
optimizer = torch.optim.Adam(play_net.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


device = model.device
play_net.to(device)

# num_classes = 5
# total = sum(label_counts.values())

# class_weights = torch.tensor(
#     [total / (num_classes * label_counts[i]) for i in range(num_classes)],
#     dtype=torch.float32,
#     device=device
# )

# print("Class weights:", class_weights)
# loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
log_softmax = torch.nn.LogSoftmax(dim=1)

save_path = "play_network_supervised_v7.pth"

def evaluate(loader):
    correct = 0
    total = 0
    play_net.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = play_net(x)
            predicted = torch.argmax(preds, dim=1)
            gt = torch.argmax(y, dim=1)
            correct += (predicted == gt).sum().item()
            total += y.size(0)
    return correct / total

game_simulator = GameSimulator(num_decks=8, deck_penetration=6.5)
num_simulations=10000
model.epsilon = 0.0

best_ev = -10000.0
print("Starting supervised training...")
for epoch in range(EPOCHS):
    play_net.train()
    total_loss = 0

    for batch_idx, (x, y) in tqdm(enumerate(train_loader), total=train_num_batches):
        x, y = x.to(device), y.to(device)

        preds = play_net(x)
        
        log_probs = log_softmax(preds)
        loss = loss_fn(log_probs, y)
        # loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(play_net.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        # print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss={loss.item():.4f}")
    avg_loss = total_loss / train_num_batches
    acc = evaluate(test_loader)
    game_simulator.set_model(model)
    simulation_results = []
    for i in tqdm(range(num_simulations), desc="Simulating Games"):
        np.random.seed(i)
        result = game_simulator.simulate_one_game()
        simulation_results.append(result)
    average_ev = np.mean(simulation_results)
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss={avg_loss:.4f}, Test Accuracy={acc:.4f}, Average EV={average_ev:.4f}")
    if average_ev > best_ev:
        best_ev = average_ev
        model.save_model(f"{save_path.split('.pth')[0]}_best.pth")
    scheduler.step()

print("Saving pretrained supervised model...")
model.save_model(f"{save_path.split('.pth')[0]}_last.pth")
print("DONE.")
