# train_rl_model_dqn.py

from model import RLModelDQN as RLModel
import numpy as np
from tqdm import tqdm
from helper import get_hand_value
import matplotlib.pyplot as plt
from argparse import ArgumentParser


class GameSimulator:
    """
    Optimized blackjack simulator.
    Trains RL model by calling remember_* and train_* methods.
    """

    def __init__(self, num_decks=8, deck_penetration=6.5, max_bet_size=20):
        self.card_values = [2,3,4,5,6,7,8,9,10,21,22,23,11]
        self.num_decks = num_decks
        self.deck_penetration = deck_penetration
        self.model = None
        self.max_bet_size = max_bet_size

        self.card_keys = [2,3,4,5,6,7,8,9,10,11]

    def _create_shoe(self):
        shoe = self.card_values * 4 * self.num_decks
        np.random.shuffle(shoe)
        return shoe

    def set_model(self, model):
        self.model = model

    def get_initial_card_counts(self):
        remaining = {v: 4*self.num_decks for v in self.card_keys}
        remaining[10] = 16*self.num_decks
        return remaining

    def deal_card(self, shoe, remaining):
        card = shoe.pop()
        key = 10 if card in (21,22,23) else card
        remaining[key] -= 1
        return card

    def transform(self, h):
        return [c if c <= 20 else 10 for c in h]

    def simulate_one_game(self, training_mode=False):

        shoe = self._create_shoe()
        remaining = self.get_initial_card_counts()
        total_profit = 0

        cut = self.num_decks*52 - int(self.deck_penetration*52)

        while len(shoe) > cut:

            if len(shoe) < 4:
                break

            bet_state = self.model._get_bet_state(remaining) if training_mode else None
            bet_size = min(self.model.get_bet_size(remaining), self.max_bet_size)

            round_profit = 0
            # deal
            player = [self.deal_card(shoe, remaining), self.deal_card(shoe, remaining)]
            dealer = [self.deal_card(shoe, remaining), self.deal_card(shoe, remaining)]
            dealer_up = dealer[0] if dealer[0] <= 20 else 10

            # Blackjack checks
            p_val = get_hand_value(self.transform(player))
            d_val = get_hand_value(self.transform(dealer))

            p_bj = (p_val == 21 and len(player) == 2)
            d_bj = (d_val == 21 and len(dealer) == 2)

            if p_bj or d_bj:
                if p_bj and d_bj:
                    profit = 0.0
                elif p_bj:
                    profit = 1.5 * bet_size
                else:
                    profit = -1.0 * bet_size

                total_profit += profit
                round_profit += profit

                if training_mode:
                    self.model.remember_bet(bet_state, bet_size, profit)

                continue

            # PLAY
            hands = [{"hand": player, "bet": bet_size, "final": None}]

            h_idx = 0
            while h_idx < len(hands):
                phand = hands[h_idx]["hand"]
                pbet = hands[h_idx]["bet"]

                while True:
                    if not shoe:
                        break

                    play_state = self.model._get_play_state(phand, dealer_up, remaining) if training_mode else None
                    action = self.model.predict(phand, dealer_up, remaining)

                    is_pair = (len(phand) == 2 and phand[0] == phand[1])
                    can_split = (len(hands) < 4)

                    # SPLIT
                    if action == 'P':
                        if is_pair and can_split and len(shoe) >= 2:
                            c1, c2 = phand
                            h1 = [c1, self.deal_card(shoe, remaining)]
                            h2 = [c2, self.deal_card(shoe, remaining)]

                            hands[h_idx] = {"hand": h1, "bet": pbet, "final": None}
                            hands.insert(h_idx+1, {"hand": h2, "bet": pbet, "final": None})

                            if training_mode:
                                next_state = self.model._get_play_state(h1, dealer_up, remaining)
                                self.model.remember_play(play_state, action, 0.0, next_state, False)
                            phand = hands[h_idx]["hand"]
                            continue
                        else:
                            action = 'S'  # invalid split → stand

                    if action == 'S':
                        if training_mode:
                            hands[h_idx]["final"] = (play_state, action)
                        break

                    if action == 'R':
                        if len(phand) == 2:
                            reward = -0.5
                            profit = reward * pbet
                            total_profit += profit
                            round_profit += profit
                            if training_mode:
                                self.model.remember_play(play_state, action, reward, None, True)
                            phand.clear()
                            break
                        else:
                            action = 'H'

                    if action == 'D':
                        if len(phand) == 2 and len(shoe) >= 1:
                            new_bet = 2*pbet
                            hands[h_idx]["bet"] = new_bet
                            phand.append(self.deal_card(shoe, remaining))

                            val = get_hand_value(self.transform(phand))
                            if val > 21:
                                reward = -2.0
                                profit = -new_bet
                                total_profit += profit
                                round_profit += profit
                                if training_mode:
                                    self.model.remember_play(play_state, action, reward, None, True)
                            else:
                                if training_mode:
                                    hands[h_idx]["final"] = (play_state, action)
                            break
                        else:
                            action = 'H'

                    if action == 'H':
                        phand.append(self.deal_card(shoe, remaining))
                        val = get_hand_value(self.transform(phand))
                        if val > 21:
                            reward = -1.0
                            profit = -pbet
                            total_profit += profit
                            round_profit += profit
                            if training_mode:
                                self.model.remember_play(play_state, action, reward, None, True)
                            break
                        else:
                            if training_mode:
                                next_state = self.model._get_play_state(phand, dealer_up, remaining)
                                self.model.remember_play(play_state, action, 0.0, next_state, False)

                h_idx += 1

            # DEALER
            d_val = get_hand_value(self.transform(dealer))
            while d_val < 17 or (d_val == 17 and 11 in self.transform(dealer)):
                dealer.append(self.deal_card(shoe, remaining))
                d_val = get_hand_value(self.transform(dealer))

            dealer_bust = d_val > 21

            # RESOLUTION
            for hd in hands:
                ph = hd["hand"]
                if not ph:
                    continue
                pv = get_hand_value(self.transform(ph))
                if pv > 21:
                    continue

                bet = hd["bet"]
                profit = 0.0
                if dealer_bust or pv > d_val:
                    profit = bet
                elif pv < d_val:
                    profit = -bet

                total_profit += profit
                round_profit += profit

                if training_mode and hd["final"] is not None:
                    (pstate, pact) = hd["final"]
                    rew = profit / bet_size
                    self.model.remember_play(pstate, pact, rew, None, True)

            if training_mode:
                self.model.remember_bet(bet_state, bet_size, round_profit)

        if training_mode:
            self.model.train_play_model()
            self.model.train_bet_model()
            self.model.decay_epsilon()

        return total_profit

    def run_multiple_simulations(self, n=1000, training_mode=False):
        results = []
        for i in tqdm(range(n)):
            np.random.seed(i if not training_mode else None)
            results.append(self.simulate_one_game(training_mode=training_mode))
        return np.mean(results)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--train_shoes", type=int, default=2000000)
    parser.add_argument("--eval_shoes", type=int, default=5000)
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="rl_model_final.pth")
    parser.add_argument("--plot_path", type=str, default="training_curve.png")
    args = parser.parse_args()

    model = RLModel(
        num_decks=8,
        bet_spread=20,
        learning_rate=1e-5,
        bet_learning_rate=1e-5,
        batch_size=256,
        target_update_freq=5000,
        epsilon_start=0.05,
        epsilon_decay=0.9999995
    )

    if args.load_path:
        model.load_model(args.load_path)

    model.set_training_stage(args.stage)

    # if args.stage == 2:
    #     model.attach_supervised_play_snapshot()
    #     print("[Stage 2] Distillation enabled.")

    sim = GameSimulator()
    sim.set_model(model)

    print("=== Training ===")
    EVs = []
    Xs = []

    for i in tqdm(range(args.train_shoes)):
        sim.simulate_one_game(training_mode=True)

        if (i+1) % 20000 == 0:
            old_eps = model.epsilon
            model.epsilon = 0.0
            ev = sim.run_multiple_simulations(args.eval_shoes)
            model.epsilon = old_eps

            EVs.append(ev)
            Xs.append(i+1)
            print(f"At {i+1} shoes: EV={ev:.4f}")

            plt.plot(Xs, EVs)
            plt.savefig(args.plot_path)
            plt.clf()
            
            model.save_model(f"{args.save_path.split('.pth')[0]}_shoe{i+1}.pth")

    print("Saving model...")
    model.save_model(args.save_path)

    print("=== Final Evaluation ===")
    model.epsilon = 0.0
    final_ev = sim.run_multiple_simulations(args.eval_shoes)
    print(f"\nFINAL EV = {final_ev:.4f}")
