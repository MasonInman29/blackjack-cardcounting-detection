import dearpygui.dearpygui as dpg
import sys
import os
import numpy as np
import threading
import time
from model.rl_model import RLModel
from model.rl_model_dqn import RLModelDQN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load(name, subdir):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_base, subdir, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

GameSimulator = _load("game_simulator", "dataset").GameSimulator
_hi_lo = _load("hi_lo", "model")
HILO = _hi_lo.HILO

class RLComparisonPlotter:
    def __init__(self):
        self.num_decks = 8
        self.start_bankroll = 100.0
        
        # Load Hi-Lo
        hilo = HILO(num_decks=self.num_decks, bet_spread=20)
        
        # Load RL model
        rl_model = RLModel(num_decks=self.num_decks, bet_spread=20)
        rl_model.load_model(os.path.join(_base, "models", "blackjack_rl_model_bet_size_only_shoe_2300000.pkl"))
        rl_model.baseline_model = hilo
        
        # Load DQN model
        dqn_model = RLModelDQN()
        dqn_model.load_model(os.path.join(_base, "models", "play_network_supervised_v7_best.pth"))
        dqn_model.epsilon = 0.0
        dqn_model.original_rl_model = rl_model
        
        self.models = {
            "Hi-Lo": hilo,
            "RL Bet Sizing": rl_model,
            "DQN Play + RL Bet": dqn_model,
        }
        self.colors = {
            "Hi-Lo": (100, 255, 150),
            "RL Bet Sizing": (255, 150, 100),
            "DQN Play + RL Bet": (100, 200, 255),
        }
        
        self.bankrolls = {m: [self.start_bankroll] for m in self.models}
        self.shoes = {m: [0] for m in self.models}
        self.running = False
        self.speed = 50
        self.simulator = GameSimulator(num_decks=self.num_decks, deck_penetration=6.5)

    def _simulate_batch(self, name, count):
        self.simulator.set_model(self.models[name])
        bankroll = self.bankrolls[name][-1]
        shoe_num = self.shoes[name][-1]
        
        for _ in range(count):
            bankroll += self.simulator.simulate_one_game()
            shoe_num += 1
            self.bankrolls[name].append(bankroll)
            self.shoes[name].append(shoe_num)

    def _sim_loop(self):
        while self.running:
            for name in self.models:
                if not self.running:
                    break
                self._simulate_batch(name, self.speed)
            time.sleep(0.05)

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self._sim_loop, daemon=True).start()
            dpg.configure_item("start_button", enabled=False)
            dpg.configure_item("stop_button", enabled=True)

    def stop(self):
        self.running = False
        dpg.set_value("status", "Stopped")
        dpg.configure_item("start_button", enabled=True)
        dpg.configure_item("stop_button", enabled=False)

    def reset(self):
        self.running = False
        self.bankrolls = {m: [self.start_bankroll] for m in self.models}
        self.shoes = {m: [0] for m in self.models}
        self._update()
        dpg.set_value("status", "Reset")
        dpg.configure_item("start_button", enabled=True)
        dpg.configure_item("stop_button", enabled=False)

    def _update(self):
        for name in self.models:
            x, y = self.shoes[name], self.bankrolls[name]
            if len(x) > 2000:
                step = len(x) // 2000
                x, y = x[::step], y[::step]
            dpg.set_value(f"series_{name}", [list(x), list(y)])
            
            bankroll = self.bankrolls[name][-1]
            shoe_count = self.shoes[name][-1]
            net = bankroll - self.start_bankroll
            ev = net / shoe_count if shoe_count > 0 else 0
            dpg.set_value(f"stat_{name}", f"${bankroll:.2f} ({net:+.2f}) | {shoe_count} shoes | EV: {ev:+.3f}/shoe")

        if any(len(self.shoes[m]) > 1 for m in self.models):
            max_shoes = max(self.shoes[m][-1] for m in self.models)
            all_br = [b for m in self.models for b in self.bankrolls[m]]
            margin = max(10, (max(all_br) - min(all_br)) * 0.1)
            dpg.set_axis_limits("x_axis", 0, max(100, max_shoes * 1.05))
            dpg.set_axis_limits("y_axis", min(all_br) - margin, max(all_br) + margin)

    def run(self):
        dpg.create_context()
        dpg.set_global_font_scale(3.0)

        with dpg.window(tag="main"):
            dpg.add_text("Hi-Lo vs RL Models")
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", tag="start_button", callback=lambda: self.start(), width=120, height=40)
                dpg.add_button(label="Stop", tag="stop_button", callback=lambda: self.stop(), width=120, height=40, enabled=False)
                dpg.add_button(label="Reset", callback=lambda: self.reset(), width=120, height=40)
                dpg.add_spacer(width=20)
                dpg.add_text("Speed:")
                dpg.add_slider_int(tag="speed", default_value=50, min_value=1, max_value=200, width=200,
                                  callback=lambda s, a: setattr(self, 'speed', a))

            with dpg.group(horizontal=True):
                dpg.add_text("Status:")
                dpg.add_text("Ready", tag="status")

            with dpg.plot(height=500, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Shoes", tag="x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Bankroll ($)", tag="y_axis")
                dpg.set_axis_limits("x_axis", 0, 100)
                dpg.set_axis_limits("y_axis", 50, 150)

                dpg.add_line_series([0, 10000], [100, 100], label="$100 baseline", parent="y_axis", tag="baseline")
                
                for name in self.models:
                    dpg.add_line_series([0], [100], label=name, parent="y_axis", tag=f"series_{name}")

            dpg.add_text("Statistics:")
            for name in self.models:
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{name}:", color=self.colors[name])
                    dpg.add_text("$100.00", tag=f"stat_{name}")

        dpg.create_viewport(title="RL Model Comparison", width=1400, height=900)
        dpg.setup_dearpygui()
        dpg.set_primary_window("main", True)
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            if self.running:
                self._update()
            dpg.render_dearpygui_frame()

        self.running = False
        dpg.destroy_context()


if __name__ == "__main__":
    RLComparisonPlotter().run()


