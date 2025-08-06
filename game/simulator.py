"""
Lemonade Stand Simulator - High-Level Orchestrator
Wraps GameSimulator (from game_logic.py) to provide AI analysis,
visualization, and extended features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from game.game_logic import GameSimulator, GameInterface
from config.constants import STARTING_CAPITAL, STARTING_REPUTATION


class LemonadeStandSimulator:
    """
    High-level simulator that delegates game mechanics to GameSimulator.
    Adds analytics, visualizations, and user-friendly orchestration.
    """

    def __init__(self):
        # Core engine from game_logic
        self.core = GameSimulator()
        self.interface = GameInterface(self.core)
        self.game_active = True
        self.winner = None

    # --- Delegate properties ---
    @property
    def day(self):
        return self.core.day

    @property
    def players(self):
        return self.core.player_manager.players

    @property
    def engine(self):
        return self.core.engine

    @property
    def meu_analyzer(self):
        return self.core.meu_analyzer

    # --- Core Game Flow ---
    def initialize_game(self):
        """Initialize game using core engine"""
        self.core.day = 1
        self.core.engine.generate_weather()
        self.game_active = True
        print("\n🍋 LEMONADE STAND SIMULATOR 🍋")
        print("Human vs AI with Bayesian MEU Analysis")
        print(f"Starting Capital: ${STARTING_CAPITAL:.2f}")
        print(f"Starting Reputation: {STARTING_REPUTATION:.1f}/10")
        print(f"Weather Forecast: {self.engine.weather_forecast.title()}")

    def play_game(self, num_days: int = 7, show_analysis: bool = True):
        """Main game loop"""
        self.initialize_game()

        for _ in range(num_days):
            if not self.game_active:
                break

            print(f"\n{'=' * 20} DAY {self.day} {'=' * 20}")
            self.interface.display_game_state()

            # --- Human Turn ---
            print(f"\n{'=' * 15} HUMAN TURN {'=' * 15}")
            try:
                decision = self.interface.get_human_decision()
                result = self.core.simulate_day('human', *decision)
                if not result.success:
                    print(f"❌ Human decision failed: {result.message}")
            except Exception as e:
                print(f"⚠️ Error in human turn: {e}")

            # --- AI Turn ---
            print(f"\n{'=' * 17} AI TURN {'=' * 17}")
            try:
                ai_decision = self.core.ai_make_decision()
                print(f"🤖 AI chose: {ai_decision}")
                self.core.simulate_day('ai', *ai_decision)
            except Exception as e:
                print(f"⚠️ Error in AI turn: {e}")

            # Show daily results
            self.interface.display_daily_results()

            # Optional AI Analysis Visualization
            if show_analysis and self.core.meu_analyzer:
                try:
                    self.visualize_ai_analysis()
                except Exception as e:
                    print(f"⚠️ Analysis visualization failed: {e}")

            # Advance to next day
            self.core.advance_day()
            over, msg = self.core.check_game_over()
            if over:
                print(msg)
                self.game_active = False
                break

            input("\nPress Enter to continue...")

        self._display_final_results()

    # --- Visualization Enhancements ---
    def visualize_ai_analysis(self):
        """Visualize AI decision analysis if available"""
        ai_state = self.players['ai']
        if not ai_state.decision_logs:
            print("No AI decision analysis available.")
            return
        last_log = ai_state.decision_logs[-1]
        if 'all_analyses' in last_log['decision'].__dict__:
            analyses = last_log['decision'].all_analyses
            self.meu_analyzer.visualize_decision_space(analyses)

    def visualize_performance(self):
        """Show performance charts (profits, money, reputation, locations)"""
        try:
            human = self.players['human']
            ai = self.players['ai']
            if not human.daily_logs or not ai.daily_logs:
                print("Not enough data to plot.")
                return

            df_h = pd.DataFrame(human.daily_logs)
            df_a = pd.DataFrame(ai.daily_logs)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Profit
            axes[0, 0].plot(df_h['day'], df_h['profit'], label="Human", marker='o')
            axes[0, 0].plot(df_a['day'], df_a['profit'], label="AI", marker='s')
            axes[0, 0].set_title("Daily Profit")
            axes[0, 0].legend()

            # Money
            human_money = [STARTING_CAPITAL] + [STARTING_CAPITAL + df_h['profit'][:i+1].sum() for i in range(len(df_h))]
            ai_money = [STARTING_CAPITAL] + [STARTING_CAPITAL + df_a['profit'][:i+1].sum() for i in range(len(df_a))]
            axes[0, 1].plot(human_money, label="Human")
            axes[0, 1].plot(ai_money, label="AI")
            axes[0, 1].set_title("Money Over Time")
            axes[0, 1].legend()

            # Reputation
            human_rep = [STARTING_REPUTATION] + [STARTING_REPUTATION + sum(df_h['reputation_change'][:i+1]) for i in range(len(df_h))]
            ai_rep = [STARTING_REPUTATION] + [STARTING_REPUTATION + sum(df_a['reputation_change'][:i+1]) for i in range(len(df_a))]
            axes[1, 0].plot(human_rep, label="Human")
            axes[1, 0].plot(ai_rep, label="AI")
            axes[1, 0].set_title("Reputation Over Time")
            axes[1, 0].legend()

            # Locations
            sns.countplot(x=df_a['location'], ax=axes[1, 1], color='red', label="AI")
            sns.countplot(x=df_h['location'], ax=axes[1, 1], color='blue', label="Human")
            axes[1, 1].set_title("Location Choices")
            axes[1, 1].legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"⚠️ Failed to visualize performance: {e}")

    # --- Final Results ---
    def _display_final_results(self):
        """Show final game summary using GameInterface logic"""
        self.interface.display_final_results()


# CLI Entry Point
if __name__ == "__main__":
    print("🍋 Lemonade Stand Simulator 🍋")
    sim = LemonadeStandSimulator()
    try:
        while True:
            raw_days = input("How many days to simulate? (3-14): ").strip()
            if not raw_days:
                print("Input cannot be empty. Please enter a number between 3 and 14.")
                continue
            if not raw_days.isdigit():
                print("Invalid input! Please enter a whole number between 3 and 14.")
                continue
            days = int(raw_days)
            if 3 <= days <= 14:
                break
            print("Number of days must be between 3 and 14.")
        sim.play_game(num_days=days, show_analysis=True)
    except KeyboardInterrupt:
        print("\n👋 Game interrupted by user.")