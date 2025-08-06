
#!/usr/bin/env python3
# main.py
# Entry point for Lemonade Stand Simulator
from game.simulator import LemonadeStandSimulator

def main():
    print("🍋 Lemonade Stand Simulator 🍋")
    sim = LemonadeStandSimulator()
    try:
        days = int(input("How many days to simulate? (3-14): "))
        sim.play_game(num_days=max(3, min(days, 14)), show_analysis=True)
    except KeyboardInterrupt:
        print("\n👋 Game interrupted by user.")

if __name__ == "__main__":
    main()