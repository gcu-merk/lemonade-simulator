
#!/usr/bin/env python3
# main.py
# Entry point for Lemonade Stand Simulator
from game.simulator import LemonadeStandSimulator

def main():
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

if __name__ == "__main__":
    main()