# main.py
#!/usr/bin/env python3
from lemonade_simulator import LemonadeSimulator, BayesianNetwork
# or whatever your main classes are

def main():
    # Initialize and run the game
    game = LemonadeSimulator()
    game.run()

if __name__ == "__main__":
    main()