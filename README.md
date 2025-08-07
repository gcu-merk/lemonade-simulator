# Lemonade Stand Simulator

A feature-rich, AI-powered lemonade stand business simulation game. Compete against an AI, make strategic decisions, and see how your business performs under changing weather, local events, and competition!

## Features
- Dynamic weather and local event system
- Competition and reputation mechanics
- AI opponent using Maximum Expected Utility (MEU) analysis
- Automated information market (all info auto-purchased)
- Detailed daily and final results display
- Modern Python codebase, modular structure

## Getting Started

### Prerequisites
- Python 3.8+
- Recommended: Create and activate a virtual environment

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/gcu-merk/lemonade-simulator.git
   cd lemonade-simulator
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Game
Run the main script:
```sh
python main.py
```


### Project Structure
- `main.py` — Entry point for the game
- `lemonade_simulator.py` — (Legacy/compatibility)
- `config/` — Game constants and settings
- `game/` — Core game logic, simulation, and AI
- `models/` — Data classes and Bayesian network
- `tests/` — Unit tests

### Code Flowchart (High-Level)

```
┌────────────────────────────┐
│        main.py            │
└────────────┬──────────────┘
             │
             ▼
   ┌──────────────────────────────┐
   │   GameSimulator (game_logic) │
   └────────────┬─────────────────┘
                │
   ┌────────────┴─────────────┐
   │                          │
   ▼                          ▼
GameEngine              PlayerManager
   │                          │
   ▼                          ▼
Weather, Customers,      PlayerState,
Quality, Sales, etc.     Purchases, etc.
   │                          │
   └────────────┬─────────────┘
                │
                ▼
        MEUAnalyzer (AI logic)
                │
                ▼
        BayesianNetwork (models)
                │
                ▼
         GameInterface (UI)
                │
                ▼
           User/AI Input
                │
                ▼
           Results Output
```


This flowchart shows the main modules and how data flows between them during a game turn.

#### The Role of the Decision Network (Bayesian Network)

The `BayesianNetwork` is the core of the AI's decision-making process. It models the uncertainties in the game—such as weather, customer behavior, and sales outcomes—using probabilistic reasoning. When the AI (via `MEUAnalyzer`) evaluates possible decisions, it uses the Bayesian network to estimate the probability of different outcomes for each choice. This allows the AI to calculate the Maximum Expected Utility (MEU) for every possible action, balancing risk and reward just like a real-world business would. The decision network is what enables the AI to make smart, adaptive choices based on incomplete information, simulating realistic business strategy.

## How to Play
- Each day, review the weather forecast and local events.
- Make business decisions: choose location, recipe, price, and quantity.
- All information is auto-purchased for both players.
- Compete against the AI to maximize profit, reputation, and consistency.
- The game ends when either player is bankrupt or after a set number of days.

## Development & Contributing
- Contributions welcome! Please open issues or pull requests.
- Code is formatted with [PEP8](https://www.python.org/dev/peps/pep-0008/).

## License
MIT License

---

*Created by gcu-merk*
