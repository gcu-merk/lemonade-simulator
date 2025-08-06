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
