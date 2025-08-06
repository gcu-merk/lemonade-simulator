"""
Game Logic for Lemonade Stand Simulator
Core game mechanics, simulation, and business logic
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any, Optional
from models.data_classes import PlayerState, MEUDecision, GameResult
from models.bayesian_network import BayesianNetwork
from game.meu_analyzer import MEUAnalyzer
from config.constants import (
    STARTING_CAPITAL,
    STARTING_REPUTATION,
    TRAFFIC_BASE_CUSTOMERS,
    PRICE_MIN,
    PRICE_MAX,
    PRICE_STEP,
    LOCATIONS,
    RECIPES,
    WEATHER_CONDITIONS,
    INFO_MARKET,
    DEFAULT_COMPETITION_STATS as COMPETITION_STATS
)

class GameEngine:
    """Core game engine handling simulation mechanics"""

    def __init__(self):
        # Weather system
        self.current_weather = None
        self.weather_forecast = None
        
        # Competition tracking
        self.competition_stats = COMPETITION_STATS.copy()
        
    def generate_weather(self) -> Tuple[str, str]:
        """Generate actual weather and forecast with some uncertainty"""
        # Generate actual weather based on realistic probabilities
        weather_probs = [0.4, 0.3, 0.2, 0.1]  # sunny, cloudy, rainy, hot
        self.current_weather = np.random.choice(WEATHER_CONDITIONS, p=weather_probs)
        
        # Generate forecast with accuracy based on realism
        forecast_accuracy = 0.7
        if np.random.random() < forecast_accuracy:
            self.weather_forecast = self.current_weather
        else:
            # Incorrect forecast - choose from other conditions
            other_weathers = [w for w in WEATHER_CONDITIONS if w != self.current_weather]
            self.weather_forecast = np.random.choice(other_weathers)
            
        return self.current_weather, self.weather_forecast

    def calculate_customers(self, location: str, weather: str, quality: float, 
                          price: float, reputation: float, event_multiplier: float = 1.0) -> int:
        """Calculate number of potential customers based on multiple factors, including event multiplier"""
        # Base traffic from location
        base_traffic = LOCATIONS[location]['base_traffic']
        
        # Weather effects on foot traffic
        weather_multipliers = {
            'sunny': 1.2, 
            'hot': 1.4, 
            'cloudy': 0.9, 
            'rainy': 0.4
        }
        weather_multiplier = weather_multipliers.get(weather, 1.0)
        
        # Location-specific weather bonus
        location_bonus = LOCATIONS[location].get('weather_bonus', 1.0)
        if weather in ['sunny', 'hot']:
            weather_multiplier *= location_bonus
            
        # Quality effect on customer attraction
        quality_multiplier = 0.5 + (quality / 10.0)
        
        # Price sensitivity - higher prices reduce customer interest
        price_multiplier = max(0.1, 2.0 - (price * 0.8))
        
        # Reputation effect
        reputation_multiplier = 0.7 + (reputation / 10.0) * 0.6
        
        # Competition effect - more competitors reduce customers
        competition = self.competition_stats.get(location, {})
        num_competitors = competition.get('num_stands', 1)
        competition_multiplier = max(0.5, 1.2 - (num_competitors * 0.1))
        
        # Calculate base customers, apply event multiplier
        customers = int(base_traffic * weather_multiplier * quality_multiplier * 
                       price_multiplier * reputation_multiplier * competition_multiplier * event_multiplier)
        
        # Add realistic randomness (±15%)
        customers = max(0, int(np.random.normal(customers, customers * 0.15)))
        
        return customers

    def calculate_quality(self, recipe: str, reputation: float) -> float:
        """Calculate effective product quality"""
        recipe_data = RECIPES[recipe]
        base_quality = recipe_data['base_quality']
        appeal = recipe_data['appeal']
        
        # Reputation bonus/penalty
        reputation_bonus = (reputation - 5.0) * 0.4
        
        # Calculate final quality
        quality = max(1.0, (base_quality * appeal) + reputation_bonus)
        
        return min(10.0, quality)  # Cap at 10

    def calculate_sales(self, customers: int, quantity: int, quality: float, 
                       price: float) -> int:
        """Calculate actual sales based on customers and constraints"""
        # Not all customers will buy - depends on quality and price
        quality_factor = min(1.0, quality / 8.0)  # Quality 8+ = 100% conversion
        price_factor = max(0.3, 1.5 - (price * 0.3))  # Higher price = lower conversion
        
        conversion_rate = quality_factor * price_factor * 0.8  # Base 80% max conversion
        potential_sales = int(customers * conversion_rate)
        
        # Sales limited by quantity available
        actual_sales = min(quantity, potential_sales)
        
        return actual_sales

    def calculate_reputation_change(self, glasses_sold: int, quantity: int, 
                                  quality: float, weather: str, event_multiplier: float = 1.0) -> float:
        """Calculate change in reputation based on performance, with event multiplier"""
        # Sales performance effect
        if glasses_sold == 0:
            performance_effect = -0.3
        else:
            sell_ratio = glasses_sold / quantity
            if sell_ratio >= 0.9:
                performance_effect = 0.3
            elif sell_ratio >= 0.7:
                performance_effect = 0.1
            elif sell_ratio >= 0.5:
                performance_effect = 0.0
            else:
                performance_effect = -0.1
        # Quality effect
        if quality >= 8:
            quality_effect = 0.2
        elif quality >= 6:
            quality_effect = 0.0
        else:
            quality_effect = -0.2
        # Weather consideration - bad weather performance is impressive
        weather_effect = 0.0
        if weather == 'rainy' and glasses_sold > 0:
            weather_effect = 0.1
        # Event multiplier gives a small bonus to reputation
        event_effect = (event_multiplier - 1.0) * 0.5  # e.g. multiplier 1.2 -> +0.1
        total_change = performance_effect + quality_effect + weather_effect + event_effect
        return np.clip(total_change, -0.5, 0.5)  # Limit change per day

    def get_traffic_level(self, location: str, weather: str) -> str:
        """Determine traffic level for Bayesian network compatibility"""
        base_traffic = LOCATIONS[location]['base_traffic']
        weather_bonus = LOCATIONS[location].get('weather_bonus', 1.0)
        
        if weather in ['sunny', 'hot']:
            effective_traffic = base_traffic * weather_bonus
        elif weather == 'cloudy':
            effective_traffic = base_traffic * 0.8
        else:  # rainy
            effective_traffic = base_traffic * 0.5
            
        if effective_traffic < 100:
            return 'low'
        elif effective_traffic < 200:
            return 'medium'
        else:
            return 'high'


class PlayerManager:
    """Manages player states and actions"""
    
    def __init__(self):
        self.players = {
            'human': PlayerState(STARTING_CAPITAL, STARTING_REPUTATION, [], []),
            'ai': PlayerState(STARTING_CAPITAL, STARTING_REPUTATION, [], [])
        }
    
    def can_afford(self, player: str, recipe: str, quantity: int, 
                   info_purchases: List[str] = None) -> bool:
        """Check if player can afford the decision"""
        if info_purchases is None:
            info_purchases = []
            
        player_state = self.players[player]
        recipe_cost = RECIPES[recipe]['cost'] * quantity
        info_cost = sum(INFO_MARKET[info]['cost'] for info in info_purchases)
        total_cost = recipe_cost + info_cost
        
        return player_state.money >= total_cost
    
    def execute_purchase(self, player: str, recipe: str, quantity: int,
                        info_purchases: List[str] = None) -> bool:
        """Execute purchase and deduct costs"""
        if info_purchases is None:
            info_purchases = []
            
        if not self.can_afford(player, recipe, quantity, info_purchases):
            return False
            
        player_state = self.players[player]
        recipe_cost = RECIPES[recipe]['cost'] * quantity
        info_cost = sum(INFO_MARKET[info]['cost'] for info in info_purchases)
        total_cost = recipe_cost + info_cost
        
        player_state.money -= total_cost
        return True
    
    def update_player_state(self, player: str, profit: float, 
                           reputation_change: float) -> None:
        """Update player money and reputation"""
        player_state = self.players[player]
        player_state.money += profit
        player_state.reputation = np.clip(
            player_state.reputation + reputation_change, 1.0, 10.0
        )
    
    def is_bankrupt(self, player: str, min_money: float = 1.0) -> bool:
        """Check if player is bankrupt"""
        return self.players[player].money < min_money
    
    def get_player_state(self, player: str) -> PlayerState:
        """Get current player state"""
        return self.players[player]


class GameSimulator:
    """Main game simulation orchestrator"""
    
    def __init__(self):
        self.day = 1
        self.engine = GameEngine()
        self.player_manager = PlayerManager()
        self.meu_analyzer = None  # Will be initialized when needed
        self.last_year_event_multipliers = {}
        self.todays_event_multipliers = {}
        self._init_event_multipliers()

    def _init_event_multipliers(self):
        """Initialize event and multiplier dictionaries for the first day or after reset."""
        self.todays_event_multipliers = {}
        for loc, data in LOCATIONS.items():
            events = data.get('local_events', {})
            if isinstance(events, dict) and events:
                event_today = np.random.choice(list(events.values()))
            elif isinstance(events, str):
                event_today = events
            else:
                event_today = None
            # Set last year's multiplier if not already set
            if loc not in self.last_year_event_multipliers:
                last_year_multiplier = np.round(np.random.uniform(1.05, 1.3), 2)
                self.last_year_event_multipliers[loc] = {'event': event_today, 'multiplier': last_year_multiplier}
            else:
                last_year_multiplier = self.last_year_event_multipliers[loc]['multiplier']
            # Now, generate today's multiplier based on last year's
            if event_today:
                if np.random.random() < 0.9:
                    # 90% chance: within ±10% of last year
                    delta = np.random.uniform(-0.1, 0.1)
                    today_multiplier = np.round(last_year_multiplier * (1 + delta), 2)
                else:
                    # 10% chance: random 0.7–1.3x last year's
                    today_multiplier = np.round(last_year_multiplier * np.random.uniform(0.7, 1.3), 2)
                # Clamp to [0.7, 1.3]
                today_multiplier = max(0.7, min(today_multiplier, 1.3))
                self.todays_event_multipliers[loc] = {'event': event_today, 'multiplier': today_multiplier}
            else:
                self.todays_event_multipliers[loc] = {'event': None, 'multiplier': 1.0}
        
    def initialize_ai(self):
        """Initialize AI components"""
        if self.meu_analyzer is None:
            self.meu_analyzer = MEUAnalyzer(self)
    
    def simulate_day(self, player: str, location: str, recipe: str, price: float,
                    quantity: int, info_purchased: List[str] = None) -> GameResult:
        """Simulate one day of business operations, applying today's event multiplier"""
        if info_purchased is None:
            info_purchased = []
        # Defensive: ensure location and recipe are always strings
        if isinstance(location, list):
            location = location[0] if location else ''
        if isinstance(recipe, list):
            recipe = recipe[0] if recipe else ''
        player_state = self.player_manager.get_player_state(player)
        # Check affordability
        if not self.player_manager.can_afford(player, recipe, quantity, info_purchased):
            return GameResult(
                success=False,
                profit=0.0,
                cups_sold=0,
                revenue=0.0,
                reputation_change=-0.5,
                customers=0,
                quality=0.0,
                waste=0,
                message="Insufficient funds"
            )
        # Execute purchase
        recipe_cost = RECIPES[recipe]['cost']
        total_cost = quantity * recipe_cost
        info_cost = sum(INFO_MARKET[info]['cost'] for info in info_purchased)
        if not self.player_manager.execute_purchase(player, recipe, quantity, info_purchased):
            return GameResult(
                success=False,
                profit=0.0,
                cups_sold=0,
                revenue=0.0,
                reputation_change=-0.5,
                customers=0,
                quality=0.0,
                waste=0,
                message="Purchase failed"
            )
        # Get today's event multiplier for this location
        event_multiplier = 1.0
        if hasattr(self, 'todays_event_multipliers') and location in self.todays_event_multipliers:
            event_multiplier = self.todays_event_multipliers[location].get('multiplier', 1.0)
        # Calculate game variables
        quality = self.engine.calculate_quality(recipe, player_state.reputation)
        customers = self.engine.calculate_customers(
            location, self.engine.current_weather, quality, price, player_state.reputation, event_multiplier
        )
        # Determine sales
        glasses_sold = self.engine.calculate_sales(customers, quantity, quality, price)
        # Calculate financial results
        revenue = glasses_sold * price
        profit = revenue - total_cost - info_cost
        waste = quantity - glasses_sold
        # Calculate reputation change
        reputation_change = self.engine.calculate_reputation_change(
            glasses_sold, quantity, quality, self.engine.current_weather, event_multiplier
        )
        # Update player state
        self.player_manager.update_player_state(player, revenue, reputation_change)
        # Log the day's results
        day_log = {
            'day': self.day,
            'location': location,
            'recipe': recipe,
            'price': price,
            'quantity': quantity,
            'glasses_sold': glasses_sold,
            'revenue': revenue,
            'profit': profit,
            'reputation_change': reputation_change,
            'weather': self.engine.current_weather,
            'quality': quality,
            'customers': customers,
            'info_purchased': info_purchased,
            'waste': waste,
            'event_multiplier': event_multiplier
        }
        player_state.daily_logs.append(day_log)
        return GameResult(
            success=True,
            profit=profit,
            cups_sold=glasses_sold,
            revenue=revenue,
            reputation_change=reputation_change,
            customers=customers,
            quality=quality,
            waste=waste
        )
    
    def ai_make_decision(self, info_budget_ratio: float = 0.15) -> Tuple[str, str, float, int, List[str]]:
        """AI makes optimal decision using MEU analysis (auto-buy all info)"""
        self.initialize_ai()
        ai_state = self.player_manager.get_player_state('ai')
        # AI always purchases all available information
        info_purchased = list(INFO_MARKET.keys())
        # Generate optimal decision using MEU analysis
        best_decision = self.meu_analyzer.find_optimal_decision(
            list(LOCATIONS.keys()),
            self.engine.weather_forecast,
            ai_state,
            info_purchased
        )
        # Defensive: ensure location and recipe are strings, not lists
        location = best_decision.location
        recipe = best_decision.recipe
        if isinstance(location, list):
            location = location[0] if location else ''
        if isinstance(recipe, list):
            recipe = recipe[0] if recipe else ''
        # Log AI decision process
        ai_state.decision_logs.append({
            'day': self.day,
            'decision': best_decision,
            'info_purchased': info_purchased,
            'reasoning': getattr(best_decision, 'reasoning', None)
        })
        return (
            location,
            recipe,
            best_decision.price,
            best_decision.quantity,
            info_purchased
        )
    
    def advance_day(self):
        """Advance to next day and generate new weather and event multipliers"""
        self.day += 1
        self.engine.generate_weather()
        self._init_event_multipliers()
    
    def check_game_over(self) -> Tuple[bool, str]:
        """Check if game should end"""
        human_bankrupt = self.player_manager.is_bankrupt('human')
        ai_bankrupt = self.player_manager.is_bankrupt('ai')
        
        if human_bankrupt and ai_bankrupt:
            return True, "Both players bankrupt!"
        elif human_bankrupt:
            return True, "Human player bankrupt - AI wins!"
        elif ai_bankrupt:
            return True, "AI bankrupt - Human wins!"
        
        return False, ""
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state summary"""
        return {
            'day': self.day,
            'weather_forecast': self.engine.weather_forecast,
            'current_weather': self.engine.current_weather,
            'players': {
                name: {
                    'money': state.money,
                    'reputation': state.reputation,
                    'total_profit': sum(log['profit'] for log in state.daily_logs),
                    'total_sales': sum(log.get('glasses_sold', log.get('cups_sold', 0)) for log in state.daily_logs)
                }
                for name, state in self.player_manager.players.items()
            }
        }
    
    def calculate_final_scores(self) -> Dict[str, float]:
        """Calculate final scores for comparison"""
        scores = {}
        for player_name, player_state in self.player_manager.players.items():
            # Score = money + reputation bonus + consistency bonus
            money_score = player_state.money
            reputation_score = player_state.reputation * 2.0  # Reputation worth 2x
            
            # Consistency bonus - reward steady performance
            if len(player_state.daily_logs) > 0:
                daily_profits = [log['profit'] for log in player_state.daily_logs]
                consistency_score = max(0, 10 - np.std(daily_profits))
            else:
                consistency_score = 0
                
            total_score = money_score + reputation_score + consistency_score
            scores[player_name] = total_score
            
        return scores


class GameInterface:
    """Handles user interaction and display"""
    
    def __init__(self, simulator: GameSimulator):
        self.sim = simulator
    
    def display_game_state(self):
        """Display current game state"""
        state = self.sim.get_game_state()
        
        print(f"\n{'='*60}")
        print(f"DAY {state['day']} - GAME STATUS")
        print(f"{'='*60}")
        print(f"Weather Forecast: {state['weather_forecast'].title()}")
        if state['current_weather']:
            print(f"Actual Weather: {state['current_weather'].title()}")
        
        for player_name, player_data in state['players'].items():
            print(f"\n{player_name.upper()}:")
            print(f"  Money: ${player_data['money']:.2f}")
            print(f"  Reputation: {player_data['reputation']:.1f}/10")
            if player_data['total_profit'] != 0:
                print(f"  Total Profit: ${player_data['total_profit']:.2f}")
                print(f"  Total Sales: {player_data['total_sales']} glasses")
    
    def display_daily_results(self):
        """Display results for the current day"""
        print(f"\n{'='*60}")
        print(f"DAY {self.sim.day} RESULTS")
        print(f"{'='*60}")
        print(f"Actual Weather: {self.sim.engine.current_weather.title()}")
        
        for player_name, player_state in self.sim.player_manager.players.items():
            if (player_state.daily_logs and 
                player_state.daily_logs[-1]['day'] == self.sim.day):
                
                log = player_state.daily_logs[-1]
                # Defensive: ensure location and recipe are strings before .title()
                location = log['location']
                recipe = log['recipe']
                if isinstance(location, list):
                    location = location[0] if location else ''
                if isinstance(recipe, list):
                    recipe = recipe[0] if recipe else ''
                print(f"\n{player_name.upper()}:")
                print(f"  Location: {str(location).title()}")
                print(f"  Recipe: {str(recipe).title()}")
                print(f"  Price: ${log['price']:.2f}")
                print(f"  Made: {log['quantity']} glasses")
                print(f"  Sold: {log.get('glasses_sold', log.get('cups_sold', 0))} glasses")
                print(f"  Waste: {log['waste']} glasses")
                print(f"  Revenue: ${log['revenue']:.2f}")
                print(f"  Profit: ${log['profit']:+.2f}")
                print(f"  Quality: {log['quality']:.1f}/10")
                print(f"  Customers: {log['customers']}")
                print(f"  Reputation: {player_state.reputation:.1f}/10 "
                      f"({log['reputation_change']:+.1f})")
                
                if log.get('info_purchased'):
                    print(f"  Info Purchased: {', '.join(log['info_purchased'])}")
    
    def get_human_decision(self) -> Tuple[str, str, float, int, List[str]]:
        """Get decision input from human player (auto-buy all info before any prompts)"""
        player_state = self.sim.player_manager.get_player_state('human')
        # Automatically purchase all available information before any prompts
        info_purchased = list(INFO_MARKET.keys())
        print(f"\n{'='*60}")
        print("YOUR TURN - Make Your Business Decisions")
        print(f"{'='*60}")
        print(f"Available Money: ${player_state.money:.2f}")
        print(f"Current Reputation: {player_state.reputation:.1f}/10")
        print(f"\n--- INFORMATION MARKET (Auto-purchased) ---")
        for info_type, info_data in INFO_MARKET.items():
            if info_type == 'competition_intel':
                print(f"{info_type}: ${info_data['cost']:.2f} - {info_data['description']}")
            else:
                print(f"{info_type}: ${info_data['cost']:.2f} - {info_data['description']}")
        # Show premium weather forecast and accuracy
        premium = INFO_MARKET.get('premium_weather', {})
        accuracy = premium.get('accuracy_boost', 0.0) + 0.6  # base 60% + boost
        
        print(f"\n--- PREMIUM WEATHER FORECAST ---")
        print(f"Premium Weather Forecast: {self.sim.engine.weather_forecast.title()} (Accuracy: {accuracy:.0%})")
        # Business decision phase
        location = self._get_location_choice()
        # Calculate available money after info costs
        info_cost = sum(INFO_MARKET[info]['cost'] for info in info_purchased)
        money_after_info = player_state.money - info_cost
        recipe = self._get_recipe_choice(money_after_info)
        price = self._get_price_choice()
        quantity = self._get_quantity_choice(recipe, money_after_info)
        return location, recipe, price, quantity, info_purchased
    
    def _handle_info_purchases(self, player_state: PlayerState) -> List[str]:
        """Handle information purchase interface"""
        info_purchased = []
        current_money = player_state.money
        
        print(f"\n--- INFORMATION MARKET ---")
        for info_type, info_data in INFO_MARKET.items():
            cost = info_data['cost']
            desc = info_data['description']
            status = "✓" if current_money >= cost else "✗"
            print(f"{status} {info_type}: ${cost:.2f} - {desc}")
        
        while True:
            raw_choice = input("\nBuy information? (type name or 'done'): ").strip()
            choice = raw_choice.lower()
            if choice == 'done':
                break
            elif choice in INFO_MARKET:
                cost = INFO_MARKET[choice]['cost']
                if current_money >= cost and choice not in info_purchased:
                    info_purchased.append(choice)
                    current_money -= cost
                    print(f"✓ Purchased {choice} for ${cost:.2f}")
                    print(f"Remaining money: ${current_money:.2f}")
                elif choice in info_purchased:
                    print("Already purchased!")
                else:
                    print("Insufficient funds!")
            else:
                print("Invalid choice. Available: " + ", ".join(INFO_MARKET.keys()))
        
        return info_purchased
    
    def _get_location_choice(self) -> str:
        """Get location choice from user by number, showing local event and last year's multiplier"""
        print(f"\n--- LOCATIONS ---")
        todays_events = getattr(self.sim, 'todays_event_multipliers', {})
        last_year_events = getattr(self.sim, 'last_year_event_multipliers', {})
        location_keys = list(LOCATIONS.keys())
        for idx, loc in enumerate(location_keys, 1):
            data = LOCATIONS[loc]
            comp = COMPETITION_STATS.get(loc, {})
            event_today = todays_events.get(loc, {}).get('event', None)
            last_year_multiplier = last_year_events.get(loc, {}).get('multiplier', 1.0)
            local_events_str = event_today if event_today else 'None'
            competitor_price = comp.get('avg_price', None)
            print(f"[{idx}] {loc.title()}:")
            print(f"    Local Events: {local_events_str}")
            print(f"    Last year's multiplier: {last_year_multiplier:.2f}")
            print(f"    Last years traffic: {data['base_traffic']}")
            print(f"    Competition: {comp.get('num_stands', 0)} stands")
            if competitor_price is not None:
                print(f"    Lowest Competitor Price: ${competitor_price:.2f} per cup")
        while True:
            raw_location = input(f"Choose location [1-{len(location_keys)}]: ").strip()
            if raw_location.isdigit():
                idx = int(raw_location)
                if 1 <= idx <= len(location_keys):
                    return location_keys[idx-1]
            print(f"Invalid selection! Enter a number between 1 and {len(location_keys)}.")

    def _get_recipe_choice(self, available_money: float) -> str:
        """Get recipe choice from user by number, sorted by cost, showing max cups for each recipe based on available money"""
        print(f"\n--- RECIPES ---")
        # Sort recipes by cost (lowest to highest)
        sorted_recipes = sorted(RECIPES.items(), key=lambda x: x[1]['cost'])
        for idx, (recipe_name, recipe_data) in enumerate(sorted_recipes, 1):
            max_cups = int(available_money / recipe_data['cost']) if recipe_data['cost'] > 0 else 0
            print(f"[{idx}] {recipe_name.title()}: ${recipe_data['cost']:.2f}/cup, "
                  f"Quality: {recipe_data['base_quality']}, "
                  f"Appeal: {recipe_data['appeal']}, "
                  f"Max cups: {max_cups}")
        while True:
            raw_recipe = input(f"Choose recipe [1-{len(sorted_recipes)}]: ").strip()
            if raw_recipe.isdigit():
                idx = int(raw_recipe)
                if 1 <= idx <= len(sorted_recipes):
                    return sorted_recipes[idx-1][0]
            print(f"Invalid selection! Enter a number between 1 and {len(sorted_recipes)}.")
    
    def _get_price_choice(self) -> float:
        """Get price choice from user"""
        while True:
            try:
                price = float(input(f"Set price (${PRICE_MIN:.2f}-${PRICE_MAX:.2f}): $"))
                if PRICE_MIN <= price <= PRICE_MAX:
                    return price
                print(f"Price must be between ${PRICE_MIN:.2f} and ${PRICE_MAX:.2f}")
            except ValueError:
                print("Invalid price! Enter a number.")
    
    def _get_quantity_choice(self, recipe: str, available_money: float) -> int:
        """Get quantity choice from user"""
        recipe_cost = RECIPES[recipe]['cost']
        max_quantity = int(available_money / recipe_cost)
        print(f"\nMax affordable quantity: {max_quantity} cups")
        
        while True:
            try:
                quantity = int(input("How many cups to make? "))
                if 1 <= quantity <= max_quantity:
                    return quantity
                print(f"Quantity must be between 1 and {max_quantity}")
            except ValueError:
                print("Invalid quantity! Enter a whole number.")
    
    def display_final_results(self):
        """Display final game results"""
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        
        scores = self.sim.calculate_final_scores()
        
        # Display player stats
        for player_name, player_state in self.sim.player_manager.players.items():
            total_profit = sum(log['profit'] for log in player_state.daily_logs)
            total_sales = sum(log['cups_sold'] for log in player_state.daily_logs)
            
            print(f"\n{player_name.upper()}:")
            print(f"  Final Money: ${player_state.money:.2f}")
            print(f"  Final Reputation: {player_state.reputation:.1f}/10")
            print(f"  Total Profit: ${total_profit:.2f}")
            print(f"  Total Sales: {total_sales} cups")
            print(f"  Final Score: {scores[player_name]:.2f}")
        
        # Determine winner
        winner = max(scores.keys(), key=lambda k: scores[k])
        if scores['human'] == scores['ai']:
            print(f"\n🤝 IT'S A TIE! 🤝")
        elif winner == 'human':
            print(f"\n🎉 HUMAN WINS! 🎉")
        else:
            print(f"\n🤖 AI WINS! 🤖")
        
        print(f"\nScore Difference: {abs(scores['human'] - scores['ai']):.2f}")


# Convenience properties for backward compatibility
@property
def recipes(self):
    """Backward compatibility property"""
    return RECIPES

@property  
def competition_stats(self):
    """Backward compatibility property"""
    return self.engine.competition_stats

# Add properties to GameSimulator for MEU analyzer compatibility
GameSimulator.recipes = recipes
GameSimulator.competition_stats = competition_stats