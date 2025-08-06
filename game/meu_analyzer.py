"""
Maximum Expected Utility analyzer using Bayesian Network
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from models.bayesian_network import BayesianNetwork
from models.data_classes import MEUDecision, PlayerState
from config.constants import RECIPES


class MEUAnalyzer:
    """Maximum Expected Utility analyzer using Bayesian Network"""

    def __init__(self, simulator_ref):
        self.sim = simulator_ref
        self.bayes_net = BayesianNetwork()

    def calculate_expected_utility(self, location: str, recipe: str, price: float,
                                 quantity: int, weather_forecast: str,
                                 player_state: PlayerState,
                                 info_purchased: List[str]) -> Tuple[float, float, List[str], Dict]:
        """Calculate MEU using Bayesian Network inference"""

        reasoning = []
        bayes_analysis = {}

        # Discretize continuous variables
        price_level = 'low' if price < 1.0 else 'medium' if price < 1.5 else 'high'
        quantity_level = self._discretize_quantity(quantity, player_state.money, recipe)
        quality_level = self._calculate_quality_level(recipe, player_state.reputation)
        competition_level = self._get_competition_level(location)
        cost_level = self._calculate_cost_level(quantity, recipe)

        # Set evidence
        evidence = {
            'forecast': weather_forecast,
            'location': location,
            'recipe': recipe,
            'price_level': price_level,
            'quantity_level': quantity_level,
            'quality_level': quality_level,
            'competition_level': competition_level,
            'cost_level': cost_level
        }

        bayes_analysis['evidence'] = evidence

        # Calculate probabilities for different outcomes
        expected_utility = 0.0
        total_probability = 0.0

        # Iterate over possible weather outcomes
        weather_probs = self._get_weather_probabilities(weather_forecast, info_purchased)

        for actual_weather, weather_prob in weather_probs.items():
            # Update evidence with actual weather
            current_evidence = evidence.copy()
            current_evidence['weather'] = actual_weather

            # Calculate traffic probabilities
            for traffic_state in ['low', 'medium', 'high']:
                traffic_prob = self.bayes_net.query_probability('traffic', traffic_state, current_evidence)

                current_evidence['traffic'] = traffic_state

                # Calculate demand probabilities
                for demand_state in ['very_low', 'low', 'medium', 'high', 'very_high']:
                    demand_prob = self.bayes_net.query_probability('demand', demand_state, current_evidence)

                    current_evidence['demand'] = demand_state

                    # Calculate sales probabilities
                    for sales_state in ['very_low', 'low', 'medium', 'high', 'very_high']:
                        sales_prob = self.bayes_net.query_probability('sales', sales_state, current_evidence)

                        current_evidence['sales'] = sales_state

                        # Calculate profit probabilities
                        for profit_state in ['loss', 'break_even', 'low_profit', 'good_profit', 'high_profit']:
                            profit_prob = self.bayes_net.query_probability('profit', profit_state, current_evidence)

                            # Calculate joint probability
                            joint_prob = weather_prob * traffic_prob * demand_prob * sales_prob * profit_prob

                            # Calculate utility for this outcome
                            utility = self._calculate_utility(profit_state, sales_state, quantity, recipe, price)

                            expected_utility += joint_prob * utility
                            total_probability += joint_prob

        # Normalize if needed
        if total_probability > 0:
            expected_utility /= total_probability

        # Calculate confidence
        confidence = self._calculate_confidence(info_purchased, weather_forecast)

        # Generate reasoning
        reasoning.extend([
            f"Bayesian Network Analysis for {location.title()} - {recipe} recipe",
            f"Price level: {price_level} (${price:.2f})",
            f"Quality level: {quality_level}",
            f"Competition level: {competition_level}",
            f"Expected utility: {expected_utility:.3f}",
            f"Confidence: {confidence:.1%}"
        ])

        bayes_analysis.update({
            'expected_utility': expected_utility,
            'confidence': confidence,
            'price_level': price_level,
            'quality_level': quality_level,
            'competition_level': competition_level
        })

        return expected_utility, confidence, reasoning, bayes_analysis

    def _discretize_quantity(self, quantity: int, money: float, recipe: str) -> str:
        """Convert quantity to discrete level"""
        cost_per_cup = RECIPES[recipe]['cost']
        max_affordable = int(money / cost_per_cup) if cost_per_cup > 0 else 0

        if max_affordable == 0:
            return 'very_low'

        ratio = quantity / max_affordable
        if ratio < 0.2:
            return 'very_low'
        elif ratio < 0.4:
            return 'low'
        elif ratio < 0.6:
            return 'medium'
        elif ratio < 0.8:
            return 'high'
        else:
            return 'very_high'

    def _calculate_quality_level(self, recipe: str, reputation: float) -> str:
        """Calculate discrete quality level"""
        base_quality = RECIPES[recipe]['base_quality']
        appeal = RECIPES[recipe]['appeal']
        rep_bonus = (reputation - 5.0) * 0.4

        quality = (base_quality * appeal) + rep_bonus

        if quality < 4:
            return 'low'
        elif quality < 7:
            return 'medium'
        else:
            return 'high'

    def _get_competition_level(self, location: str) -> str:
        """Get competition level for location"""
        competition_map = {
            'park': 'low',
            'school': 'medium', 
            'office': 'high',
            'mall': 'very_high',
            'beach': 'medium',
            'downtown': 'high',
            'suburb': 'low'
        }
        return competition_map.get(location.lower(), 'medium')

    def _calculate_cost_level(self, quantity: int, recipe: str) -> str:
        """Calculate cost level based on quantity and recipe"""
        cost_per_cup = RECIPES[recipe]['cost']
        total_cost = quantity * cost_per_cup

        if total_cost < 5:
            return 'very_low'
        elif total_cost < 10:
            return 'low'
        elif total_cost < 20:
            return 'medium'
        elif total_cost < 40:
            return 'high'
        else:
            return 'very_high'

    def _get_weather_probabilities(self, forecast: str, info_purchased: List[str]) -> Dict[str, float]:
        """Get weather outcome probabilities based on forecast and info"""
        base_probs = {
            'sunny': {'sunny': 0.7, 'cloudy': 0.2, 'rainy': 0.1},
            'cloudy': {'sunny': 0.3, 'cloudy': 0.5, 'rainy': 0.2},
            'rainy': {'sunny': 0.1, 'cloudy': 0.3, 'rainy': 0.6}
        }

        # Adjust probabilities based on purchased information
        probs = base_probs.get(forecast, {'sunny': 0.4, 'cloudy': 0.4, 'rainy': 0.2})
        
        if 'weather_report' in info_purchased:
            # More accurate forecast
            for weather in probs:
                if weather == forecast:
                    probs[weather] = min(0.9, probs[weather] + 0.2)
                else:
                    probs[weather] = max(0.05, probs[weather] - 0.1)

        return probs

    def _calculate_utility(self, profit_state: str, sales_state: str, quantity: int, 
                          recipe: str, price: float) -> float:
        """Calculate utility value for outcome"""
        # Base utilities for profit states
        profit_utilities = {
            'loss': -2.0,
            'break_even': 0.0,
            'low_profit': 1.0,
            'good_profit': 3.0,
            'high_profit': 5.0
        }

        # Base utilities for sales states  
        sales_utilities = {
            'very_low': -1.0,
            'low': -0.5,
            'medium': 0.0,
            'high': 0.5,
            'very_high': 1.0
        }

        base_utility = profit_utilities.get(profit_state, 0.0)
        sales_bonus = sales_utilities.get(sales_state, 0.0)

        # Add reputation bonus for high-quality recipes
        quality_bonus = 0.0
        if recipe in RECIPES:
            quality_bonus = RECIPES[recipe].get('base_quality', 5) * 0.1

        # Risk adjustment based on quantity
        risk_penalty = 0.0
        if quantity > 50:
            risk_penalty = -0.5

        return base_utility + sales_bonus + quality_bonus + risk_penalty

    def _calculate_confidence(self, info_purchased: List[str], forecast: str) -> float:
        """Calculate confidence level based on available information"""
        base_confidence = 0.6

        # Increase confidence for purchased information
        if 'weather_report' in info_purchased:
            base_confidence += 0.2
        if 'market_research' in info_purchased:
            base_confidence += 0.15
        if 'competitor_analysis' in info_purchased:
            base_confidence += 0.1

        # Adjust for forecast certainty
        forecast_confidence = {
            'sunny': 0.8,
            'cloudy': 0.6,
            'rainy': 0.7
        }
        
        confidence_multiplier = forecast_confidence.get(forecast, 0.6)
        
        return min(0.95, base_confidence * confidence_multiplier)

    def find_optimal_decision(self, location: str, weather_forecast: str,
                            player_state: PlayerState, 
                            info_purchased: List[str]) -> MEUDecision:
        """Find optimal decision using MEU analysis"""
        # Defensive: if location is a list, use the first element
        if isinstance(location, list):
            location = location[0] if location else ''
        best_utility = float('-inf')
        best_decision = None
        all_analyses = []

        # Generate decision alternatives
        recipes = list(RECIPES.keys())
        prices = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        quantities = [10, 20, 30, 40, 50, 75, 100]


        for recipe in recipes:
            for price in prices:
                for quantity in quantities:
                    # Check affordability
                    cost = quantity * RECIPES[recipe]['cost']
                    if cost > player_state.money:
                        continue

                    # Defensive: ensure location is a string for utility calculation
                    loc = location
                    if isinstance(loc, list):
                        loc = loc[0] if loc else ''

                    # Calculate expected utility
                    utility, confidence, reasoning, analysis = self.calculate_expected_utility(
                        loc, recipe, price, quantity, weather_forecast,
                        player_state, info_purchased
                    )

                    analysis_data = {
                        'location': loc,
                        'recipe': recipe,
                        'price': price,
                        'quantity': quantity,
                        'utility': utility,
                        'confidence': confidence,
                        'cost': cost,
                        'reasoning': reasoning,
                        'analysis': analysis
                    }
                    
                    all_analyses.append(analysis_data)

                    # Update best decision
                    if utility > best_utility:
                        best_utility = utility
                        best_decision = analysis_data

        if best_decision is None:
            # Fallback decision if no affordable options
            cheapest_recipe = min(RECIPES.keys(), key=lambda r: RECIPES[r]['cost'])
            max_quantity = int(player_state.money / RECIPES[cheapest_recipe]['cost'])
            best_decision = {
                'location': location if not isinstance(location, list) else (location[0] if location else ''),
                'recipe': cheapest_recipe,
                'price': 1.0,
                'quantity': max(1, max_quantity),
                'utility': 0.0,
                'confidence': 0.5,
                'cost': max_quantity * RECIPES[cheapest_recipe]['cost'],
                'reasoning': ['Fallback decision - insufficient funds for optimal choice'],
                'analysis': {}
            }

        return MEUDecision(
            location=best_decision['location'],
            recipe=best_decision['recipe'],
            price=best_decision['price'],
            quantity=best_decision['quantity'],
            expected_utility=best_decision['utility'],
            confidence=best_decision['confidence'],
            total_cost=best_decision['cost'],
            reasoning=best_decision['reasoning'],
            all_analyses=all_analyses,
            bayes_analysis=best_decision['analysis']
        )

    def visualize_decision_space(self, analyses: List[Dict], save_path: str = None):
        """Visualize the decision space and utilities"""
        if not analyses:
            return

        df = pd.DataFrame(analyses)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Utility vs Price by Recipe
        sns.scatterplot(data=df, x='price', y='utility', hue='recipe', 
                       size='quantity', ax=axes[0,0])
        axes[0,0].set_title('Expected Utility vs Price by Recipe')
        axes[0,0].set_xlabel('Price ($)')
        axes[0,0].set_ylabel('Expected Utility')
        
        # Utility vs Quantity by Recipe
        sns.scatterplot(data=df, x='quantity', y='utility', hue='recipe',
                       size='confidence', ax=axes[0,1])
        axes[0,1].set_title('Expected Utility vs Quantity by Recipe')
        axes[0,1].set_xlabel('Quantity')
        axes[0,1].set_ylabel('Expected Utility')
        
        # Confidence distribution
        sns.histplot(data=df, x='confidence', bins=20, ax=axes[1,0])
        axes[1,0].set_title('Confidence Distribution')
        axes[1,0].set_xlabel('Confidence')
        axes[1,0].set_ylabel('Count')
        
        # Top decisions
        top_decisions = df.nlargest(10, 'utility')
        sns.barplot(data=top_decisions, x='utility', y='recipe', ax=axes[1,1])
        axes[1,1].set_title('Top 10 Decisions by Expected Utility')
        axes[1,1].set_xlabel('Expected Utility')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def sensitivity_analysis(self, base_decision: Dict, location: str, 
                           weather_forecast: str, player_state: PlayerState,
                           info_purchased: List[str]) -> Dict:
        """Perform sensitivity analysis on key parameters"""
        
        sensitivity_results = {
            'price_sensitivity': [],
            'quantity_sensitivity': [],
            'weather_sensitivity': []
        }
        
        base_recipe = base_decision['recipe']
        base_price = base_decision['price']
        base_quantity = base_decision['quantity']
        
        # Price sensitivity
        price_range = [p for p in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] 
                      if abs(p - base_price) <= 0.5]
        
        for price in price_range:
            utility, confidence, _, _ = self.calculate_expected_utility(
                location, base_recipe, price, base_quantity, weather_forecast,
                player_state, info_purchased
            )
            sensitivity_results['price_sensitivity'].append({
                'price': price,
                'utility': utility,
                'confidence': confidence
            })
        
        # Quantity sensitivity  
        quantity_range = [q for q in [10, 20, 30, 40, 50, 75, 100]
                         if abs(q - base_quantity) <= 30]
        
        for quantity in quantity_range:
            cost = quantity * RECIPES[base_recipe]['cost']
            if cost <= player_state.money:
                utility, confidence, _, _ = self.calculate_expected_utility(
                    location, base_recipe, base_price, quantity, weather_forecast,
                    player_state, info_purchased
                )
                sensitivity_results['quantity_sensitivity'].append({
                    'quantity': quantity,
                    'utility': utility,
                    'confidence': confidence
                })
        
        # Weather sensitivity
        for weather in ['sunny', 'cloudy', 'rainy']:
            utility, confidence, _, _ = self.calculate_expected_utility(
                location, base_recipe, base_price, base_quantity, weather,
                player_state, info_purchased
            )
            sensitivity_results['weather_sensitivity'].append({
                'weather': weather,
                'utility': utility,
                'confidence': confidence
            })
        
        return sensitivity_results