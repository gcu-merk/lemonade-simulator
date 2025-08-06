"""
Bayesian Network implementation for the Lemonade Stand Simulator

This module provides a complete Bayesian network for modeling decision-making
under uncertainty in the lemonade stand business simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Tuple, List, Optional, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

class BayesianNode:
    """
    Represents a node in the Bayesian network with conditional probability tables
    """
    def __init__(self, name: str, states: List[str], parents: List[str], 
                 cpt: Dict[Tuple, Dict[str, float]]):
        self.name = name
        self.states = states
        self.parents = parents
        self.cpt = cpt
        self._validate_cpt()

    def _validate_cpt(self):
        """Validate that CPT probabilities sum to 1.0 for each parent combination"""
        for parent_combo, state_probs in self.cpt.items():
            total_prob = sum(state_probs.values())
            if not np.isclose(total_prob, 1.0, atol=1e-6):
                logger.warning(f"CPT for {self.name} with parents {parent_combo} "
                             f"sums to {total_prob:.6f}, not 1.0")

    def get_probability(self, state: str, parent_states: Optional[Dict[str, str]] = None) -> float:
        """Get probability of a state given parent states"""
        if parent_states is None:
            parent_states = {}

        # Create key from parent states in correct order
        key = tuple(parent_states.get(parent, None) for parent in self.parents)
        if not self.parents:
            key = ()

        try:
            return self.cpt.get(key, {}).get(state, 0.0)
        except KeyError:
            logger.warning(f"No CPT entry found for {self.name} with key {key}")
            return 0.0

    def get_all_probabilities(self, parent_states: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get all state probabilities given parent states"""
        if parent_states is None:
            parent_states = {}

        key = tuple(parent_states.get(parent, None) for parent in self.parents)
        if not self.parents:
            key = ()

        return self.cpt.get(key, {state: 0.0 for state in self.states})


class BayesianNetwork:
    """Bayesian Network for lemonade stand decision making"""

    def __init__(self):
        self.nodes: Dict[str, BayesianNode] = {}
        self._build_network()
        self._validate_network()

    def _build_network(self):
        """Build the complete Bayesian network structure"""
        logger.info("Building Bayesian network structure...")

        # 1. Weather Node (root node)
        weather_cpt = {
            (): {'sunny': 0.4, 'cloudy': 0.3, 'rainy': 0.2, 'hot': 0.1}
        }
        self.nodes['weather'] = BayesianNode(
            'weather', ['sunny', 'cloudy', 'rainy', 'hot'], [], weather_cpt
        )

        # 2. Forecast Node (depends on weather)
        self.nodes['forecast'] = BayesianNode(
            'forecast', ['sunny', 'cloudy', 'rainy', 'hot'], 
            ['weather'], self._build_forecast_cpt()
        )

        # 3. Traffic Node (depends on weather and location)
        self.nodes['traffic'] = BayesianNode(
            'traffic', ['low', 'medium', 'high'],
            ['weather', 'location'], self._build_traffic_cpt()
        )

        # 4. Demand Node (depends on traffic, price, quality, competition)
        self.nodes['demand'] = BayesianNode(
            'demand', ['very_low', 'low', 'medium', 'high', 'very_high'],
            ['traffic', 'price_level', 'quality_level', 'competition_level'],
            self._build_demand_cpt()
        )

        # 5. Sales Node (depends on demand and quantity)
        self.nodes['sales'] = BayesianNode(
            'sales', ['very_low', 'low', 'medium', 'high', 'very_high'],
            ['demand', 'quantity_level'], self._build_sales_cpt()
        )

        # 6. Profit Node (depends on sales and costs)
        self.nodes['profit'] = BayesianNode(
            'profit', ['loss', 'break_even', 'low_profit', 'good_profit', 'high_profit'],
            ['sales', 'cost_level'], self._build_profit_cpt()
        )

        logger.info(f"Network built with {len(self.nodes)} nodes")

    def _build_forecast_cpt(self) -> Dict[Tuple, Dict[str, float]]:
        """Build conditional probability table for weather forecast"""
        forecast_cpt = {}
        base_accuracy = 0.6  # Base forecast accuracy
        
        for actual in ['sunny', 'cloudy', 'rainy', 'hot']:
            forecast_prob = {}
            for predicted in ['sunny', 'cloudy', 'rainy', 'hot']:
                if actual == predicted:
                    forecast_prob[predicted] = base_accuracy
                else:
                    forecast_prob[predicted] = (1 - base_accuracy) / 3
            forecast_cpt[(actual,)] = forecast_prob
        
        return forecast_cpt

    def _build_traffic_cpt(self) -> Dict[Tuple, Dict[str, float]]:
        """Build conditional probability table for traffic"""
        return {
            ('sunny', 'park'): {'low': 0.05, 'medium': 0.25, 'high': 0.70},
            ('hot', 'park'): {'low': 0.10, 'medium': 0.30, 'high': 0.60},
            ('cloudy', 'park'): {'low': 0.25, 'medium': 0.55, 'high': 0.20},
            ('rainy', 'park'): {'low': 0.70, 'medium': 0.25, 'high': 0.05},

            ('sunny', 'school'): {'low': 0.15, 'medium': 0.50, 'high': 0.35},
            ('hot', 'school'): {'low': 0.10, 'medium': 0.45, 'high': 0.45},
            ('cloudy', 'school'): {'low': 0.20, 'medium': 0.60, 'high': 0.20},
            ('rainy', 'school'): {'low': 0.25, 'medium': 0.55, 'high': 0.20},

            ('sunny', 'mall'): {'low': 0.25, 'medium': 0.45, 'high': 0.30},
            ('hot', 'mall'): {'low': 0.10, 'medium': 0.35, 'high': 0.55},
            ('cloudy', 'mall'): {'low': 0.15, 'medium': 0.50, 'high': 0.35},
            ('rainy', 'mall'): {'low': 0.10, 'medium': 0.40, 'high': 0.50}
        }

    def _build_demand_cpt(self) -> Dict[Tuple, Dict[str, float]]:
        """Build conditional probability table for demand"""
        cpt = {}

        # Base demand probabilities for different traffic levels
        traffic_effects = {
            'low': [0.4, 0.3, 0.2, 0.08, 0.02],      # very_low, low, medium, high, very_high
            'medium': [0.15, 0.25, 0.35, 0.2, 0.05],
            'high': [0.05, 0.15, 0.3, 0.35, 0.15]
        }

        # Effect multipliers
        price_effects = {
            'low': [0.8, 0.9, 1.1, 1.3, 1.5],       # Boosts higher demand
            'medium': [1.0, 1.0, 1.0, 1.0, 1.0],    # Neutral
            'high': [1.5, 1.3, 1.0, 0.7, 0.4]       # Reduces higher demand
        }

        quality_effects = {
            'low': [1.3, 1.2, 0.9, 0.7, 0.5],
            'medium': [1.0, 1.0, 1.0, 1.0, 1.0],
            'high': [0.7, 0.8, 1.1, 1.3, 1.5]
        }

        competition_effects = {
            'low': [0.8, 0.9, 1.0, 1.2, 1.4],
            'medium': [1.0, 1.0, 1.0, 1.0, 1.0],
            'high': [1.4, 1.2, 1.0, 0.8, 0.6]
        }

        demand_states = ['very_low', 'low', 'medium', 'high', 'very_high']

        for traffic in ['low', 'medium', 'high']:
            for price in ['low', 'medium', 'high']:
                for quality in ['low', 'medium', 'high']:
                    for competition in ['low', 'medium', 'high']:
                        # Calculate adjusted probabilities
                        base_probs = np.array(traffic_effects[traffic])
                        price_mult = np.array(price_effects[price])
                        quality_mult = np.array(quality_effects[quality])
                        comp_mult = np.array(competition_effects[competition])

                        adjusted_probs = base_probs * price_mult * quality_mult * comp_mult
                        # Normalize to ensure probabilities sum to 1
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()

                        cpt[(traffic, price, quality, competition)] = {
                            state: prob for state, prob in zip(demand_states, adjusted_probs)
                        }

        return cpt

    def _build_sales_cpt(self) -> Dict[Tuple, Dict[str, float]]:
        """Build conditional probability table for sales given demand and quantity"""
        cpt = {}
        demand_states = ['very_low', 'low', 'medium', 'high', 'very_high']
        quantity_states = ['very_low', 'low', 'medium', 'high', 'very_high']
        sales_states = ['very_low', 'low', 'medium', 'high', 'very_high']

        for demand in demand_states:
            for quantity in quantity_states:
                demand_idx = demand_states.index(demand)
                quantity_idx = quantity_states.index(quantity)

                sales_probs = [0.0] * 5
                limiting_factor = min(demand_idx, quantity_idx)

                # Distribute probability around the limiting factor
                for i in range(5):
                    if i == limiting_factor:
                        sales_probs[i] += 0.6
                    elif abs(i - limiting_factor) == 1:
                        sales_probs[i] += 0.2
                    else:
                        sales_probs[i] += 0.05

                # Apply waste penalty if quantity >> demand
                if quantity_idx > demand_idx + 1:
                    for i in range(len(sales_probs) - 1):
                        sales_probs[i] += sales_probs[i + 1] * 0.3
                        sales_probs[i + 1] *= 0.7

                # Normalize
                total = sum(sales_probs)
                if total > 0:
                    sales_probs = [p / total for p in sales_probs]

                cpt[(demand, quantity)] = {
                    state: prob for state, prob in zip(sales_states, sales_probs)
                }

        return cpt

    def _build_profit_cpt(self) -> Dict[Tuple, Dict[str, float]]:
        """Build conditional probability table for profit"""
        cpt = {}
        sales_states = ['very_low', 'low', 'medium', 'high', 'very_high']
        cost_states = ['very_low', 'low', 'medium', 'high', 'very_high']
        profit_states = ['loss', 'break_even', 'low_profit', 'good_profit', 'high_profit']

        for sales in sales_states:
            for cost in cost_states:
                sales_idx = sales_states.index(sales)
                cost_idx = cost_states.index(cost)

                # Profit = sales - costs (conceptually)
                profit_score = sales_idx - cost_idx
                profit_probs = [0.0] * 5

                if profit_score <= -2:      # High cost, low sales
                    profit_probs[0] = 0.7   # loss
                    profit_probs[1] = 0.2   # break_even
                    profit_probs[2] = 0.1   # low_profit
                elif profit_score == -1:
                    profit_probs[0] = 0.4
                    profit_probs[1] = 0.4
                    profit_probs[2] = 0.2
                elif profit_score == 0:
                    profit_probs[1] = 0.3   # break_even
                    profit_probs[2] = 0.4   # low_profit
                    profit_probs[3] = 0.3   # good_profit
                elif profit_score == 1:
                    profit_probs[2] = 0.2
                    profit_probs[3] = 0.5
                    profit_probs[4] = 0.3
                else:  # profit_score >= 2
                    profit_probs[3] = 0.3
                    profit_probs[4] = 0.7   # high_profit

                cpt[(sales, cost)] = {
                    state: prob for state, prob in zip(profit_states, profit_probs)
                }

        return cpt

    def _validate_network(self):
        """Validate the network structure and CPTs"""
        logger.info("Validating network structure...")
        
        for node_name, node in self.nodes.items():
            # Check that all parent nodes exist
            for parent in node.parents:
                if parent not in self.nodes and parent not in ['location', 'recipe', 'price_level', 'quantity_level', 'quality_level', 'competition_level', 'cost_level']:
                    logger.warning(f"Node {node_name} has undefined parent: {parent}")
        
        logger.info("Network validation complete")

    def query_probability(self, target_node: str, target_state: str,
                         evidence: Dict[str, str]) -> float:
        """
        Query probability using simplified inference
        
        Args:
            target_node: Node to query
            target_state: State of the target node
            evidence: Dictionary of evidence (node_name -> state)
            
        Returns:
            Probability of target_state given evidence
        """
        if target_node not in self.nodes:
            logger.error(f"Node {target_node} not found in network")
            return 0.0

        node = self.nodes[target_node]

        # If no parents, return prior probability
        if not node.parents:
            return node.get_probability(target_state)

        # Check if all parents are in evidence
        missing_parents = [p for p in node.parents if p not in evidence]
        if missing_parents:
            logger.debug(f"Missing evidence for parents: {missing_parents}")
            # Use uniform distribution for missing evidence
            return 0.5  # Default uncertainty

        return node.get_probability(target_state, evidence)

    def query_all_states(self, target_node: str, evidence: Dict[str, str]) -> Dict[str, float]:
        """Query probabilities for all states of a target node"""
        if target_node not in self.nodes:
            logger.error(f"Node {target_node} not found in network")
            return {}

        node = self.nodes[target_node]
        return node.get_all_probabilities(evidence)

    def get_node_states(self, node_name: str) -> List[str]:
        """Get all possible states for a node"""
        if node_name not in self.nodes:
            return []
        return self.nodes[node_name].states

    def get_network_structure(self) -> Dict[str, List[str]]:
        """Get the network structure as a dictionary of node -> parents"""
        return {name: node.parents for name, node in self.nodes.items()}

    def visualize_network(self, show_cpt_info: bool = False):
        """Create a visualization of the Bayesian network structure"""
        fig_width = 20 if show_cpt_info else 16
        fig, axes = plt.subplots(1, 2 if not show_cpt_info else 3, figsize=(fig_width, 8))
        
        if not show_cpt_info:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes

        # Create directed graph
        G = nx.DiGraph()

        # Node positions for better layout
        node_positions = {
            'weather': (0, 3),
            'forecast': (1, 4),
            'location': (1, 2),    # Decision node
            'traffic': (2, 3),
            'recipe': (2, 1),      # Decision node
            'price_level': (2, 0), # Decision node
            'quantity_level': (2, -1), # Decision node
            'quality_level': (3, 1),
            'competition_level': (3, 2),
            'cost_level': (3, 0),
            'demand': (4, 2),
            'sales': (5, 2),
            'profit': (6, 2)       # Utility node
        }

        # Add edges based on dependencies
        edges = [
            ('weather', 'forecast'),
            ('weather', 'traffic'),
            ('location', 'traffic'),
            ('location', 'competition_level'),
            ('recipe', 'quality_level'),
            ('recipe', 'cost_level'),
            ('price_level', 'demand'),
            ('quality_level', 'demand'),
            ('competition_level', 'demand'),
            ('traffic', 'demand'),
            ('demand', 'sales'),
            ('quantity_level', 'sales'),
            ('sales', 'profit'),
            ('cost_level', 'profit')
        ]

        G.add_edges_from(edges)

        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            if node in ['location', 'recipe', 'price_level', 'quantity_level']:
                node_colors.append('lightblue')  # Decision nodes
            elif node == 'profit':
                node_colors.append('lightgreen')  # Utility node
            elif node in ['forecast', 'quality_level', 'competition_level', 'cost_level']:
                node_colors.append('lightyellow')  # Derived nodes
            else:
                node_colors.append('lightcoral')  # Chance nodes

        # Draw network
        nx.draw(G, node_positions, ax=ax1, with_labels=True, node_color=node_colors,
                node_size=2000, font_size=8, font_weight='bold', arrows=True,
                edge_color='gray', arrowsize=20)

        ax1.set_title('Lemonade Stand Decision Network', fontsize=14, fontweight='bold')
        ax1.text(0.02, 0.98, 'Blue: Decision Nodes\nGreen: Utility Node\nYellow: Derived\nRed: Chance Nodes',
                transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Show MEU equation
        ax2.text(0.1, 0.9, 'Maximum Expected Utility (MEU) Calculation:',
                fontsize=14, fontweight='bold', transform=ax2.transAxes)

        meu_text = """
MEU(d) = Σ P(s|d) × U(s)

Where:
• d = decision (location, recipe, price, quantity)
• s = state (weather, traffic, demand, sales, profit)
• P(s|d) = probability of state s given decision d
• U(s) = utility of state s

Specific Utility Function:
U(profit, reputation) = α × profit + β × reputation_change

Where:
• α = weight for profit (typically 1.0)
• β = weight for reputation (typically 2.0)

Decision Process:
1. For each possible decision d:
   a. Calculate P(s|d) using Bayesian Network
   b. Sum over all possible outcomes: Σ P(s|d) × U(s)
2. Choose d* = argmax MEU(d)

Network Statistics:
• Nodes: {len(self.nodes)}
• Decision Nodes: 4 (location, recipe, price, quantity)
• Chance Nodes: {len([n for n in self.nodes if n not in ['profit']])}
• Utility Nodes: 1 (profit)
        """

        ax2.text(0.1, 0.8, meu_text, fontsize=11, transform=ax2.transAxes,
                verticalalignment='top', fontfamily='monospace')

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        # Optional: Show CPT statistics
        if show_cpt_info:
            cpt_info = "CPT Information:\n" + "="*20 + "\n"
            for name, node in self.nodes.items():
                cpt_info += f"{name}:\n"
                cpt_info += f"  States: {len(node.states)}\n"
                cpt_info += f"  Parents: {len(node.parents)}\n"
                cpt_info += f"  CPT Size: {len(node.cpt)}\n\n"

            ax3.text(0.05, 0.95, cpt_info, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            ax3.set_title('Network Statistics', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def __str__(self) -> str:
        """String representation of the network"""
        return (f"BayesianNetwork with {len(self.nodes)} nodes: "
                f"{list(self.nodes.keys())}")

    def __repr__(self) -> str:
        return self.__str__()