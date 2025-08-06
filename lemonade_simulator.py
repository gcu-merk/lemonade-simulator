#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lemonade Stand Simulator - Human vs AI with MEU Analysis and Bayesian Networks

A business simulation game with complete Bayesian network modeling and MEU analysis.
Includes visualization of decision networks, probability tables, and MEU calculations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass
import itertools
from scipy.stats import beta
import networkx as nx

# ── Simulation Constants ──────────────────────────────────────────────────────
STARTING_CAPITAL: float = 20.0
STARTING_REPUTATION: float = 5.0
TRAFFIC_BASE_CUSTOMERS: Dict[str, int] = {'low': 50, 'medium': 150, 'high': 300}
PRICE_MIN: float = 0.50
PRICE_MAX: float = 2.00
PRICE_STEP: float = 0.25
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BayesianNode:
    """Represents a node in the Bayesian network"""
    name: str
    states: List[str]
    parents: List[str]
    cpt: Dict[Tuple, Dict[str, float]]  # Conditional Probability Table

    def get_probability(self, state: str, parent_states: Dict[str, str] = None) -> float:
        """Get probability of a state given parent states"""
        if parent_states is None:
            parent_states = {}

        # Create key from parent states
        key = tuple(parent_states.get(parent, None) for parent in self.parents)
        if not self.parents:
            key = ()

        return self.cpt.get(key, {}).get(state, 0.0)

@dataclass
class MEUDecision:
    """Data class to store MEU analysis results"""
    location: str
    recipe: str
    price: float
    quantity: int
    expected_utility: float
    confidence: float
    reasoning: List[str]
    bayes_analysis: Dict[str, Any]

@dataclass
class PlayerState:
    """Data class to track each player's state"""
    money: float
    reputation: float
    daily_logs: List[Dict[str, Any]]
    decision_logs: List[Dict[str, Any]]

class BayesianNetwork:
    """Bayesian Network for lemonade stand decision making"""

    def __init__(self):
        self.nodes = {}
        self._build_network()

    def _build_network(self):
        """Build the complete Bayesian network structure"""

        # 1. Weather Node (root node)
        weather_cpt = {
            (): {'sunny': 0.4, 'cloudy': 0.3, 'rainy': 0.2, 'hot': 0.1}
        }
        self.nodes['weather'] = BayesianNode('weather',
                                           ['sunny', 'cloudy', 'rainy', 'hot'],
                                           [], weather_cpt)

        # 2. Forecast Node (depends on weather)
        forecast_cpt = {}
        accuracy = 0.6  # Base forecast accuracy
        for actual in ['sunny', 'cloudy', 'rainy', 'hot']:
            forecast_prob = {}
            for predicted in ['sunny', 'cloudy', 'rainy', 'hot']:
                if actual == predicted:
                    forecast_prob[predicted] = accuracy
                else:
                    forecast_prob[predicted] = (1 - accuracy) / 3
            forecast_cpt[(actual,)] = forecast_prob

        self.nodes['forecast'] = BayesianNode('forecast',
                                            ['sunny', 'cloudy', 'rainy', 'hot'],
                                            ['weather'], forecast_cpt)

        # 3. Traffic Node (depends on weather and location)
        traffic_cpt = {}
        # For each weather-location combination
        weather_location_effects = {
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

        for (weather, location), probs in weather_location_effects.items():
            traffic_cpt[(weather, location)] = probs

        self.nodes['traffic'] = BayesianNode('traffic',
                                           ['low', 'medium', 'high'],
                                           ['weather', 'location'], traffic_cpt)

        # 4. Demand Node (depends on traffic, price, quality, competition)
        # Simplified demand model
        self.nodes['demand'] = BayesianNode('demand',
                                          ['very_low', 'low', 'medium', 'high', 'very_high'],