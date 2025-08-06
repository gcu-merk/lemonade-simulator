"""
Data classes for the Lemonade Stand Simulator
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any

@dataclass
class MEUDecision:
    """Data class to store MEU analysis results"""
    location: str
    recipe: str
    price: float
    quantity: int
    expected_utility: float
    confidence: float
    total_cost: float
    reasoning: List[str]
    all_analyses: List[Dict[str, Any]]
    bayes_analysis: Dict[str, Any]

@dataclass
class PlayerState:
    """Data class to track each player's state"""
    money: float
    reputation: float
    daily_logs: List[Dict[str, Any]]
    decision_logs: List[Dict[str, Any]]

@dataclass
class GameResult:
    """Data class to store daily game results"""
    success: bool
    profit: float
    cups_sold: int
    revenue: float
    reputation_change: float
    customers: int
    quality: float
    waste: int
    message: str = ""
