# -*- coding: utf-8 -*-
"""
Constants for Lemonade Stand Simulator

This module contains all game constants and configuration values.
These values should not be changed during runtime.
"""

from typing import Dict, Any

# ── Game Setup Constants ──────────────────────────────────────────────────
STARTING_CAPITAL: float = 20.0
STARTING_REPUTATION: float = 5.0
DEFAULT_GAME_DAYS: int = 7

# ── Pricing Constants ──────────────────────────────────────────────────────
PRICE_MIN: float = 0.50
PRICE_MAX: float = 2.00
PRICE_STEP: float = 0.25

# ── Traffic Base Customers ─────────────────────────────────────────────────
TRAFFIC_BASE_CUSTOMERS: Dict[str, int] = {
    'low': 50,
    'medium': 150,
    'high': 300
}

# ── Recipe Configurations ──────────────────────────────────────────────────
RECIPES: Dict[str, Dict[str, Any]] = {
    'basic': {
        'cost': 0.30,
        'base_quality': 5,
        'appeal': 1.0,
        'name': 'Basic Lemonade',
        'description': 'Simple, classic lemonade'
    },
    'premium': {
        'cost': 0.60,
        'base_quality': 8,
        'appeal': 1.3,
        'name': 'Premium Lemonade',
        'description': 'High-quality ingredients with premium taste'
    },
    'organic': {
        'cost': 0.80,
        'base_quality': 7,
        'appeal': 1.2,
        'name': 'Organic Lemonade',
        'description': 'All-natural, organic ingredients'
    },
    'sugar_free': {
        'cost': 0.50,
        'base_quality': 6,
        'appeal': 0.9,
        'name': 'Sugar-Free Lemonade',
        'description': 'Healthy alternative with natural sweeteners'
    }
}

# ── Location Configurations ────────────────────────────────────────────────
LOCATIONS: Dict[str, Dict[str, Any]] = {
    'park': {
        'base_traffic': 200,
        'weather_bonus': 1.5,
        'name': 'City Park',
        'description': 'Popular outdoor location, weather dependent',
        'local_events': {
            1: 'Food Truck Festival',
            3: 'Outdoor Concert',
            5: 'Charity Run'
        }
    },
    'school': {
        'base_traffic': 120,
        'weather_bonus': 1.2,
        'name': 'Elementary School',
        'description': 'Steady customer base, less weather dependent',
        'local_events': {
            2: 'Science Fair',
            4: 'Sports Day',
            6: 'Book Fair'
        }
    },
    'mall': {
        'base_traffic': 180,
        'weather_bonus': 0.8,
        'name': 'Shopping Mall',
        'description': 'Indoor location, consistent traffic',
        'local_events': {
            1: 'Fashion Expo',
            3: 'Tech Showcase',
            7: 'Holiday Sale'
        }
    }
}

# ── Weather Conditions ─────────────────────────────────────────────────────
WEATHER_CONDITIONS: list[str] = ['sunny', 'cloudy', 'rainy', 'hot']

# Weather probability distribution (must sum to 1.0)
WEATHER_PROBABILITIES: Dict[str, float] = {
    'sunny': 0.4,
    'cloudy': 0.3,
    'rainy': 0.2,
    'hot': 0.1
}

# Weather effect multipliers on customer traffic
WEATHER_EFFECTS: Dict[str, float] = {
    'sunny': 1.2,
    'hot': 1.4,
    'cloudy': 0.9,
    'rainy': 0.4
}

# ── Information Market ─────────────────────────────────────────────────────
INFO_MARKET: Dict[str, Dict[str, Any]] = {
    'premium_weather': {
        'cost': 2.0,
        'description': '90% Accurate weather forecast',
        'accuracy_boost': 0.3
    },
    'local_events': {
        'cost': 1.5,
        'description': 'Information about local events affecting traffic',
        'traffic_boost': 0.15
    },
    'competition_intel': {
        'cost': 3.0,
        'description': 'Complete competitor analysis and strategies',
        'competition_insight': True
    }
}

# ── Competition Default Stats ──────────────────────────────────────────────
DEFAULT_COMPETITION_STATS: Dict[str, Dict[str, Any]] = {
    'park': {
        'num_stands': 2,
        'avg_price': 1.25,
        'difficulty': 'medium'
    },
    'school': {
        'num_stands': 1,
        'avg_price': 1.00,
        'difficulty': 'easy'
    },
    'mall': {
        'num_stands': 3,
        'avg_price': 1.50,
        'difficulty': 'hard'
    }
}

# ── Bayesian Network State Definitions ────────────────────────────────────
BAYES_STATES: Dict[str, list[str]] = {
    'weather': ['sunny', 'cloudy', 'rainy', 'hot'],
    'forecast': ['sunny', 'cloudy', 'rainy', 'hot'],
    'traffic': ['low', 'medium', 'high'],
    'demand': ['very_low', 'low', 'medium', 'high', 'very_high'],
    'sales': ['very_low', 'low', 'medium', 'high', 'very_high'],
    'profit': ['loss', 'break_even', 'low_profit', 'good_profit', 'high_profit'],
    'price_level': ['low', 'medium', 'high'],
    'quantity_level': ['very_low', 'low', 'medium', 'high', 'very_high'],
    'quality_level': ['low', 'medium', 'high'],
    'competition_level': ['low', 'medium', 'high'],
    'cost_level': ['very_low', 'low', 'medium', 'high', 'very_high']
}

# ── Forecast Accuracy Settings ────────────────────────────────────────────
BASE_FORECAST_ACCURACY: float = 0.6
PREMIUM_FORECAST_ACCURACY: float = 0.9

# ── MEU Analysis Settings ──────────────────────────────────────────────────
# AI information budget as percentage of available money
AI_INFO_BUDGET_RATIO: float = 0.15

# Utility function weights
UTILITY_WEIGHTS: Dict[str, float] = {
    'profit_weight': 1.0,        # α in utility function
    'reputation_weight': 2.0     # β in utility function
}

# ── Game Mechanics Constants ───────────────────────────────────────────────
# Reputation bounds
REPUTATION_MIN: float = 1.0
REPUTATION_MAX: float = 10.0

# Reputation change constants
REPUTATION_CHANGES: Dict[str, float] = {
    'no_sales': -0.3,
    'poor_sales': -0.1,      # < 50% of quantity sold
    'good_sales': 0.1,       # 50-89% of quantity sold
    'excellent_sales': 0.3,  # >= 90% of quantity sold
    'high_quality_bonus': 0.2,  # Quality >= 8
    'low_quality_penalty': -0.2  # Quality <= 4
}

# Bankruptcy threshold
BANKRUPTCY_THRESHOLD: float = 1.0

# Quality calculation parameters
QUALITY_CALCULATION: Dict[str, float] = {
    'reputation_multiplier': 0.4,  # How much reputation affects quality
    'base_reputation': 5.0         # Neutral reputation level
}

# Price sensitivity parameters
PRICE_SENSITIVITY: Dict[str, float] = {
    'base_multiplier': 2.0,
    'price_coefficient': 0.8,
    'minimum_multiplier': 0.1
}

# Competition effect parameters
COMPETITION_EFFECTS: Dict[str, float] = {
    'base_multiplier': 1.2,
    'stands_penalty': 0.1,
    'minimum_multiplier': 0.5
}

# Randomness parameters
RANDOMNESS_SETTINGS: Dict[str, float] = {
    'customer_std_ratio': 0.15,  # Standard deviation as ratio of mean customers
    'min_customers': 0           # Minimum customers (after randomness)
}

# ── Discretization Thresholds ──────────────────────────────────────────────
# For converting continuous values to discrete Bayesian states
DISCRETIZATION_THRESHOLDS: Dict[str, Dict[str, float]] = {
    'price_level': {
        'low_max': 1.0,
        'medium_max': 1.5
        # high is anything above medium_max
    },
    'quantity_ratio': {
        'very_low_max': 0.2,
        'low_max': 0.4,
        'medium_max': 0.6,
        'high_max': 0.8
        # very_high is anything above high_max
    },
    'quality_level': {
        'low_max': 4.0,
        'medium_max': 7.0
        # high is anything above medium_max
    },
    'cost_level': {
        'very_low_max': 5.0,
        'low_max': 10.0,
        'medium_max': 15.0,
        'high_max': 20.0
        # very_high is anything above high_max
    },
    'traffic_level': {
        'low_max': 100,
        'medium_max': 200
        # high is anything above medium_max
    }
}

# ── Profit State Mapping ───────────────────────────────────────────────────
PROFIT_STATE_VALUES: Dict[str, float] = {
    'loss': -5.0,
    'break_even': 0.0,
    'low_profit': 2.0,
    'good_profit': 8.0,
    'high_profit': 15.0
}

SALES_STATE_VALUES: Dict[str, float] = {
    'very_low': 0.1,
    'low': 0.3,
    'medium': 0.5,
    'high': 0.7,
    'very_high': 0.9
}

# ── Display Constants ──────────────────────────────────────────────────────
DISPLAY_SETTINGS: Dict[str, Any] = {
    'separator_width': 60,
    'separator_char': '=',
    'currency_precision': 2,
    'reputation_precision': 1,
    'utility_precision': 3,
    'percentage_precision': 1
}

# Game title and branding
GAME_TITLE: str = "🍋 LEMONADE STAND SIMULATOR 🍋"
GAME_SUBTITLE: str = "Human vs AI with Bayesian MEU Analysis"

# ── Validation Constants ───────────────────────────────────────────────────
VALID_INPUTS: Dict[str, list[str]] = {
    'yes_no': ['y', 'yes', 'n', 'no'],
    'locations': list(LOCATIONS.keys()),
    'recipes': list(RECIPES.keys()),
    'info_types': list(INFO_MARKET.keys())
}

# Input prompts
PROMPTS: Dict[str, str] = {
    'continue_game': "Press Enter to continue to next day...",
    'view_network': "View Bayesian Network? (y/n): ",
    'view_analysis': "View AI's MEU analysis? (y/n): ",
    'view_performance': "View performance analysis? (y/n): ",
    'buy_info': "Buy information? (type name or 'done'): ",
    'choose_location': "Choose location (park/school/mall): ",
    'choose_recipe': "Choose recipe: ",
    'set_price': f"Set price (${PRICE_MIN:.2f}-${PRICE_MAX:.2f}): $",
    'set_quantity': "How many cups to make? "
}

# ── Error Messages ─────────────────────────────────────────────────────────
ERROR_MESSAGES: Dict[str, str] = {
    'insufficient_funds': "Insufficient funds!",
    'invalid_location': "Invalid location!",
    'invalid_recipe': "Invalid recipe!",
    'invalid_price': f"Price must be between ${PRICE_MIN:.2f} and ${PRICE_MAX:.2f}",
    'invalid_quantity': "Quantity must be between 1 and maximum affordable",
    'invalid_choice': "Invalid choice!",
    'bankruptcy': "💸 PLAYER BANKRUPT! 💸"
}

# ── Success Messages ───────────────────────────────────────────────────────
SUCCESS_MESSAGES: Dict[str, str] = {
    'human_wins': "🎉 HUMAN WINS! 🎉",
    'ai_wins': "🤖 AI WINS! 🤖",
    'tie_game': "🤝 IT'S A TIE! 🤝",
    'purchase_success': "Purchased {item} for ${cost:.2f}"
}

# ── File and Path Constants ────────────────────────────────────────────────
DEFAULT_SAVE_PATH: str = "game_saves"
LOG_FILE_PATH: str = "logs"
VISUALIZATION_SAVE_PATH: str = "visualizations"