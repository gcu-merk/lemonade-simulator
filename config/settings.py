# -*- coding: utf-8 -*-
"""
Settings for Lemonade Stand Simulator

This module contains configurable game settings that can be modified at runtime.
Unlike constants.py, these values can be changed during gameplay or through
configuration files/environment variables.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# ── Enums for Setting Options ──────────────────────────────────────────────
class DifficultyLevel(Enum):
    """Game difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class GameMode(Enum):
    """Game modes"""
    CLASSIC = "classic"           # Human vs AI
    HUMAN_ONLY = "human_only"     # Single player
    AI_ONLY = "ai_only"          # AI demonstration
    TUTORIAL = "tutorial"         # Guided tutorial
    SANDBOX = "sandbox"           # Free play with cheats

class AIPersonality(Enum):
    """AI behavior styles"""
    OPTIMAL = "optimal"           # Pure MEU optimization
    CONSERVATIVE = "conservative" # Risk-averse decisions
    AGGRESSIVE = "aggressive"     # High-risk, high-reward
    RANDOM = "random"            # Random decisions for testing

# ── Main Settings Class ────────────────────────────────────────────────────
@dataclass
class GameSettings:
    """Configurable game settings"""
    
    # ── Core Game Settings ─────────────────────────────────────────────────
    game_mode: GameMode = GameMode.CLASSIC
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    num_days: int = 7
    enable_weather_forecast: bool = True
    enable_information_market: bool = True
    
    # ── Player Settings ────────────────────────────────────────────────────
    starting_capital_multiplier: float = 1.0  # Multiplies base starting capital
    starting_reputation_bonus: float = 0.0    # Added to base starting reputation
    bankruptcy_protection: bool = False        # Prevent bankruptcy in tutorial mode
    
    # ── AI Settings ────────────────────────────────────────────────────────
    ai_personality: AIPersonality = AIPersonality.OPTIMAL
    ai_info_budget_ratio: float = 0.15        # Percentage of money AI spends on info
    ai_risk_tolerance: float = 1.0            # Multiplier for risk assessment
    ai_decision_speed: float = 1.0            # Simulation delay multiplier
    
    # ── Economy Settings ───────────────────────────────────────────────────
    price_volatility: float = 1.0             # Market price sensitivity
    competition_intensity: float = 1.0        # Competition effect multiplier
    weather_impact: float = 1.0               # Weather effect multiplier
    customer_randomness: float = 1.0          # Customer demand variability
    
    # ── Information Market Settings ────────────────────────────────────────
    info_cost_multiplier: float = 1.0         # Multiplies all information costs
    forecast_base_accuracy: float = 0.6       # Base weather forecast accuracy
    premium_forecast_accuracy: float = 0.9    # Premium weather forecast accuracy
    
    # ── Display and UI Settings ────────────────────────────────────────────
    show_detailed_analysis: bool = True       # Show MEU analysis by default
    show_bayesian_network: bool = False       # Show network diagram by default
    auto_continue: bool = False               # Auto-advance days (for AI-only mode)
    visualization_enabled: bool = True        # Enable matplotlib visualizations
    color_scheme: str = "default"             # Color scheme for visualizations
    
    # ── Debugging and Development ──────────────────────────────────────────
    debug_mode: bool = False                  # Enable debug output
    log_decisions: bool = True                # Log all decisions to file
    save_game_data: bool = False              # Save game data for analysis
    enable_cheats: bool = False               # Enable cheat commands
    
    # ── Advanced Settings ──────────────────────────────────────────────────
    bayesian_inference_method: str = "exact" # "exact" or "approximate"
    meu_calculation_precision: int = 3        # Decimal places for MEU calculations
    enable_reputation_system: bool = True     # Enable reputation mechanics
    enable_waste_penalty: bool = True         # Penalize unsold inventory
    
    # ── Tutorial Settings ──────────────────────────────────────────────────
    tutorial_hints: bool = True               # Show helpful hints
    tutorial_step_by_step: bool = True        # Pause between tutorial steps
    tutorial_skip_ai_turn: bool = False       # Skip AI demonstrations
    
    # ── Difficulty Modifiers ───────────────────────────────────────────────
    difficulty_modifiers: Dict[str, float] = field(default_factory=lambda: {
        "customer_boost": 1.0,
        "competition_reduction": 1.0,
        "info_cost_reduction": 1.0,
        "weather_predictability": 1.0
    })

# ── Global Settings Instance ───────────────────────────────────────────────
_settings_instance: Optional[GameSettings] = None

def get_settings() -> GameSettings:
    """Get the global settings instance"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = GameSettings()
        load_settings_from_env()
    return _settings_instance

def reset_settings() -> GameSettings:
    """Reset settings to default values"""
    global _settings_instance
    _settings_instance = GameSettings()
    return _settings_instance

def load_settings_from_env() -> None:
    """Load settings from environment variables"""
    settings = get_settings()
    
    # Core game settings
    if os.getenv('LEMONADE_GAME_MODE'):
        try:
            settings.game_mode = GameMode(os.getenv('LEMONADE_GAME_MODE'))
        except ValueError:
            pass
    
    if os.getenv('LEMONADE_DIFFICULTY'):
        try:
            settings.difficulty = DifficultyLevel(os.getenv('LEMONADE_DIFFICULTY'))
        except ValueError:
            pass
    
    if os.getenv('LEMONADE_NUM_DAYS'):
        try:
            settings.num_days = int(os.getenv('LEMONADE_NUM_DAYS'))
        except ValueError:
            pass
    
    # Boolean settings
    bool_env_vars = {
        'LEMONADE_DEBUG_MODE': 'debug_mode',
        'LEMONADE_ENABLE_CHEATS': 'enable_cheats',
        'LEMONADE_AUTO_CONTINUE': 'auto_continue',
        'LEMONADE_SHOW_ANALYSIS': 'show_detailed_analysis',
        'LEMONADE_SHOW_NETWORK': 'show_bayesian_network',
    }
    
    for env_var, setting_name in bool_env_vars.items():
        if os.getenv(env_var):
            setattr(settings, setting_name, os.getenv(env_var).lower() == 'true')
    
    # Float settings
    float_env_vars = {
        'LEMONADE_STARTING_CAPITAL_MULT': 'starting_capital_multiplier',
        'LEMONADE_AI_INFO_BUDGET': 'ai_info_budget_ratio',
        'LEMONADE_WEATHER_IMPACT': 'weather_impact',
        'LEMONADE_COMPETITION_INTENSITY': 'competition_intensity',
    }
    
    for env_var, setting_name in float_env_vars.items():
        if os.getenv(env_var):
            try:
                setattr(settings, setting_name, float(os.getenv(env_var)))
            except ValueError:
                pass

def apply_difficulty_settings(difficulty: DifficultyLevel) -> None:
    """Apply difficulty-specific settings"""
    settings = get_settings()
    settings.difficulty = difficulty
    
    if difficulty == DifficultyLevel.EASY:
        settings.difficulty_modifiers = {
            "customer_boost": 1.3,          # 30% more customers
            "competition_reduction": 0.7,   # 30% less competition
            "info_cost_reduction": 0.8,     # 20% cheaper information
            "weather_predictability": 1.2   # More predictable weather
        }
        settings.starting_capital_multiplier = 1.2
        settings.bankruptcy_protection = True
        settings.tutorial_hints = True
        
    elif difficulty == DifficultyLevel.MEDIUM:
        settings.difficulty_modifiers = {
            "customer_boost": 1.0,
            "competition_reduction": 1.0,
            "info_cost_reduction": 1.0,
            "weather_predictability": 1.0
        }
        settings.starting_capital_multiplier = 1.0
        settings.bankruptcy_protection = False
        
    elif difficulty == DifficultyLevel.HARD:
        settings.difficulty_modifiers = {
            "customer_boost": 0.8,          # 20% fewer customers
            "competition_reduction": 1.3,   # 30% more competition
            "info_cost_reduction": 1.2,     # 20% more expensive information
            "weather_predictability": 0.8   # Less predictable weather
        }
        settings.starting_capital_multiplier = 0.8
        settings.forecast_base_accuracy = 0.5
        settings.tutorial_hints = False
        
    elif difficulty == DifficultyLevel.EXPERT:
        settings.difficulty_modifiers = {
            "customer_boost": 0.6,          # 40% fewer customers
            "competition_reduction": 1.5,   # 50% more competition
            "info_cost_reduction": 1.5,     # 50% more expensive information
            "weather_predictability": 0.6   # Much less predictable weather
        }
        settings.starting_capital_multiplier = 0.6
        settings.forecast_base_accuracy = 0.4
        settings.enable_information_market = True  # Force information buying
        settings.tutorial_hints = False

def configure_ai_personality(personality: AIPersonality) -> None:
    """Configure AI behavior based on personality"""
    settings = get_settings()
    settings.ai_personality = personality
    
    if personality == AIPersonality.CONSERVATIVE:
        settings.ai_risk_tolerance = 0.6
        settings.ai_info_budget_ratio = 0.20  # Spend more on information
        
    elif personality == AIPersonality.AGGRESSIVE:
        settings.ai_risk_tolerance = 1.8
        settings.ai_info_budget_ratio = 0.05  # Spend less on information, more on inventory
        
    elif personality == AIPersonality.RANDOM:
        settings.ai_risk_tolerance = 1.0
        settings.ai_info_budget_ratio = 0.10
        
    # OPTIMAL uses default values

def get_effective_value(base_value: float, setting_name: str) -> float:
    """Get effective value after applying difficulty modifiers"""
    settings = get_settings()
    modifier = settings.difficulty_modifiers.get(setting_name, 1.0)
    return base_value * modifier

def is_feature_enabled(feature: str) -> bool:
    """Check if a specific feature is enabled"""
    settings = get_settings()
    
    feature_map = {
        'weather_forecast': settings.enable_weather_forecast,
        'information_market': settings.enable_information_market,
        'reputation_system': settings.enable_reputation_system,
        'waste_penalty': settings.enable_waste_penalty,
        'visualizations': settings.visualization_enabled,
        'debug': settings.debug_mode,
        'cheats': settings.enable_cheats,
        'tutorial_hints': settings.tutorial_hints,
    }
    
    return feature_map.get(feature, False)

def get_display_settings() -> Dict[str, Any]:
    """Get display-related settings"""
    settings = get_settings()
    return {
        'show_detailed_analysis': settings.show_detailed_analysis,
        'show_bayesian_network': settings.show_bayesian_network,
        'auto_continue': settings.auto_continue,
        'visualization_enabled': settings.visualization_enabled,
        'color_scheme': settings.color_scheme,
        'tutorial_hints': settings.tutorial_hints,
    }

def save_settings_to_file(filepath: str = "game_settings.ini") -> None:
    """Save current settings to a configuration file"""
    settings = get_settings()
    
    config_content = f"""[Game]
mode = {settings.game_mode.value}
difficulty = {settings.difficulty.value}
num_days = {settings.num_days}
enable_weather_forecast = {settings.enable_weather_forecast}
enable_information_market = {settings.enable_information_market}

[Player]
starting_capital_multiplier = {settings.starting_capital_multiplier}
starting_reputation_bonus = {settings.starting_reputation_bonus}
bankruptcy_protection = {settings.bankruptcy_protection}

[AI]
personality = {settings.ai_personality.value}
info_budget_ratio = {settings.ai_info_budget_ratio}
risk_tolerance = {settings.ai_risk_tolerance}

[Economy]
price_volatility = {settings.price_volatility}
competition_intensity = {settings.competition_intensity}
weather_impact = {settings.weather_impact}
customer_randomness = {settings.customer_randomness}

[Display]
show_detailed_analysis = {settings.show_detailed_analysis}
show_bayesian_network = {settings.show_bayesian_network}
auto_continue = {settings.auto_continue}
visualization_enabled = {settings.visualization_enabled}

[Debug]
debug_mode = {settings.debug_mode}
log_decisions = {settings.log_decisions}
save_game_data = {settings.save_game_data}
enable_cheats = {settings.enable_cheats}
"""
    
    with open(filepath, 'w') as f:
        f.write(config_content)

def load_settings_from_file(filepath: str = "game_settings.ini") -> bool:
    """Load settings from a configuration file"""
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(filepath)
        
        settings = get_settings()
        
        # Load game settings
        if 'Game' in config:
            game_section = config['Game']
            if 'mode' in game_section:
                settings.game_mode = GameMode(game_section['mode'])
            if 'difficulty' in game_section:
                settings.difficulty = DifficultyLevel(game_section['difficulty'])
            if 'num_days' in game_section:
                settings.num_days = int(game_section['num_days'])
            if 'enable_weather_forecast' in game_section:
                settings.enable_weather_forecast = game_section.getboolean('enable_weather_forecast')
            if 'enable_information_market' in game_section:
                settings.enable_information_market = game_section.getboolean('enable_information_market')
        
        # Load other sections similarly...
        # (Implementation would continue for all sections)
        
        return True
    except Exception as e:
        print(f"Error loading settings from {filepath}: {e}")
        return False

# ── Preset Configurations ──────────────────────────────────────────────────
PRESET_CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "beginner": {
        "game_mode": GameMode.TUTORIAL,
        "difficulty": DifficultyLevel.EASY,
        "num_days": 3,
        "bankruptcy_protection": True,
        "tutorial_hints": True,
        "show_detailed_analysis": True,
    },
    
    "standard": {
        "game_mode": GameMode.CLASSIC,
        "difficulty": DifficultyLevel.MEDIUM,
        "num_days": 7,
        "show_detailed_analysis": True,
        "visualization_enabled": True,
    },
    
    "challenge": {
        "game_mode": GameMode.CLASSIC,
        "difficulty": DifficultyLevel.HARD,
        "num_days": 10,
        "ai_personality": AIPersonality.OPTIMAL,
        "weather_impact": 1.5,
        "competition_intensity": 1.3,
    },
    
    "expert": {
        "game_mode": GameMode.CLASSIC,
        "difficulty": DifficultyLevel.EXPERT,
        "num_days": 14,
        "ai_personality": AIPersonality.OPTIMAL,
        "starting_capital_multiplier": 0.5,
        "info_cost_multiplier": 2.0,
    },
    
    "demo": {
        "game_mode": GameMode.AI_ONLY,
        "difficulty": DifficultyLevel.MEDIUM,
        "num_days": 5,
        "auto_continue": True,
        "ai_decision_speed": 0.5,
        "show_detailed_analysis": True,
    }
}

def apply_preset(preset_name: str) -> bool:
    """Apply a preset configuration"""
    if preset_name not in PRESET_CONFIGURATIONS:
        return False
    
    settings = get_settings()
    preset = PRESET_CONFIGURATIONS[preset_name]
    
    for key, value in preset.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    
    # Apply difficulty-specific settings if difficulty was changed
    if 'difficulty' in preset:
        apply_difficulty_settings(preset['difficulty'])
    
    return True