# -*- coding: utf-8 -*-
"""
Lemonade Stand Simulator - Configuration Package

This package contains all configuration-related modules for the lemonade stand simulator.
It provides both static constants and dynamic settings for the game.

Modules:
    constants: Static game constants that never change during runtime
    settings: Dynamic game settings that can be modified during gameplay

Usage:
    # Import specific constants
    from config.constants import STARTING_CAPITAL, RECIPES, LOCATIONS
    
    # Import settings system
    from config.settings import get_settings, DifficultyLevel, GameMode
    
    # Or import everything from config
    from config import constants, settings
"""

# Version information
__version__ = "1.0.0"
__author__ = "Lemonade Stand Simulator Team"

# Import key classes and functions for easy access
from .constants import (
    # Core game constants
    STARTING_CAPITAL,
    STARTING_REPUTATION,
    DEFAULT_GAME_DAYS,
    
    # Game configuration dictionaries
    RECIPES,
    LOCATIONS,
    INFO_MARKET,
    WEATHER_CONDITIONS,
    WEATHER_PROBABILITIES,
    
    # Bayesian network states
    BAYES_STATES,
    
    # Display constants
    GAME_TITLE,
    GAME_SUBTITLE,
    DISPLAY_SETTINGS,
    
    # Validation constants
    VALID_INPUTS,
    PROMPTS,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
)

from .settings import (
    # Main settings class
    GameSettings,
    get_settings,
    reset_settings,
    
    # Enums
    DifficultyLevel,
    GameMode,
    AIPersonality,
    
    # Configuration functions
    apply_difficulty_settings,
    configure_ai_personality,
    apply_preset,
    
    # Utility functions
    is_feature_enabled,
    get_effective_value,
    get_display_settings,
    
    # File operations
    save_settings_to_file,
    load_settings_from_file,
    load_settings_from_env,
    
    # Preset configurations
    PRESET_CONFIGURATIONS,
)

# Convenience aliases for common operations
def get_game_config():
    """
    Get a complete game configuration dictionary.
    
    Returns:
        dict: Combined constants and current settings
    """
    settings = get_settings()
    return {
        'constants': {
            'starting_capital': STARTING_CAPITAL * settings.starting_capital_multiplier,
            'starting_reputation': STARTING_REPUTATION + settings.starting_reputation_bonus,
            'recipes': RECIPES,
            'locations': LOCATIONS,
            'info_market': INFO_MARKET,
            'weather_conditions': WEATHER_CONDITIONS,
        },
        'settings': {
            'game_mode': settings.game_mode,
            'difficulty': settings.difficulty,
            'num_days': settings.num_days,
            'ai_personality': settings.ai_personality,
            'debug_mode': settings.debug_mode,
        },
        'modifiers': settings.difficulty_modifiers,
    }

def validate_configuration():
    """
    Validate the current configuration for consistency.
    
    Returns:
        tuple: (is_valid: bool, issues: list)
    """
    issues = []
    settings = get_settings()
    
    # Check for valid ranges
    if settings.num_days < 1 or settings.num_days > 100:
        issues.append("Number of days must be between 1 and 100")
    
    if settings.starting_capital_multiplier < 0.1 or settings.starting_capital_multiplier > 10.0:
        issues.append("Starting capital multiplier must be between 0.1 and 10.0")
    
    if settings.ai_info_budget_ratio < 0.0 or settings.ai_info_budget_ratio > 1.0:
        issues.append("AI info budget ratio must be between 0.0 and 1.0")
    
    # Check for conflicting settings
    if settings.game_mode == GameMode.TUTORIAL and not settings.tutorial_hints:
        issues.append("Tutorial mode should have tutorial hints enabled")
    
    if settings.game_mode == GameMode.AI_ONLY and not settings.auto_continue:
        issues.append("AI-only mode should have auto-continue enabled")
    
    # Check for features that require other features
    if settings.show_detailed_analysis and not settings.visualization_enabled:
        issues.append("Detailed analysis requires visualizations to be enabled")
    
    return len(issues) == 0, issues

def get_effective_constants():
    """
    Get constants modified by current settings.
    
    Returns:
        dict: Constants adjusted for current difficulty and settings
    """
    settings = get_settings()
    
    return {
        'starting_capital': STARTING_CAPITAL * settings.starting_capital_multiplier,
        'starting_reputation': STARTING_REPUTATION + settings.starting_reputation_bonus,
        'forecast_accuracy': {
            'base': settings.forecast_base_accuracy,
            'premium': settings.premium_forecast_accuracy,
        },
        'info_costs': {
            name: data['cost'] * settings.info_cost_multiplier 
            for name, data in INFO_MARKET.items()
        },
        'weather_impact_multiplier': settings.weather_impact,
        'competition_intensity_multiplier': settings.competition_intensity,
    }

def setup_game_configuration(preset: str = None, **kwargs):
    """
    Setup game configuration with optional preset and custom overrides.
    
    Args:
        preset: Name of preset configuration to apply
        **kwargs: Custom setting overrides
    
    Returns:
        GameSettings: Configured settings object
    """
    # Reset to defaults
    settings = reset_settings()
    
    # Apply preset if provided
    if preset:
        if not apply_preset(preset):
            raise ValueError(f"Unknown preset: {preset}")
    
    # Apply custom overrides
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            raise ValueError(f"Unknown setting: {key}")
    
    # Validate configuration
    is_valid, issues = validate_configuration()
    if not is_valid:
        raise ValueError(f"Invalid configuration: {', '.join(issues)}")
    
    return settings

def print_configuration_summary():
    """Print a summary of the current configuration."""
    settings = get_settings()
    config = get_game_config()
    
    print("=" * 60)
    print("LEMONADE STAND SIMULATOR - CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"Game Mode: {settings.game_mode.value.title()}")
    print(f"Difficulty: {settings.difficulty.value.title()}")
    print(f"Number of Days: {settings.num_days}")
    print(f"AI Personality: {settings.ai_personality.value.title()}")
    
    print(f"\nStarting Conditions:")
    print(f"  Capital: ${config['constants']['starting_capital']:.2f}")
    print(f"  Reputation: {config['constants']['starting_reputation']:.1f}/10")
    
    print(f"\nFeatures Enabled:")
    features = [
        ('Weather Forecast', settings.enable_weather_forecast),
        ('Information Market', settings.enable_information_market),
        ('Reputation System', settings.enable_reputation_system),
        ('Waste Penalty', settings.enable_waste_penalty),
        ('Visualizations', settings.visualization_enabled),
        ('Debug Mode', settings.debug_mode),
        ('Tutorial Hints', settings.tutorial_hints),
    ]
    
    for feature_name, enabled in features:
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature_name}")
    
    if settings.difficulty_modifiers:
        print(f"\nDifficulty Modifiers:")
        for modifier, value in settings.difficulty_modifiers.items():
            print(f"  {modifier.replace('_', ' ').title()}: {value:.1f}x")
    
    print("=" * 60)

# Package metadata
__all__ = [
    # Constants
    'STARTING_CAPITAL', 'STARTING_REPUTATION', 'DEFAULT_GAME_DAYS',
    'RECIPES', 'LOCATIONS', 'INFO_MARKET', 'WEATHER_CONDITIONS',
    'WEATHER_PROBABILITIES', 'BAYES_STATES', 'GAME_TITLE', 'GAME_SUBTITLE',
    'DISPLAY_SETTINGS', 'VALID_INPUTS', 'PROMPTS', 'ERROR_MESSAGES', 'SUCCESS_MESSAGES',
    
    # Settings classes and enums
    'GameSettings', 'DifficultyLevel', 'GameMode', 'AIPersonality',
    
    # Settings functions
    'get_settings', 'reset_settings', 'apply_difficulty_settings',
    'configure_ai_personality', 'apply_preset', 'is_feature_enabled',
    'get_effective_value', 'get_display_settings', 'save_settings_to_file',
    'load_settings_from_file', 'load_settings_from_env', 'PRESET_CONFIGURATIONS',
    
    # Convenience functions
    'get_game_config', 'validate_configuration', 'get_effective_constants',
    'setup_game_configuration', 'print_configuration_summary',
]

# Package initialization
def _initialize_package():
    """Initialize the config package with default settings."""
    # Load settings from environment if available
    load_settings_from_env()
    
    # Validate initial configuration
    is_valid, issues = validate_configuration()
    if not is_valid and not get_settings().debug_mode:
        # Only warn in non-debug mode, don't fail
        print(f"Configuration warnings: {', '.join(issues)}")

# Initialize when package is imported
_initialize_package()