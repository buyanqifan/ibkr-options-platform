"""Utility hooks for language-aware components."""

from dash import Input, Output, State, callback_context
from functools import wraps


def with_language(callback_func):
    """Decorator to add language parameter to callbacks.
    
    Usage:
        @with_language
        def my_callback(lang, other_inputs...):
            text = TRANSLATIONS[lang]["my.key"]
            return ...
    """
    @wraps(callback_func)
    def wrapper(*args, **kwargs):
        # Extract language from args (usually first input after self)
        # This is a simplified version - full implementation would need
        # to inject language-store as a State
        return callback_func(*args, **kwargs)
    
    return wrapper


def get_translation(lang: str, key: str, default: str = None) -> str:
    """Get translation for a key in the specified language.
    
    Args:
        lang: Language code ('en' or 'zh')
        key: Translation key (e.g., 'navbar.dashboard')
        default: Default text if key not found
        
    Returns:
        Translated text or default
    """
    from app.i18n import TRANSLATIONS
    
    if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
        return TRANSLATIONS[lang][key]
    
    # Fallback to English
    if key in TRANSLATIONS.get('en', {}):
        return TRANSLATIONS['en'][key]
    
    # Last resort: return key itself or provided default
    return default or key
