"""Shared debug counter defaults and helper utilities."""

DEFAULT_DEBUG_COUNTERS = {
    "holdings_seen": 0,
    "cc_signals": 0,
    "cc_signal_missing": 0,
    "cc_confidence_block": 0,
    "cc_share_block": 0,
    "cc_option_filter_block": 0,
    "sp_signals": 0,
    "put_block": 0,
    "sp_held_block": 0,
    "sp_assignment_cooldown_block": 0,
    "sp_weak_filter_block": 0,
    "assigned_stock_track": 0,
    "assigned_stock_exit": 0,
    "immediate_cc": 0,
    "stock_buy": 0,
    "stock_sell": 0,
    "no_suitable_options": 0,
}


def increment_debug_counter(algo, key, amount=1):
    if key not in DEFAULT_DEBUG_COUNTERS:
        raise ValueError(f"Unknown debug counter: {key}")

    counters = getattr(algo, "debug_counters", None)
    if not isinstance(counters, dict):
        counters = dict(DEFAULT_DEBUG_COUNTERS)
        setattr(algo, "debug_counters", counters)
    else:
        for counter_key, default_value in DEFAULT_DEBUG_COUNTERS.items():
            counters.setdefault(counter_key, default_value)

    counters[key] = counters.get(key, 0) + amount
