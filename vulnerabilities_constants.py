# IMPORTANT: This is the canonical, fixed order used everywhere.
KEYS = [
    "Access Control",
    "Arithmetic",
    "Bad Randomness",
    "Denial Of Service",
    "Front Running",
    "Reentrancy",
    "Short Addresses",
    "Time Manipulation",
    "Unchecked Low Level Calls"
]
# Mapping from dataset category names to KEYS.
CATEGORY_TO_KEY = {
    "access_control": "Access Control",
    "arithmetic": "Arithmetic",
    "bad_randomness": "Bad Randomness",
    "denial_of_service": "Denial Of Service",
    "front_running": "Front Running",
    "reentrancy": "Reentrancy",
    "short_addresses": "Short Addresses",
    "time_manipulation": "Time Manipulation",
    "unchecked_low_level_calls": "Unchecked Low Level Calls",
}


def expected_map(category):
    """
    Given a vulnerability category (e.g. 'reentrancy'),
    return the perfect expected output map:

        {
            "Access Control": 0,
            "Arithmetic": 0,
            "Bad Randomness": 0,
            ...
            "Reentrancy": 1,
            ...
        }
    """
    out = {k: 0 for k in KEYS}
    expected_key = CATEGORY_TO_KEY[category]
    out[expected_key] = 1
    return out