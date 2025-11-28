CATEGORIES = [
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

KEYS_TO_CATEGORIES = {
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


def expected_map(category_key):
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
    output_map = {k: 0 for k in CATEGORIES} # All zeros.
    expected_key = KEYS_TO_CATEGORIES[category_key]
    output_map[expected_key] = 1
    return output_map