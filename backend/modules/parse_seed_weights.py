def parse_seed_weights(seed_weights):
    """
    Accepts seed weights as string in "12345:0.1,23456:0.2,3456:0.3" format
    Validates them
    If valid: returns as [[12345, 0.1], [23456, 0.2], [3456, 0.3]]
    If invalid: returns False
    """

    # Must be a string
    if not isinstance(seed_weights, str):
        return False
    # String must not be empty
    if len(seed_weights) == 0:
        return False

    pairs = []

    for pair in seed_weights.split(","):
        split_values = pair.split(":")

        # Seed and weight are required
        if len(split_values) != 2:
            return False

        if len(split_values[0]) == 0 or len(split_values[1]) == 1:
            return False

        # Try casting the seed to int and weight to float
        try:
            seed = int(split_values[0])
            weight = float(split_values[1])
        except ValueError:
            return False

        # Seed must be 0 or above
        if not seed >= 0:
            return False

        # Weight must be between 0 and 1
        if not (weight >= 0 and weight <= 1):
            return False

        # This pair is valid
        pairs.append([seed, weight])

    # All pairs are valid
    return pairs
