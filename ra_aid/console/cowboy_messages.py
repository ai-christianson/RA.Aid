import random

COWBOY_MESSAGES = [
    "Yeeehaw! 🤠",
    "Yippee ki yay! 🤠",
    "Saddle up partner! 🤠",
    "This ain't my first rodeo! 🤠",
    "Lock and load, partner! 🤠",
    "I'm just a baby 👶",
    "I'll try not to destroy everything 😏",
]


def get_cowboy_message() -> str:
    """Randomly select and return a cowboy message.

    Returns:
        str: A randomly selected cowboy message
    """
    return random.choice(COWBOY_MESSAGES)
