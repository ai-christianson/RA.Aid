from ra_aid.console.cowboy_messages import COWBOY_MESSAGES, get_cowboy_message


def test_get_cowboy_message_returns_string():
    """Test that get_cowboy_message returns a non-empty string"""
    message = get_cowboy_message()
    assert isinstance(message, str)
    assert len(message) > 0


def test_cowboy_message_contains_emoji():
    """Test that returned message contains at least one of the expected emojis"""
    message = get_cowboy_message()
    expected_emojis = ["🤠", "👶", "😏"]
    assert any(emoji in message for emoji in expected_emojis)


def test_message_from_predefined_list():
    """Test that returned message is from our predefined list"""
    message = get_cowboy_message()
    assert message in COWBOY_MESSAGES
