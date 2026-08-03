from orchestrator.telegram import parse_command


def test_parse_command_with_bot_suffix() -> None:
    command = parse_command("/approve@my_bot CHANGE-123")
    assert command is not None
    assert command.name == "/approve"
    assert command.args == "CHANGE-123"


def test_plain_text_is_not_command() -> None:
    assert parse_command("hello") is None


def test_command_arguments_are_trimmed() -> None:
    command = parse_command("/reject   CHANGE-1 reason")
    assert command is not None
    assert command.args == "CHANGE-1 reason"
