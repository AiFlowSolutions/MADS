import pytest
import sys
import os

# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.mads.config import configure_llm, is_termination_msg_chat, create_task_folders


# Tests for the configure_llm function

def test_configure_llm_llama3():
    llm_name = "llama3-70b-8192"
    api_key = "your_api_key"
    expected_config = [{
        "model": llm_name,
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": api_key,
        "temperature": 0,
        "cache_seed": None,
    }]
    assert configure_llm(llm_name, api_key) == expected_config

def test_configure_llm_gpt_3_5_turbo():
    llm_name = "gpt-3.5-turbo-0125"
    api_key = "your_api_key"
    expected_config = [{
        "model": llm_name,
        "api_key": api_key,
        "temperature": 0,
        "cache_seed": None,
    }]
    assert configure_llm(llm_name, api_key) == expected_config

def test_configure_llm_invalid_name():
    with pytest.raises(ValueError):
        configure_llm("invalid_llm_name", "your_api_key")


# Tests for the Termination message function

def test_is_termination_msg_chat_empty():
    message = ""
    assert is_termination_msg_chat(message) == True

def test_is_termination_msg_chat_terminate():
    message = "This is a message TERMINATE"
    assert is_termination_msg_chat(message) == True

def test_is_termination_msg_chat_not_terminate():
    message = "This is not a termination message"
    assert is_termination_msg_chat(message) == False

def test_is_termination_msg_chat_dict():
    message = {"content": "TERMINATE"}
    assert is_termination_msg_chat(message) == True

def test_is_termination_msg_chat_dict_empty_content():
    message = {"content": ""}
    assert is_termination_msg_chat(message) == True

# Tests for the create folder function

def test_create_task_folders(mocker):
    # Mock os.makedirs to prevent actual folder creation
    mock_makedirs = mocker.patch("os.makedirs")

    create_task_folders()

    # Assert that os.makedirs was called with the correct parameters
    mock_makedirs.assert_any_call('tasks', exist_ok=True)
    mock_makedirs.assert_any_call(os.path.join('tasks', 'datasets'), exist_ok=True)
    mock_makedirs.assert_any_call(os.path.join('tasks', 'generated_files'), exist_ok=True)
    assert mock_makedirs.call_count == 3

def test_create_task_folders_exception(mocker):
    # Mock os.makedirs to raise an exception
    mock_makedirs = mocker.patch("os.makedirs", side_effect=Exception("Test error"))

    # Mock print to verify error message
    mock_print = mocker.patch("builtins.print")

    create_task_folders()

    # Assert that the error message was printed
    mock_print.assert_called_with("An error occurred: Test error")