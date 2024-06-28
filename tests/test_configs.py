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
