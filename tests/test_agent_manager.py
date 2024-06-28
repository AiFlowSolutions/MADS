import pytest

from src.mads.agent_manager import AgentManager

def test_data_initialization():

    llm_config = {"param1": "value1"}  # random configuration
    data_name_with_csv = "data.csv"
    agent_manager_with_csv = AgentManager(llm_config, data_name_with_csv)
    assert agent_manager_with_csv.data == "data"