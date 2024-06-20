import logging
from backend.agent_manager import AgentManager
from backend.config import llm_choosing, tasks

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatManager:
    """
    This class manages the initiation of chats for a sequence of data science tasks.
    
    Attributes:
        task (function): A function that returns the user's problem.
        datasets (str): The datasets needed for the task.
        agent_manager (AgentManager): An instance of the AgentManager class.
        user (ConversableAgent): The user agent.
        include_chats (list): A list of chat IDs to include in the conversation (i.e. which agents to include).
    """
    def __init__(self, datasets, user_problem, data, model, include_chats=None):
        try:
            self.task = tasks(user_problem)
            self.datasets = datasets
            self.agent_manager = AgentManager(llm_config=llm_choosing(model), data_name=data)
            self.user = self.agent_manager.user_agent()
            self.include_chats = include_chats if include_chats is not None else list(range(1, 7))
        except Exception as e:
            logging.error("Error during initialization", exc_info=True)
            raise

    def initiate_chats(self):
        try:
            carryover_msg = f"The needed data is in {self.datasets}"

            chat_configurations = [
                self._build_chat_config(
                    chat_id,
                    {
                        "summary_method": "reflection_with_llm",
                        **({"carryover": carryover_msg} if chat_id in [1, 2, 3] else {})
                    },
                )
                for chat_id in self.include_chats
            ]

            chat_results = self.user.initiate_chats(chat_configurations)
            return chat_results
        except Exception as e:
            logging.error("Error during chat initiation", exc_info=True)
            raise

    def _get_summary_prompt(self):
        try:
            return {
                1: """
                    Summarize the content and format summary EXACTLY as follows:
                    ---
                    *Data set name*:
                    `Acme Corp`
                    ---
                    *User problem*:
                    `Regression, Classification or Time-series`
                    ---
                    *Target variable*
                    `write here the target variable`
                    ---
                    *Indications to read data*
                    `Write the code that you read the data`
                    """,
                2: """
                    Summarize the content and format summary EXACTLY as follows:
                    ---
                    *Problem*:
                    `Write the machine learning problem`
                    ---
                    *Correlations*:
                    `Relevant correlations`
                    ---
                    *Columns name*:
                    `Columns: column1, column2...`
                    ---               
                    *Relevant insights*:
                    `Useful insights found for the next agents`
                    ---
                    """,
                3: """
                    Summarize the content and format summary EXACTLY as follows:
                    ---
                    *Problem*:
                    `problem written here`
                    ---
                    *Machine learning model to use*:
                    `ML model`
                    ---
                    *Explanation and alternatives*:
                    `Explanation of why that model was chosen`
                    ---
                    """,
                4: """
                    Summarize the content and format summary EXACTLY as follows:
                    ---
                    *Transformations*:
                    `Transformations you've done to the data`
                    ---
                    *Splitting*:
                    `The split you've done and where you save the data`
                    ---
                    **Read the data**
                    `pd.read_csv('X_train.csv')`
                    `pd.read_csv('y_train.csv')`
                    `pd.read_csv('X_test.csv')`
                    `pd.read_csv('y_test.csv')`
                    """,
                5: """
                    Summarize the content and format summary EXACTLY as follows:
                    ---
                    *ML model used*:
                    `ML model`
                    ---
                    *Place where you saved predictions*:
                    `acmecorp.com`
                    ---
                    *Results of the evaluations*:
                    `Metric: result`
                    ---
                    """,
                6: """
                    Summarize the content and format summary EXACTLY as follows:
                    ---
                    *Data set name*:
                    `name of the dataset`
                    ---
                    *User problem*:
                    `problem of the user`
                    ---
                    *Target variable*
                    `the variable we're trying to predict`
                    ---    
                    **Correlations:**
                    'Correlations found'
                    ---
                    **Columns:**
                    'the columns'    
                    ---
                    **Relevant Insights:**
                    'The relevant insights'
                    ---
                    *Machine learning model to use*:
                    Random Forest Regressor
                    ---
                    *Explanation and alternatives*:
                    'Explanations of the machine learning model used'
                    ---
                    *Transformations*:
                    `the transformations made to the data`
                    ---
                    *Splitting*:
                    `The splitting done to the data`
                    ---
                    *Results of the evaluations*:
                    `results of the metrics`
                    ---
                    """
            }
        except Exception as e:
            logging.error("Error in _get_summary_prompt", exc_info=True)
            raise

    def _build_chat_config(self, chat_id, base_config):
        try:
            config = base_config.copy()
            summary_args = self._get_summary_prompt()
            chat_methods = self._get_chat_methods()

            config.update({
                "chat_id": chat_id,
                "recipient": chat_methods[chat_id](),
                "message": self.task[chat_id - 1],
                "summary_args": {"summary_prompt":summary_args[chat_id]}
            })

            if chat_id == 1:
                config["clear_history"] = True
                config["silent"] = False

            return config
        except Exception as e:
            logging.error(f"Error in _build_chat_config for chat_id {chat_id}", exc_info=True)
            raise

    def _get_chat_methods(self):
        try:
            return {
                1: self.agent_manager.problem_definer_agent,
                2: self.agent_manager.data_analyst_agent,
                3: self.agent_manager.model_consultor_agent,
                4: self.agent_manager.feature_engineer_agent,
                5: self.agent_manager.model_builder_agent,
                6: self.agent_manager.report_builder_agent,
            }
        except Exception as e:
            logging.error("Error in _get_chat_methods", exc_info=True)
            raise
