from dotenv import load_dotenv
from backend.chat_manager import ChatManager
from backend.config import serialize_chat_result
from backend.files_manipulation import *
import os

# Load environment variables from .env file
load_dotenv()

user_problems_dict={
    1:'I want to predict whether future customers will be satisfied',
    2:'I want to forecast the passengers rate',
    3:'I want to predict car price',
    4:'I want to predict crab age',
    5:'I want to forecast the daily minimum temperature',
    6:'I want to predict cholesterol',
    7:'I want to predict bike rents for the day',
    8:'I want to predict the type of anemia',
    9:'I want to forecast the eletrical production',
    10:'I want to predict employee attrition',
    11:'I want to predict Fuel Consumption',
    12:'I want to forecast gasoline prices',
    13:'I want to predict house prices',
    14:'I want to predict the cost of medical insurance',
    15:'I want to forecast microsoft stock',
    16:'I want to forecast the monthly beer production',
    17:'I want to predict the mushroom class (is edible or poisenous)',
    18:'I want to predict the type of obesity',
    19:'I want to forecast energy consumption',
    20:'I want to forecast population growth',
    21:'I want to predict the variable target',
    22:'I want to predict the vibration',
    23:'I want to predict the overal rank of happiness',
    24:'I want to forecast RUL of water pump',
    25:'I want to forecast the shampoos sale',
    26:'I want to predict students Performance',
    27:'I want to predict costumer Churn Category and Churn Reason',
    28:'I want to predict recurrence of thyroid cancer, if yes or no',
    29:'I want to predict the classification of the wine quality',
    30:'I want to predict the classification of the wine quality'

}
datasets_dict={
    1:'Airline_customer_satisfaction.csv',
    2:'AirPassengers.csv',
    3:'cars.csv',
    4:'CrabAgePrediction.csv',
    5:'daily-minimum-temperatures-in-me.csv',
    6:'dataset_2190_cholesterol.csv',
    7:'day.csv',
    8:'diagnosed_cbc_data_v4.csv',
    9:'Electric_Production.csv',
    10:'employee_attrition',
    11:'FuelConsumption.csv',
    12:'Hourly_Gasoline_Prices.csv',
    13:'kc_house_data.csv',
    14:'medical_insurance.csv',
    15:'Microsoft_Stock.csv',
    16:'monthly-beer-production.csv',
    17:'mushrooms.csv',
    18:'Obesity Classification.csv',
    19:'pjme.csv',
    20:'POP.csv',
    21:'predictive_maintenance.csv',
    22:'predictive-maintenance-dataset.csv',
    23:'report_2018-2019.csv',
    24:'rul_hrs.csv',
    25:'sales-of-shampoo-over-a-three-ye.csv',
    26:'Student_Performance.csv',
    27:'telco.csv',
    28:'Thyroid_Diff.csv',
    29:'winequality-red.csv',
    30:'winequality-white.csv'
    }

def accumulate_cost_values(chat_results):
    # Initialize accumulators
    accumulated = {
        'usage_including_cached_inference': {
            'total_cost': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0
        },
        'usage_excluding_cached_inference': {
            'total_cost': 0,
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0
        }
    }

    # Perform accumulation
    for cr in chat_results:
        inc_cache_inf = cr.cost['usage_including_cached_inference']
        exc_cache_inf = cr.cost['usage_excluding_cached_inference']

        # Including cached inference accumulations
        accumulated['usage_including_cached_inference']['total_cost'] += inc_cache_inf.get('total_cost', 0)
        for model_data in inc_cache_inf.values():
            if isinstance(model_data, dict):
                accumulated['usage_including_cached_inference']['total_prompt_tokens'] += model_data.get('prompt_tokens', 0)
                accumulated['usage_including_cached_inference']['total_completion_tokens'] += model_data.get('completion_tokens', 0)
                accumulated['usage_including_cached_inference']['total_tokens'] += model_data.get('total_tokens', 0)

        # Excluding cached inference accumulations
        accumulated['usage_excluding_cached_inference']['total_cost'] += exc_cache_inf.get('total_cost', 0)
        for model_data in exc_cache_inf.values():
            if isinstance(model_data, dict):
                accumulated['usage_excluding_cached_inference']['total_prompt_tokens'] += model_data.get('prompt_tokens', 0)
                accumulated['usage_excluding_cached_inference']['total_completion_tokens'] += model_data.get('completion_tokens', 0)
                accumulated['usage_excluding_cached_inference']['total_tokens'] += model_data.get('total_tokens', 0)

    return accumulated

def run_chat_for_dataset(upload_file, user_problem):
    model = "llama3"
    selected_agents = [1, 2, 3, 4, 5, 6]
    
    if model:
        data_name = os.path.splitext(upload_file)[0] if upload_file else 'default'
        chat_manager = ChatManager(f'datasets/{upload_file}', user_problem, data_name, model=model, include_chats=selected_agents)
        chat_results = chat_manager.initiate_chats()
        accumulated = accumulate_cost_values(chat_results)
        serialized_results = [serialize_chat_result(cr) for cr in chat_results]
        full_report = "\n\n" + "-" * 80 + "\n\n".join(serialized_results)
        # You might want to append or include the total cost into the full report.
        full_report_with_cost = f"{full_report}\n\nAccumulated Costs: {accumulated}"
        save_full_report(data_name, full_report_with_cost)


import time

def save_checkpoint(checkpoint_file, index):
    with open(checkpoint_file, 'w') as f:
        f.write(str(index))

def load_checkpoint(checkpoint_file):
    try:
        with open(checkpoint_file, 'r') as f:
            return int(f.read())
    except FileNotFoundError:
        return 0  # If the checkpoint file does not exist, start from the beginning


def main():
    checkpoint_file = 'last_completed_checkpoint.txt'
    
    # Initialize an empty string to store execution times
    execution_times_log = ""

    # Start a timer for the total execution time
    start_time_total = time.time()
    
    last_completed_index = load_checkpoint(checkpoint_file)
    for i in [5,12,13,20,21,24,27]:
        start_time_function = time.time()  # Timer for this function
        
        upload_file = datasets_dict[i]
        user_problem = user_problems_dict[i]
        
        try:
            run_chat_for_dataset(upload_file, user_problem)
            
            # Calculate function duration and append it to the log
            duration = time.time() - start_time_function
            execution_times_log += f"Dataset {i} processing time: {duration:.2f} seconds\n"
            
            save_checkpoint(checkpoint_file, i)
        
        except Exception as e:
            print(f"An error occurred while processing dataset {i}: {e}")
            # Append the error information to the log
            execution_times_log += f"An error occurred while processing dataset {i}: {e}\n"
            
            if "rate limit" in str(e).lower():
                print("Rate limit exceeded. Waiting to retry...")
                time.sleep(10)
                continue
                
            else:
                continue

    duration_total = time.time() - start_time_total
    execution_times_log += f"Total execution time: {duration_total:.2f} seconds\n"
    
    generated_files_dir = 'tasks/generated_files'
    download_generated_files(generated_files_dir)
    
    # Write the execution times to a .txt file
    with open('execution_times.txt', 'w') as file:
        file.write(execution_times_log)

# Main execution point of the program
if __name__ == "__main__":
    main()
