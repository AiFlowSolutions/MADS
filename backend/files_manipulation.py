import os
import streamlit as st

def write_file(location:str, content:bytes) -> None:
    """
    Write binary content to a file specified by `location`.
    
    Args:
    location (str): The path where the file will be created and written to.
    content (bytes): The binary content to write into the file.

    Returns:
    None
    """
    with open(location, "wb") as f:
        f.write(content)

def save_full_report(name:str, results:str) -> None:
    """
    Save the results of a session into a text file with a given `name`.
    
    Args:
        name (str): The unique identifier for the session, used in the filename.
        results (str): The textual representation of the results to save in the file.

    Returns:
        None
    """
    # Creating the full file path using an f-string for literal string interpolation.
    file_path = fr'chat_history/chat_{name}.txt'
    with open(file_path, 'w', encoding='utf-8') as file:  # Using UTF-8 encoding for text data.
        file.write(results)  # Writing the results to the file.

def download_generated_files(directory) -> None:
    """
    Offer download buttons for all files within a specified directory.
    
    Args:
        directory (str): The directory to search for files to download.

    Returns:
    None
    """
    # Listing all files in the given directory.
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)  # Joining directory path and file name to get the full path.
        with open(file_path, "rb") as fp:  # Opening file in binary read mode.
            st.download_button(
                label="Download " + file_name,  # Label for the button.
                data=fp,  # File pointer passed as data for the download.
                file_name=file_name,  # Suggested name for the downloaded file.
                mime="text/plain"  # MIME type for plain text files.
            )