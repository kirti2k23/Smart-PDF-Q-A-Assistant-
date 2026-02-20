import os
from pathlib import Path
import logging

"""
This script creates the necessary directory structure and files for the Smart PDF QA Assistant project.

"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Project_Folder = "Smart_PDF_QA_Assistant"

Directories = [
    f'{Project_Folder}/app.py',
    f'{Project_Folder}/rag_pipeline.py',
    "data/",  
]

for directory in Directories:
    filepath = Path(directory)
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Directory '{filedir}' created for the file name {filename}")

    if (not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"File '{filepath}' created.")
    else:
        logging.info(f"File '{filepath}' already exists and is not empty. Skipping creation.")