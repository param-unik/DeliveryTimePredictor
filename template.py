import os
import logging
from pathlib import Path


project_name = "src"

while True:
    project_name = input("Enter the project name default(src): ")
    break

list_of_file = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"config/config.yaml",
    "schema.yaml",
    "app.py",
    "main.py",
    "logs.py",
    "exception.py",
    "setup.py",
]

for file_path in list_of_file:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)

    if (not os.path.isfile(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as file:
            pass
    else:
        logging.warning("File already exists at  path ", {file_path})
