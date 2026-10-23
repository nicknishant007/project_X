import os
import yaml

CONFIG_PATH = "configs/ingestion.yaml"

with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

DATA_DIR = config['data_dir']

SUPPORTED_FORMATS = set(config['supported_extensions'])

class LocalScanner:
    @staticmethod
    def scan():
        for root, _, files in os.walk(DATA_DIR):
            for file_name in files:
                
                extension=os.path.splitext(file_name)[1].lower()

                if extension in SUPPORTED_FORMATS:
                    yield os.path.join(root, file_name)


#VERY VERY IMPORTANT

'''os=folder retrieval and file path handling
yaml=configuration management
DATA_DIR=directory to scan for files
SUPPORTED_FORMATS=set of file extensions to process(in config)
@staticmethod=method that can be called without creating an instance of the class
scan method=walks through the directory and yields file paths that match supported formats
for root, _, files in os.walk(DATA_DIR)=traverse the directory tree
os.path.splitext(file_name)[1].lower()=get file extension and convert to lowercase
if extension in SUPPORTED_FORMATS=check if the file extension is supported
yield os.path.join(root, file_name)=return the full path of the file for processing

#IMPORTANT
data/incoming
      ↓
Recursive folder scan
      ↓
Supported files only
      ↓
Yield file paths'''