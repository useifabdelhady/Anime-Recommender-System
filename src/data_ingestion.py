import os
import sys
import pandas as pd
import shutil

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # src folder
project_root = os.path.dirname(current_dir)  # parent of src folder
sys.path.append(project_root)

# Now import from other modules
from logger import get_logger
from custom_exception import CustomException
from config.paths_config import RAW_DIR, CONFIG_PATH, project_root
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.local_data_path = self.config["local_data_path"]
        self.file_names = self.config["file_names"]
        self.animelist_max_rows = self.config.get("animelist_max_rows", 5000000)
        
        # Create RAW_DIR with absolute path
        self.raw_dir_absolute = os.path.join(project_root, RAW_DIR)
        os.makedirs(self.raw_dir_absolute, exist_ok=True)
        
        logger.info("Data Ingestion Started....")
    
    def copy_and_process_local_files(self):
        try:
            print(f"Local data path: {self.local_data_path}")
            print(f"Files to process: {self.file_names}")
            
            for file_name in self.file_names:
                source_path = os.path.join(self.local_data_path, file_name)
                destination_path = os.path.join(self.raw_dir_absolute, file_name)
                
                print(f"\nProcessing: {file_name}")
                print(f"Source path: {source_path}")
                print(f"Destination path: {destination_path}")
                print(f"Source exists: {os.path.exists(source_path)}")
                
                # Check if source file exists
                if not os.path.exists(source_path):
                    # Try with different path separators
                    alt_source_path = source_path.replace('/', '\\')
                    print(f"Trying alternative path: {alt_source_path}")
                    if os.path.exists(alt_source_path):
                        source_path = alt_source_path
                        print("Alternative path works!")
                    else:
                        raise FileNotFoundError(f"Source file not found: {source_path}")
                
                if file_name == "animelist.csv":
                    # For animelist.csv, read only the specified number of rows
                    logger.info(f"Processing large file: {file_name} - limiting to {self.animelist_max_rows:,} rows")
                    data = pd.read_csv(source_path, nrows=self.animelist_max_rows)
                    data.to_csv(destination_path, index=False)
                    logger.info(f"Large file processed: Only {len(data):,} rows saved to {destination_path}")
                else:
                    # For other files, copy directly
                    logger.info(f"Copying file: {file_name}")
                    shutil.copy2(source_path, destination_path)
                    logger.info(f"File copied successfully: {file_name}")
                
                # Log file size information
                file_size = os.path.getsize(destination_path) / (1024 * 1024)  # Size in MB
                logger.info(f"File {file_name}: {file_size:.2f} MB")
                
        except FileNotFoundError as fe:
            logger.error(f"File not found: {str(fe)}")
            raise CustomException("Source file not found", fe)
        except Exception as e:
            logger.error(f"Error while processing local files: {str(e)}")
            raise CustomException("Failed to process local data", e)
    
    def run(self):
        try:
            logger.info("Starting Data Ingestion Process....")
            self.copy_and_process_local_files()
            logger.info("Data Ingestion Completed...")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            raise ce
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise CustomException("Data Ingestion failed", e)
        finally:
            logger.info("Data Ingestion DONE...")

if __name__ == "__main__":
    # Debug info
    print(f"Current directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Config path: {CONFIG_PATH}")
    print(f"Config exists: {os.path.exists(CONFIG_PATH)}")
    
    # Run data ingestion
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()