""" File Logger
    Created by @JustinShirzad

    Create simple logs for your Python project.
    Uses process names and timestamps to automatically create and write to a log file.

    e.g. 
    [ INFO ] : 14:30:45 12-07-2025 : my_script - Line 11 : Process started
    [ ERROR ] : 14:30:46 12-07-2025 : my_script - Line 12 : Connection failed
"""

import os
import sys
import logging
import inspect
from datetime import datetime 

class LogSaver:
    def __init__(
            self,
            log_dir = "logs",
            process_name = None,
            id = None,
            show_timestamp = True,
            show_process_name = True,
            show_line_number = True,
            show_level = True
            ):
        
        self.log_dir = log_dir
        self.process_name = process_name
        self.id = id
        self.log_file, self.process_name = self.create_log_file()
        self.show_timestamp = show_timestamp
        self.show_process_name = show_process_name
        self.show_line_number = show_line_number
        self.show_level = show_level
        self.setup_logger()

    def create_log_file(self):
        relative_path = os.path.join(self.log_dir)

        # Creat directory if it doesn't exist
        os.makedirs(relative_path, exist_ok=True)

        # Get process name
        if self.process_name is None:
            process_frame = inspect.currentframe().f_back.f_back
            process_func_name = process_frame.f_code.co_name
            process_name = process_func_name

            # Check if the parent function is the main module
            if process_func_name == "<module>":
                script_name = os.path.basename(sys.argv[0])
                if script_name.endswith(".py"):
                    script_name = script_name[:-3]
                process_name = script_name
        else:
            process_name = self.process_name

        # Add the id value
        if self.id is not None:
            process_name = f"{process_name}_{self.id}"

        # Create timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Initialise log file
        log_filename = f"{process_name}_{timestamp}.log"
        log_file = os.path.join(relative_path, log_filename)
        with open(log_file, "w") as log:
            log.write(f"|=== START OF LOG FOR: {process_name} ===|\n")
            log.write(f"|=== Started: {datetime.now().strftime('%H:%M:%S %d-%m-%Y')} ===|\n")

        return log_file, process_name
    
    def setup_logger(self):
        self.logger = logging.getLogger(self.process_name)
        self.logger.setLevel(logging.DEBUG)
            
        # Create file handler to write to the correct log file
        self.logger.handlers.clear()
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)

        log_format = '[ %(levelname)s ] : %(asctime)s : %(name)s - %(message)s'

        if not self.show_level:
            log_format = log_format.replace('[ %(levelname)s ] : ', '')
        if not self.show_timestamp:
            log_format = log_format.replace('%(asctime)s : ', '')
        if not self.show_process_name:
            log_format = log_format.replace('%(name)s', '')
        if not self.show_line_number and self.show_process_name:
            log_format = log_format.replace(' - ', '')
        
        # Format the log messages for the file
        log_formatter = logging.Formatter(
            fmt=log_format,
            datefmt='%H:%M:%S %d-%m-%Y'
        )
        file_handler.setFormatter(log_formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)

    def create_log(self, log_level, message):
        # Navigate to the frame where the logging method was called
        process_frame = inspect.currentframe().f_back.f_back
        process_lineno = process_frame.f_lineno

        formatted_message = f"Line {process_lineno} : {message}"

        if not self.show_line_number:
            formatted_message = f" : {message}"

        if log_level == "debug":
            self.logger.debug(formatted_message)
        elif log_level == "info":
            self.logger.info(formatted_message)
        elif log_level == "warning":
            self.logger.warning(formatted_message)
        elif log_level == "error":
            self.logger.error(formatted_message)
        elif log_level == "critical":
            self.logger.critical(formatted_message)

    # Logging methods (Single letter)
    def D(self, message):
        self.create_log("debug", str(message))

    def I(self, message):
        self.create_log("info", str(message))
    
    def W(self, message):
        self.create_log("warning", str(message))

    def E(self, message):
        self.create_log("error", str(message))

    def C(self, message):
        self.create_log("critical", str(message))

    # Logging methods (Full words)
    def debug(self, message):
        self.create_log("debug", str(message))

    def info(self, message):
        self.create_log("info", str(message))

    def warning(self, message):
        self.create_log("warning", str(message))

    def error(self, message):
        self.create_log("error", str(message))

    def critical(self, message):
        self.create_log("critical", str(message))

if __name__ == "__main__":
        
    logs = LogSaver()

    logs.info("This is an info message.")
    logs.debug("This is a debug message.")
    logs.warning("This is a warning message.")
    logs.error("This is an error message.")
    logs.critical("This is a critical message.")