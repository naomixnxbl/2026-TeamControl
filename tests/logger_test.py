# Use pip to install the package with "pip install -e ." in the terminal to install it into your Python environment.
# Run this script to see the demo the logging and try it out for yourself.

from TeamControl.utils.Logger import LogSaver

logs = LogSaver()

logs.info("This is an info message.")
logs.debug("This is a debug message.")
logs.warning("This is a warning message.")
logs.error("This is an error message.")
logs.critical("This is a critical message.")