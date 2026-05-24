import os
import multiprocessing

print(f"Total cores: {multiprocessing.cpu_count()}")
print(f"Usable cores: {max(1, multiprocessing.cpu_count() - 1)}")
print(f"Actual PID: {os.getpid()}") # this is the unique process running ID