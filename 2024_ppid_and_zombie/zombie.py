import os
import time

def create_zombie():
    pid = os.fork()

    if pid > 0:  # If current process is parent
        print(f"Parent process (PID: {os.getpid()})")
        print(f"Child process (PID: {pid}) created, but will not be waited for.")
        time.sleep(1000)  # To do not exit() parent process, do sleep
    elif pid == 0:  # If current process is child
        print(f"Child process (PID: {os.getpid()}) exiting...")
        os._exit(0)  # Exit child process
    else:
        print("Fork failed")

if __name__ == "__main__":
    create_zombie()

