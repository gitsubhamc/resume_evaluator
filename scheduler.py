from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import os

def run_main_script():
    print(f"Running main.py at {datetime.now()}")
    os.system("python ./main.py")

if __name__ == "__main__":
    scheduler = BackgroundScheduler()

    scheduler.add_job(run_main_script, 'interval', minutes=5)

    scheduler.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        scheduler.shutdown()
