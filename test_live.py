import time
import sys
from live_caption import LiveCaptioner

def callback(text):
    print(f"Callback received: {text}")

c = LiveCaptioner(model_size="tiny")
c.add_callback(callback)
print("Starting...")
c.start()
print("Sleeping for 10 seconds. Play some audio to test...")
time.sleep(10)
c.stop()
