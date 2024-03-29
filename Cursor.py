 import pyautogui
import time

# Set the interval (in seconds) for moving the mouse
interval = 60  # Move the mouse every minute

try:
    while True:
        # Move the mouse cursor a tiny amount to prevent screen from sleeping
        pyautogui.moveRel(1, 0, duration=0.1)
        pyautogui.moveRel(-1, 0, duration=0.1)
        time.sleep(interval)
except KeyboardInterrupt:
    print("Program stopped by the user.")
