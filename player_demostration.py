import time
from PIL import ImageGrab

time.sleep(3)
while True:
    img = ImageGrab.grab()
    img = img.resize((960, 540))
    img.save(f"imgs/{str(time.time())}.jpg")
    print("screen shot..")
    time.sleep(0.2)
