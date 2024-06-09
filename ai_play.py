import time
import joblib
import cv2
from sklearn.linear_model import LogisticRegression
from PIL import ImageGrab
from pynput import keyboard
from pynput.keyboard import Key

time.sleep(3)
# 0、创建键盘
kb = keyboard.Controller()
# 1、加载模型
model: LogisticRegression = joblib.load("auto_play.m")
while True:
    # 2、准备数据
    img = ImageGrab.grab().resize((960, 540)).save("current.jpg")

    x = cv2.imread("current.jpg", 0).reshape(-1)
    pre_x = [x]
    # 3、预测
    pred = model.predict(pre_x)
    print(pred)
    # 如果需要跳，则按下空格
    if pred[0] == 0:
        kb.press(Key.space)

    time.sleep(0.1)
