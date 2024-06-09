import os
import cv2
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

jump_path = os.path.join("imgs", "jump")  # 需要跳的图片的根目录
none_path = os.path.join("imgs", "none")  # 不需要跳的图片的根目录


files = [os.path.join(jump_path, jump) for jump in os.listdir(jump_path)] + [
    os.path.join(none_path, none) for none in os.listdir(none_path)
]

X = []
Y = [0] * len(os.listdir(jump_path)) + [1] * len(os.listdir(none_path))

for file in files:
    image = cv2.imread(file, 0)
    flattened_image = image.reshape(-1)
    X.append(flattened_image)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


model = LogisticRegression(max_iter=500)

model.fit(X_train, Y_train)

score_x = model.score(X_train, Y_train)
score_y = model.score(X_test, Y_test)
print("score_x", score_x)
print("score_y", score_y)

joblib.dump(model, "auto_play.m")
