import cv2
import os
import urllib
import urllib.request

# 加载训练好的数据文件
recogizer = cv2.face.LBPHFaceRecognizer_create()
# 加载数据
recogizer.read('./face recognition/face trainer/trainer.yml')
# 名称
names = []
# 警报全局变量
warningtime = 0


# 功能模块
def warning():
    print("！！！人脸识别错误！！！")


"""
例如：
报警模块
短信验证模块
加密数据发送模块
......
"""


# 准备识别的图片
def face_detect_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
    face_detect = cv2.CascadeClassifier(
        'D:/OpenCV 4.8.0/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 225), thickness=2)
        cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)
        # 人脸识别
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        if confidence > 80:
            global warningtime
            warningtime += 1
            if warningtime >= 100:
                warning()  # 触发报警模块
                warningtime = 0
            cv2.putText(img, 'unknow', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)  # 识别结果显示在图片上
        else:
            cv2.putText(img, str(names[ids - 1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result', img)


# 加载视频
camera = cv2.VideoCapture('./1.mp4')

while True:
    flag, frame = camera.read()
    if not flag:
        break
cv2.destroyAllWindows()
camera.release()
