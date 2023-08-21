import cv2 as cv

camera = cv.VideoCapture(0)


flag = 1
num = 1

while(camera.isOpened()):
    ret_flag,Vshow = camera.read()
    cv.imshow("Capture_Test",Vshow)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        cv.imwrite("D:/PyCharm2023.1.2/project/Face Recognition opencv-python/face recognition/face"+"face"+str(num)+".jpg",Vshow)
        print("保存成功"+"face"+str(num)+".jpg")
        print("----------")
        num+=1
    elif k == ord(' '):
        break

camera.release()
cv.destroyAllWindows()