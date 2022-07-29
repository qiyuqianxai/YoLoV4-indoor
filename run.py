import cv2
import time
import numpy as np
from yolo import YOLO
from PIL import Image
import matplotlib as plt

#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
def detect_image(image_path):
    print('Start detect!')
    yolo = YOLO()
    try:
        image = Image.open(image_path)
        image.show()
    except:
        print('Open Error! Try again!')
        pass
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
#         r_image.save(image_path.split('.')[0] + '_result.png')
    print('Finish detect!')
	
	
# detect_image('test/image/door_000.jpg')
# detect_image('test/image/knife_000.jpg')
# detect_image('test/image/phone_000.jpg')
# detect_image('test926/image/scissor_000.jpg')
detect_image('1.jpg')