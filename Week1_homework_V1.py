import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

inmput_address=input()
# 读入图片
img_1=cv2.imread('D:\Artificial_Intelligence\pic\week1_homework/night_01.jpg')
#cv2.imshow('img_1',img_1)
#key = cv2.waitKey()

#对图像进行Gamma变换
def gamma_correlation(img,gamma=1.0):
    table=[]
    for i in range(256):
        table.append(((i/255)**gamma)*255)
    table=np.array(table).astype(img.dtype)
    return cv2.LUT(img,table)
img_brighter=gamma_correlation(img_1,0.4)
cv2.imshow('img_brighter',img_brighter)
key = cv2.waitKey()
#对图像进行颜色进行随机变换
def random_color_change(img):
    B,G,R=cv2.split(img)
    b_random_number=random.randint(-50,50)
    B=B+b_random_number
    B[B<0]=0
    B[B>255]=255
    B=B.astype(img.dtype)

    g_random_number = random.randint(-50, 50)
    G = G + g_random_number
    G[G < 0] = 0
    G[G > 255] = 255
    G = G.astype(img.dtype)

    r_random_number = random.randint(-50, 50)
    R = R + r_random_number
    R[R < 0] = 0
    R[R > 255] = 255
    R = R.astype(img.dtype)
    return cv2.merge([B,G,R])
img_random=random_color_change(img_1)
cv2.imshow('img_random',img_random)
key = cv2.waitKey()

#对图像进行直方图均衡
img_yuv=cv2.cvtColor(img_1,cv2.COLOR_BGR2YUV)
img_yuv[:,:,0]=cv2.equalizeHist(img_yuv[:,:,0])#色彩空间转换将BGR改成YUV Y明亮度 U色度 V饱和度
img_out=cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
cv2.imshow('img_out',img_out)
key = cv2.waitKey()

#对图片进行缩小，旋转，仿射
img_resize=cv2.resize(img_brighter,None,fx=0.8,fy=0.8)
M = cv2.getRotationMatrix2D((img_resize.shape[1] / 2, img_resize.shape[0] / 2), 45, 0.5) # 旋转中心，旋转角度，放缩大小
img_rotate = cv2.warpAffine(img_resize, M, (img_resize.shape[1], img_resize.shape[0]))
cv2.imshow('img_rotate',img_rotate)
key = cv2.waitKey()

rows, cols, ch = img_resize.shape
pts1 = np.float32([[0, 0], [cols , 0], [0, rows]])# 三点确定平面，Base点
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.9, rows * 0.1], [cols * 0.1, rows * 0.9]])# 三点确定平面，移动之后点
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img_resize, M, (cols, rows))
cv2.imshow('dst',dst)
key = cv2.waitKey()

#对图片投影变换
rows, cols, ch = img_brighter.shape
pts3 = np.float32([[0, 0], [rows , 0], [0, cols],[rows,cols]])# 四点确定变换，Base点
pts4 = np.float32([[rows * 0.1, cols * 0.1], [rows * 0.5, cols * 0.1], [rows * 0.1, cols * 0.9],[rows*0.9,cols*0.9]])# 四点确定变换，移动之后点
M_warp = cv2.getPerspectiveTransform(pts3, pts4)
img_warp = cv2.warpPerspective(img_brighter, M_warp, (cols, rows))
cv2.imshow('img_warp',img_warp)
key = cv2.waitKey()
print (pts3)
print (pts4)
print (img_warp.shape)
