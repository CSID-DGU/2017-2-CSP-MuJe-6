import cv2
import os

# os.setcwd("C:/Users/yunhwan/workspace/icpc_yun/2017-2-CSP-MuJe-6"")
# print(os.getcwd())
#
# # goto = 2
# # #img = cv2.imread('src/back_img'+str(goto)+'.png')
# #
# img =  cv2.imread('red.png')
#
# img = cv2.resize(img, (100,100), interpolation=cv2.INTER_AREA)
# print(img.shape)
#
# #cv2.imshow( "new", img)
#
#
# # cv2.waitKey(0)
#
# print('back_img'+'red'+'.png')
img = cv2.imread('suit'+'2'+'.png')
cv2.imshow('shw',img)
cv2.waitKey(0)
