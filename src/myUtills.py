import cv2

class BitwiseImage:
    def __init__(self, img):
        self.img = img

    def setImage(self, frame, y, x):
        rows,cols,channels = self.img.shape

        roi = frame[0:rows, 0:cols]

        # create mask from logo
        img2gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # black out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(self.img,self.img,mask = mask)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)

        #y,x의 좌표 위치부터 ROI 지정
        frame[y:rows+y, x:cols+x ] = dst
