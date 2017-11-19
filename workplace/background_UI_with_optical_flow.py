import numpy as np
import cv2

class BitwiseImage:
    def __init__(self, img):
        self.img = img

    def setImage(self, frame, x, y):
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
        frame[x:rows+x, y:cols+y ] = dst

def checkHandPosition(x, y):
    global wearing
    if (x>34 and x<110) and (y>41 and y<111):
        wearing = BitwiseImage(cv2.imread('suit3.png'))
        print("hello")
    elif (x>355 and x<422) and (y>41 and y<111):
        print("haha")


# ShiTomasi corner detection parameters
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Lucas Kanade optical flow parameters
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Random colors
color = np.random.randint(0,255,(100,3))

if __name__ == '__main__':

    ##Background UI 삽입
    backgroundUI = BitwiseImage(cv2.imread('back_img.png'))
    wearing = BitwiseImage(cv2.imread('suit2.png'))
    ##Optical Flow 삽입
    cap = cv2.VideoCapture(0)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Set tracking points
    p0 = np.array([[[200.0,200.0]]])
    p0 = np.float32(p0)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):

        ret,realframe = cap.read()

        #trackingframe = realframe
        frame_gray = cv2.cvtColor(realframe, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            checkHandPosition(a,b)
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            realframe = cv2.circle(realframe,(a,b),5,color[i].tolist(),-1)


        ###Background UI 삽입
        backgroundUI.setImage(realframe,0,0)


        wearing.setImage(realframe,0,0)

        img = cv2.add(realframe,mask)
        cv2.imshow('frame',img)


        #카메라 종료
        k = cv2.waitKey(30) & 0xff
        if k == 27: #esc key
            cap.release()
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cap.release()
    cv2.destroyAllWindows()


