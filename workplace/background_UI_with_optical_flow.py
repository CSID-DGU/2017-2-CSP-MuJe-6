import cv2
import numpy as np

from src import myUtills


def checkHandPosition(y, x, ):
    global wearing
    if (x>34 and x<110) and (y>41 and y<111):
        wearing = myUtills(cv2.imread('suit3.png'))
        print("hello")
    elif (x>355 and x<422) and (y>41 and y<111):
        print("haha")

if __name__ == '__main__':

    ##Background UI 삽입
    backgroundUI = myUtills(cv2.imread('back_img.png'))
    wearing = myUtills(cv2.imread('suit2.png'))
    ##Optical Flow 삽입
    cap = cv2.VideoCapture(0)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Set tracking points
    p0 = np.array([[[200.0,200.0]], [[180.0,180.0]]])
    p0 = np.float32(p0)
    #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while(1):

        ret,realframe = cap.read()
        realframe = cv2.flip(realframe,1)  #좌우반전

        #trackingframe = realframe
        frame_gray = cv2.cvtColor(realframe, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        if(st.all() == 1):

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel() #현재 프레임의 좌표값
                c,d = old.ravel() #이전 프레임의 좌표값
                #checkHandPosition(a,b)
                mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2) #현재와 이전의 프레임을 이어줌
                realframe = cv2.circle(realframe,(a,b),5,color[i].tolist(),-1)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1,1,2)
            print("all ")


        ###Background UI 삽입
        backgroundUI.setImage(realframe, 0, 0)
       # wearing.setImage(realframe, 0, 0)

        if(st.all() == 1):
            img = cv2.add(realframe, mask)
        else:
            print("else")
            img = realframe

        cv2.imshow('frame', img)

        # 카메라 종료
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # esc key
            cap.release()
            break

    cap.release()
    cv2.destroyAllWindows()


