import cv2
import numpy as np
import time
from src import myUtills

import cv2


class BitwiseImage:
    def __init__(self, img):
        self.img = img

    def setImage(self, frame, y, x):
        rows, cols, channels = self.img.shape

        roi = frame[y:rows + y, x:cols + x]

        # create mask from logo
        img2gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # black out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(self.img, self.img, mask=mask)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)

        # y,x의 좌표 위치부터 ROI 지정
        frame[y:rows + y, x:cols + x] = dst


class detector:
    box_coordinate = 0  # box 좌표값

    def draw_detections(img, rects, thickness=1):
        for x, y, w, h in rects:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)

    def detect_body(self, frame):

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        found, w = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)  # found는 좌상단, 우하단 좌표값

        for x, y, w, h in found:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            detector.box_coordinate = [(x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h)]

        detector.draw_detections(frame, found)

        return frame

    def detect_body2(self, frame):

        # pre_path = 'c:\\opencv\\library\\opencv-master\\opencv-master\\data\\haarcascades\\'
        '''
        path_lowerbody = pre_path + 'haarcascade_lowerbody.xml'
        path_fullbody = pre_path + 'haarcascade_fullbody.xml'
        path_face = pre_path + 'haarcascade_frontalface_default.xml'
        path_upper = pre_path + 'haarcascade_upperbody.xml'
        path_eye = pre_path + 'haarcascade_eye.xml'
        path_hog = pre_path + 'hogcascade_pedestrians.xml'
        '''

        path_face = 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(path_face)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = face_cascade.detectMultiScale(gray, 1.3, 5)
        print(found)

        for (x, y, w, h) in found:
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            detector.box_coordinate = [(x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h)]

        detector.draw_detections(frame, found)

        return frame


class overlayer:
    img = cv2.imread('blue.png')
    clothes = BitwiseImage(img)
    isResize = False

    def overlay(self, frame_detected, box_coordinate):

        if box_coordinate is None:
            print("no!! box_coordinate")
            return frame_detected
        else:

            print(box_coordinate)
            frame = frame_detected

            x1 = box_coordinate[0][0]
            y1 = box_coordinate[0][1]
            x2 = box_coordinate[1][0]
            y2 = box_coordinate[1][1]

            if x1 is None:
                return frame

            else:

                if(not self.isResize):
                    # 1.크기 설정
                    ratio = (x2 - x1)  # 4.3 기준
                    r = ratio / self.clothes.img.shape[1]
                    dim = (int(ratio), int(self.clothes.img.shape[0] * r))
                    resized = cv2.resize(self.clothes.img, dim, interpolation=cv2.INTER_AREA)
                    img2 = resized
                    rows, cols, channels = img2.shape
                    self.clothes.img = img2
                    self.isResize = True

                # 2. y축 위치 설정
                x_move = -80
                y_move = int((y2 - y1) * 1.5)

                # 사람이 감지되었다고 가정

                self.clothes.setImage(frame, y1 + y_move, x1 + x_move)
                return frame

def draw_text(frame, text, x, y, color=(255, 255, 255), thickness=20, size=5):
    if x is not None and y is not None:
        cv2.putText(
            frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

def checkHandPosition(y, x):
    if (x>34 and x<110) and (y>41 and y<111):
        print("좌측 상단 원 클릭")
    elif (x>355 and x<422) and (y>41 and y<111):
        print("우측 상단 원 클릭")

def main():
    cap = cv2.VideoCapture(0)
    #
    init_time = time.time()

    counter = 5 # 화면에 띄울 숫자
    end_time = init_time + counter + 1 # 타이머 끝나는 시간
    secondPassed = init_time + 1 # 1초가 지났는지 안지났는지 비교하는용

    timer_over = False # 타이머 끝났는지 확인용

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 좌우반전

        if ret == True:
            center_x = int(frame.shape[1] / 2.5)
            center_y = int(frame.shape[0] / 2)

            if (time.time() < end_time):
                draw_text(frame, str(counter), center_x, center_y)
            if (time.time() > secondPassed): # 1초가 지난 경우
                counter -= 1
                secondPassed += 1

            cv2.imshow('frame', frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() > end_time):
                timer_over = True
                break
        else:
            break

        # Lucas Kanade optical flow parameters
        lk_params = dict( winSize  = (15,15),
                           maxLevel = 2,
                           criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Random colors
        color = np.random.randint(0,255,(100,3))

        ##Background UI 삽입
        backgroundUI = BitwiseImage(cv2.imread('back_img.png'))

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Set tracking points
        p0 = np.array([[[200.0, 200.0]], [[180.0, 180.0]]]) #tracking하는 포인트 위치와 개수 설정
        p0 = np.float32(p0)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

    while(timer_over == True):
        # optical flow
        ret, realframe = cap.read()
        realframe = cv2.flip(realframe, 1)  # 좌우반전

        frame_gray = cv2.cvtColor(realframe, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if (st.all() == 1):

            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()  # 현재 프레임의 좌표값
                c, d = old.ravel()  # 이전 프레임의 좌표값
                # checkHandPosition(a,b)
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)  # 현재와 이전의 프레임을 이어줌
                realframe = cv2.circle(realframe, (a, b), 5, color[i].tolist(), -1)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        #Background UI 삽입
        backgroundUI.setImage(realframe, 0, 0)
        # wearing.setImage(realframe, 0, 0)

        if (st.all() == 1): # st == 1 이면 프레임 안
            img = cv2.add(realframe, mask)
        else: # 프레임 밖으로 벗어난 경우
            img = realframe

        cv2.imshow('frame', img)

        # 카메라 종료
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # esc key
            cap.release()
            break

        # Detections

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()