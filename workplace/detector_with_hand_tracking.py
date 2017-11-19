import numpy as np
import cv2

# 옵터컬 플로우용 패러미터
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))  # 랜덤 색상

# 파일명
# cap = cv2.VideoCapture("C:\\cv\\Videos\\won_arms.mp4")
cap = cv2.VideoCapture("s2.mp4")
cap.set(cv2.CAP_PROP_FPS, 1)

# 첫 프레임 잡아서 피쳐 좌표 찍어놓기
ret, old_frame = cap.read()
#########################
old_frame = old_frame[0:720, 200:1000]
#########################

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# 여기 값을 수정!!

p0 = np.array([[[510, 380]],  # 좌손목
               [[670, 370]],  # 우손목
               [[513, 318]],  # 좌꿈치
               [[660, 285]]],  # 우꿈치
              dtype='float32')
#########################
p0 = np.array([[[310, 380]],  # 좌손목
               [[470, 370]],  # 우손목
               [[313, 318]],  # 좌꿈치
               [[460, 285]]],  # 우꿈치
              dtype='float32')
#########################
mask = np.zeros_like(old_frame)

#################################
body_detector = detector()
clothes_overlayer = overlayer()
#################################


while (1):
    ret, frame = cap.read()

    ####################################
    # frame 조정 720,1280
    frame = frame[0:720, 200:1000]

    # 1.detect body
    frame_detected = body_detector.detect_body2(frame)
    box_coordinate = body_detector.box_coordinate

    # 2.overlay clothes
    frame_overaid = clothes_overlayer.overlay(frame_detected, box_coordinate)
    ####################################

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cv2.destroyAllWindows()
cap.release()


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

        # haar_version

    def detect_body2(self, frame):

        pre_path = 'c:\\opencv\\library\\opencv-master\\opencv-master\\data\\haarcascades\\'

        path_lowerbody = pre_path + 'haarcascade_lowerbody.xml'
        path_fullbody = pre_path + 'haarcascade_fullbody.xml'
        path_face = pre_path + 'haarcascade_frontalface_default.xml'
        path_upper = pre_path + 'haarcascade_upperbody.xml'
        path_eye = pre_path + 'haarcascade_eye.xml'
        path_hog = pre_path + 'hogcascade_pedestrians.xml'

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

    def overlay(self, frame_detected, box_coordinate):
        frame = frame_detected

        x1 = box_coordinate[0][0]
        y1 = box_coordinate[0][1]
        x2 = box_coordinate[1][0]
        y2 = box_coordinate[1][1]

        img = overlayer.img

        # 1.크기 설정
        ratio = x2 - x1
        r = ratio / img.shape[1]
        dim = (int(ratio), int(img.shape[0] * r))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img2 = resized
        rows, cols, channels = img2.shape

        # 2. y축 위치 설정
        y_move = int((y2 - y1) / 1.7)

        # 사람이 감지되었다고 가정
        roi = frame[y1 + y_move:rows + y1 + y_move, x1:cols + x1]

        # create mask from logo
        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # black out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only region of logo from logo image.
        # img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2)
        frame[y1 + y_move:rows + y1 + y_move, x1:cols + x1] = dst

        return frame