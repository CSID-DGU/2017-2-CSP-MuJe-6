# -*- coding:utf-8 -*-
import cv2
import math
import numpy as np


class BitwiseImage:
    def __init__(self, img):
        self.img = img

    def setImage(self, frame, y, x):
        rows, cols, channels = self.img.shape

        # y,x의 좌표 위치부터 ROI 지정
        roi = frame[y:rows + y, x:cols + x]

        # create mask from logo
        img2gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # black out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(self.img, self.img, mask=mask)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        frame[y:rows + y, x:cols + x] = dst


class detector:
    box_coordinate = None  # box 좌표값

    def draw_detections(self, img, rects, thickness=1):
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
        # print(found)

        for (x, y, w, h) in found:
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            detector.box_coordinate = [(x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h)]

        self.draw_detections(frame, found)

        return frame


class listOfClothes:
    def __init__(self):
        self.array = [["001_red_dress", "002_tux", "003_tux_pants"],
                      ["004_green_dress", "005_hwaian_shirts", "006_blue_pants"],
                      ["007_blue_dress", "008_pink", "009_darkgreen pants"]
                      ]  # 옷 폴더 이름
        self.whatClothes = [[0, 0, 1], [0, 0, 1],[0,0,1]]

    def getClothes(self, i, j):
        return self.array[i][j]


resizing_coef = 5.5


class overlayer:
    isResize = False
    img_array = listOfClothes()

    def overlay(self, frame_detected, box_coordinate, clothesIndex):

        img = self.img_array.getClothes(clothesIndex[0], clothesIndex[1])
        img = self.img_array.getClothes(clothesIndex[0], clothesIndex[1])
        clothesToWear = BitwiseImage(cv2.imread("clothes/" + img + "/body.png"))  # 입을 옷을 Bitewise 세팅

        if box_coordinate is None:
            # print("no!! box_coordinate")
            return frame_detected
        else:

            # print(box_coordinate)
            frame = frame_detected

            x1, y1 = box_coordinate[0]
            x2, y2 = box_coordinate[1]

            if x1 is None:
                return frame
            else:
                # if (not self.isResize):
                # 1.크기 설정
                # if not self.isResize :
                ratio = math.ceil(x2 - x1) * resizing_coef  # 옷사이즈
                r = ratio / clothesToWear.img.shape[1]
                dim = (int(ratio), int(clothesToWear.img.shape[0] * r))
                resized = cv2.resize(clothesToWear.img, dim, interpolation=cv2.INTER_AREA)
                img2 = resized
                # rows, cols, channels = img2.shape
                clothesToWear.img = img2
                self.isResize = True

                # 2. y축 위치 설정
                x_move = -80
                y_move = int((y2 - y1))

                # 사람이 감지되었다고 가정
                cv2.imshow("body", clothesToWear.img)
                clothesToWear.setImage(frame, y1 + y_move, x1 + x_move)
                return frame


class arm_overlayer:
    img_array = listOfClothes()
    isResize = False

    def rotationDegree(self, x1, y1, x2, y2):

        return (-math.atan((y2 - y1) / (x2 - x1)) * (180 / 3.141592))

    def overlay(self, frame_detected, box_coordinate, hand, clothesIndex):

        left = hand[0][0]  # 왼손좌표
        right = hand[1][0]  # 오른손좌표

        img = self.img_array.getClothes(clothesIndex[0], clothesIndex[1])  # 옷 폴더 이름 string 값

        leftInstance = BitwiseImage(cv2.imread("clothes/" + img + "/left.png"))  # 입을 옷을 Bitewise 세팅
        rightInstance = BitwiseImage(cv2.imread("clothes/" + img + "/right.png"))

        leftToWear = leftInstance.img  # 입을 옷 이미지
        rightToWear = rightInstance.img
        armArray = [leftToWear, rightToWear]  # for 문 돌리기 위해서 list화

        #     print("this is bc : ", box_coordinate)
        if box_coordinate == 0:
            return frame_detected
        elif box_coordinate != 0:
            #        print(box_coordinate , "in else ")
            frame = frame_detected
            # (box_coordinate)
            #       print("this is box:        ",box_coordinate)
            x1, y1 = box_coordinate[0]
            x2, y2 = box_coordinate[1]

            for i in range(2):
                # if (not self.isResize):
                # 1.크기 설정
                ratio = (x2 - x1) * 3.5 # 옷사이즈
                r = ratio / armArray[i].shape[1]
                dim = (int(ratio), int(armArray[i].shape[0] * r))
                resized = cv2.resize(armArray[i], dim, interpolation=cv2.INTER_AREA)
                armArray[i] = resized

                # 회전
                num_rows, num_cols = armArray[i].shape[:2]
                degree = self.rotationDegree(x2, y2, right[0], right[1]) if i == 1 else (
                self.rotationDegree(x2, y2, left[0], left[1]) * -1)
                degree = degree if i == 1 else degree * -1  # 왼손 오른손 구분

                r_rotation = (int((num_rows/5)*2), int((num_cols)/3))
                frame = cv2.circle(frame, r_rotation, 10, (0,255,255), -1)
                rotation_matrix = cv2.getRotationMatrix2D(r_rotation, degree,
                                                          1) if i == 1 else cv2.getRotationMatrix2D(
                    (num_cols / 2 + 30, num_rows / 2 - 30), degree, 1)
                imgRotation = cv2.warpAffine(armArray[i], rotation_matrix, (num_cols, num_rows))
                armArray[i] = imgRotation
                # print("Arm ratation degree:", degree)

            kernel = np.ones((5, 5), np.float32) / 25
            leftInstance.img = armArray[0]
            rightInstance.img = armArray[1]
            # cv2.imshow("left", leftInstance.img)
            # cv2.imshow("right",rightInstance.img)
            # 2. y축 위치 설정
            x_move = -50
            y_move = -30

            rightInstance.setImage(frame, y2 + y_move + 10, x2 - x_move - 40)
            leftInstance.setImage(frame, y2 + y_move, x1 + x_move - 120)
            return frame


class pants_overlayer:
    img_array = listOfClothes()
    isResize = False

    def rotationDegree(self, x1, y1, x2, y2):

        return (-math.atan((y2 - y1) / (x2 - x1)) * (180 / 3.141592) + 90)  # 팔각도에서 -90

    def overlay(self, frame_detected, box_coordinate, p1, clothesIndex):

        left, right = box_coordinate
        left_leg = p1[2][0]  # 왼손좌표
        right_right = p1[3][0]  # 오른손좌표

        img = self.img_array.getClothes(clothesIndex[0], clothesIndex[1])  # 옷 폴더 이름 string 값

        leftInstance = BitwiseImage(cv2.imread("clothes/" + img + "/left.png"))  # 입을 옷을 Bitewise 세팅
        rightInstance = BitwiseImage(cv2.imread("clothes/" + img + "/right.png"))
        bodyInstance = BitwiseImage(cv2.imread("clothes/" + img + "/body.png"))

        leftToWear = leftInstance.img  # 입을 옷 이미지
        rightToWear = rightInstance.img
        bodyToWear = bodyInstance.img

        armArray = [leftToWear, rightToWear, bodyToWear]  # for 문 돌리기 위해서 list화

        if box_coordinate is None:
            return frame_detected
        else:
            frame = frame_detected
            # print(pants)
            x1, y1 = box_coordinate[0]
            x2, y2 = box_coordinate[1]
            rotation_x1, rotation_y1 = (x1 + x2) / 2, y2 + 50  # 회전 위한 좌상단 좌표
            rotation_x2, rotation_y2 = p1[3][0]  # 회전 위한 우하단 좌표
            rotation_x3, rotation_y3 = p1[2][0]

            for i in range(3):
                # 1.크기 설정
                ratio = (x2 - x1) * 6
                r = ratio / armArray[i].shape[1]
                dim = (int(ratio), int(armArray[i].shape[0] * r))
                resized = cv2.resize(armArray[i], dim, interpolation=cv2.INTER_AREA)
                armArray[i] = resized

                # 회전
                if i != 2:
                    num_rows, num_cols = armArray[i].shape[:2]
                    degree = self.rotationDegree(rotation_x1, rotation_y1, rotation_x2,
                                                 rotation_y2) if i == 1 else -1 * self.rotationDegree(rotation_x1,
                                                                                                      rotation_y1,
                                                                                                      rotation_x3,
                                                                                                      rotation_y3) + 180
                    degree = degree if i == 1 else degree * -1  # 왼손 오른손 구분
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2 - 20, num_rows / 2 - 20), degree, 1)
                    imgRotation = cv2.warpAffine(armArray[i], rotation_matrix, (num_cols, num_rows))
                    armArray[i] = imgRotation

                    #           print("rotation param:", rotation_x1,rotation_x2,rotation_x2,rotation_y2)
                    #           print("pants degree : ",degree)

            kernel = np.ones((5, 5), np.float32) / 25
            leftInstance.img = armArray[0]
            rightInstance.img = armArray[1]
            bodyInstance.img = armArray[2]

            cv2.imshow("left2", armArray[0])
            cv2.imshow("right2", armArray[1])
            # cv2.imshow("body2",bodyInstance.img)
            # 2. y축 위치 설정
            x_move = -94
            y_move = 90
            leg_move = 110

            # 바지위치
            rightInstance.setImage(frame, right[1] + y_move + leg_move, right[0] + x_move)
            bodyInstance.setImage(frame, right[1] + y_move, right[0] + x_move - 30)

            return frame
