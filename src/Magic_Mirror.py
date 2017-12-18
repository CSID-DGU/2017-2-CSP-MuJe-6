# -*- coding:utf-8 -*-
import cv2
import numpy as np
import time
import random
import myUtills
import os


def draw_text(frame, text, x, y, color=(255, 255, 255), thickness=20, size=5):
    if x is not None and y is not None:
        cv2.putText(
            frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


def checkHandPosition(y, x):
    if (420 < x < 548):
        print("x범위 입니다. ")
        # if (y > 86 and< y < 110):
        if (80 < y < 110):
            print("위쪽 화살표 클릭")
            return 0
        # elif (y > 122 and y < 162):
        elif (110 < y < 162):
            print("첫번째 상자 클릭")
            return 1
        # elif (y > 174 and y < 214):
        elif (162 < y < 214):
            print("두번째 상자 클릭")
            return 2
        # elif (y > 226 and y < 266):
        elif (214 < y < 266):
            print("세번째 상자 클릭")
            return 3
        # elif (y > 278 and y < 302):
        elif (278 < y < 302):
            print("아래쪽 화살표 클릭")
            return 4
        return 100


def main():
    # cap = cv2.VideoCapture('video_2.mp4')
    cap = cv2.VideoCapture(1)

    # Random colors
    color = np.random.randint(0, 255, (100, 3))

    # Button check array
    btnCheckArray = [100, 100, 100]
    btnIdx = 0  # 배열에 들어갈 위치를 정하는 인덱스

    # backUI변화를 체크해 옷이 몇번째 배열에 있는지 확인
    clothesArrayIdx = 0

    while (True):

        init_time = time.time()

        counter = 20  # 화면에 띄울 숫자
        end_time = init_time + counter + 1  # 타이머 끝나는 시간
        secondPassed = init_time + 1  # 1초가 지났는지 안지났는지 비교하는용

        timer_over = False  # 타이머 끝났는지 확인용

        # 웹캠용
        left_hand = (200, 260)
        right_hand = (440, 260)
        left_foot = (260, 465)
        right_foot = (380, 465)

        # #video_3용
        # left_hand = (312, 610)
        # right_hand = (600, 430)
        # left_foot = (390, 940)
        # right_foot = (620, 950)



        while (cap.isOpened()):
            ret, frame = cap.read()
            # print(frame.shape)
            if ret != True:
                print("ret failed = ", ret)
                continue

            dot_color = random.choice(color)
            frame = cv2.flip(frame, 1)  # 좌우반전
            frame = cv2.circle(frame, left_hand, 10, dot_color.tolist(), -1)
            frame = cv2.circle(frame, right_hand, 10, dot_color.tolist(), -1)
            frame = cv2.circle(frame, left_foot, 10, dot_color.tolist(), -1)
            frame = cv2.circle(frame, right_foot, 10, dot_color.tolist(), -1)

            if ret == True:
                center_x = int(frame.shape[1] / 20)
                center_y = int(frame.shape[0] / 3.8)

                if (time.time() < end_time):
                    draw_text(frame, str(counter), center_x, center_y)
                if (time.time() > secondPassed):  # 1초가 지난 경우
                    counter -= 1
                    secondPassed += 1

                # frame = cv2.resize(frame, (360, 640))  # Resize image
                cv2.imshow('frame', frame)
                if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() > end_time):
                    timer_over = True
                    print("Timer over")
                    break
            else:
                #print("Failed to read frame, ret= ", ret)
                break

        # Lucas Kanade optical flow parameters
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        ##Background UI 삽입
        back_img = cv2.imread('back_img0.png')
        # back_img = cv2.resize(back_img, (640, 480))
        backgroundUI = myUtills.BitwiseImage(back_img)

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        if ret != True:
            print("Failed to read frame... ret=", ret)
            continue

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Set tracking points
        p0 = np.array(
            [[list(left_hand)], [list(right_hand)], [list(left_foot)], [list(right_foot)]])  # tracking하는 포인트 위치와 개수 설정
        p0 = np.float32(p0)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        # body detector and overlayer

        body_detector = myUtills.detector()

        clothes_overlayer = myUtills.overlayer()
        right_overlayer = myUtills.arm_overlayer()
        pants_overlayer = myUtills.pants_overlayer()
        kindOfClothes = myUtills.listOfClothes().whatClothes

        countframe = 0

        # clothes managing
        clothesIndex = [2,2]  # 더미변수 (checkPosition 에서 받을 예정)
        top = []
        pants = []

        headNotDetected = 0

        # 오버레이 루프
        while (True):

            # print(clothesIndex)


            ret, realframe = cap.read()
            countframe += 1
            if (ret == False):
                print("Inapproporiate frame, break")
                break
            # if (timeCode == True):
            #   e1 = cv2.getTickCount()

            if (countframe >  10):
                countframe = 0
            if (countframe >= 0):
                # 1. optical flow

                realframe = cv2.flip(realframe, 1)  # 좌우반전

                frame_gray = cv2.cvtColor(realframe, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                if (st.all() == 1):
                #if (True):

                    # draw the tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()  # 현재 프레임의 좌표값
                        c, d = old.ravel()  # 이전 프레임의 좌표값
                        # checkHandPosition(a,b)
                        #  print("트레킹 위치 :", a,b)
                        # mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)  # 현재와 이전의 프레임을 이어줌
                        realframe = cv2.circle(realframe, (a, b), 10, color[i].tolist(), -1)

                    # Now update the previous frame and previous points
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)

                # print("Optical flow complete")
                # 2.detect body

                realframe = body_detector.detect_body2(realframe)
                box_coordinate = body_detector.box_coordinate
                # print("Body detection complete")

                # 3. check hand position and run action
                #   print(countframe)
                btnCheckIdx = 0  # 배열의 원소 세개가 모두 같은지 확인하는 변수
                if (countframe == 0):  # 3초 정도의 delay
                    #      print("아아")
                    # leftPosition = checkHandPosition(p1[0][0][1], p1[0][0][0])  # 왼손 위치 체크
                    rightPosition = checkHandPosition(p1[1][0][1], p1[1][0][0])  # 오른손 위치 체크
                    btnCheckArray[btnIdx % 3] = rightPosition  # 배열에 손위치값 담음
                    btnIdx += 1
                    print("rightPosition", rightPosition)
                    for i, item in enumerate(btnCheckArray):  # 배열 원소 세개가 모두 같은지 확인
                        if (btnCheckArray[i] == btnCheckArray[2]):
                            btnCheckIdx += 1
                    if (btnCheckIdx == 3):  # 배열의 원소 세개가 모두 같으면
                        if (rightPosition == 0):
                            # 위쪽 화살표 클릭
                            clothesArrayIdx = (clothesArrayIdx + 2) % 3  # 가야되는 페이지
                            backgroundUI = myUtills.BitwiseImage(cv2.imread('back_img' + str(clothesArrayIdx) + '.png'))
                        elif (rightPosition == 1):
                            # 첫번째 아이템 클릭
                            clothesIndex = [clothesArrayIdx, 0]
                        elif (rightPosition == 2):
                            # 두번째 아이템 클릭
                            #print("clothesIndex[0 ]" ,clothesIndex[0],"clothesIndex[1 ]" ,clothesIndex[1])
                            clothesIndex = [clothesArrayIdx, 1]
                        elif (rightPosition == 3):
                            # 세번째 아이템 클릭
                            clothesIndex = [clothesArrayIdx, 2]
                        elif (rightPosition == 4):
                            # 아래쪽 화살표 클릭
                            clothesArrayIdx = (clothesArrayIdx + 1) % 3  # 가야되는 페이지
                            backgroundUI = myUtills.BitwiseImage(cv2.imread('back_img' + str(clothesArrayIdx) + '.png'))

                # 4.overlay clothes

                # print("left_h:",p1[0][0])
                # print("right_h:",p1[1][0])
                # print("left_f:", p1[2][0])
                # print("right_f:", p1[3][0])



                # #### overlay part start #####


                ##### Restart if head not detected ####
                # if (box_coordinate == None) & (countframe == 0):
                #     headNotDetected += 1
                #     print("Head not detected, ++")
                #     if (headNotDetected == 10):
                #         print("Head not detected, exit loop")
                #         break # 타이머로 감
                # if (box_coordinate != None):
                #     print("Head detected!, 0 again")
                #     headNotDetected = 0

                # #### If 'a' exit loop ####
                k = cv2.waitKey(10) & 0xff
                if k == 97:  # esc key
                    break

                # 상하의 구분
                #print("box_Coordinate: ", box_coordinate)
                print("clothesIndex: ", clothesIndex)
                if (box_coordinate != None):
                    if kindOfClothes[clothesIndex[0]][clothesIndex[1]] == 0:  # 0이면 상의, 1이면 하의
                        top = clothesIndex
                    elif kindOfClothes[clothesIndex[0]][clothesIndex[1]] == 1:
                        pants = clothesIndex

                    # 입히기
                if len(pants) != 0:
                    realframe = pants_overlayer.overlay(realframe, box_coordinate, p1, pants)

                if len(top) != 0:
                    realframe = right_overlayer.overlay(realframe, box_coordinate, p1, top)  # 프레임, 얼굴좌표, (왼좌표. 오른좌표)
                    realframe = clothes_overlayer.overlay(realframe, box_coordinate, top)

                # if (st.all() == 0) & (countframe == 0):
                #     print("St.all = 0")
                #     break

                #### overlay part end ####

                # Background UI 삽입
                backgroundUI.setImage(realframe, 0, 0)

                realframe = cv2.add(realframe, mask)

                # img = cv2.resize(img, (360, 640))  # Resize image
                cv2.imshow('frame', realframe)

                # 카메라 종료
                k = cv2.waitKey(30) & 0xff
                if k == 27:  # esc key
                    cap.release()
                    break

                    # Detections


if __name__ == '__main__':
    main()
