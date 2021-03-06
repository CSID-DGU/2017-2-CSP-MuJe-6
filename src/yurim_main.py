import cv2
import numpy as np
import time
import random
from src import myUtills
import threading

def draw_text(frame, text, x, y, color=(255, 255, 255), thickness=20, size=5):
    if x is not None and y is not None:
        cv2.putText(
            frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

def checkHandPosition(y,x):
    if (x>552 and x<687):
        if (y>131 and y<217):
            print("위쪽 화살표 클릭")
            return 0
        elif (y>247 and y<364):
            print("첫번째 상자 클릭")
            return 1
        elif (y>394 and y<511):
            print("두번째 상자 클릭")
            return 2
        elif (y>541 and y<658):
            print("세번째 상자 클릭")
            return 3
        elif (y>688 and y<774):
            print("아래쪽 화살표 클릭")
            return 4
    return 100

def main():

    #cap = cv2.VideoCapture('video_2.mp4')
    cap = cv2.VideoCapture(0)

    # Random colors
    color = np.random.randint(0,255,(100,3))

    # Button check array
    btnCheckArray = [100, 100, 100]
    btnIdx = 0  # 배열에 들어갈 위치를 정하는 인덱스

    # backUI변화를 체크해 옷이 몇번째 배열에 있는지 확인
    clothesArrayIdx = 0;

    while(True):

        init_time = time.time()

        counter = 5  # 화면에 띄울 숫자
        end_time = init_time + counter + 1  # 타이머 끝나는 시간
        secondPassed = init_time + 1  # 1초가 지났는지 안지났는지 비교하는용

        timer_over = False  # 타이머 끝났는지 확인용

        left_hand = (250, 100)
        right_hand = (480, 100)
        # left_arm = (220, 300)
        # right_arm = (480,300)

        while(cap.isOpened()):
            ret, frame = cap.read()

            dot_color = random.choice(color)
            frame = cv2.flip(frame, 1)  # 좌우반전
            frame = cv2.circle(frame, left_hand, 5, dot_color.tolist(), -1)
            frame = cv2.circle(frame, right_hand, 5, dot_color.tolist(), -1)

            if ret == True:
                center_x = int(frame.shape[1] / 2.5)
                center_y = int(frame.shape[0] / 2)

                if (time.time() < end_time):
                    draw_text(frame, str(counter), center_x, center_y)
                if (time.time() > secondPassed): # 1초가 지난 경우
                    counter -= 1
                    secondPassed += 1

                #frame = cv2.resize(frame, (360, 640))  # Resize image
                #frame = cv2.resize(frame, (360, 640))  # Resize image
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

        ##Background UI 삽입
        back_img = cv2.imread('back_img.png')
        #back_img = cv2.resize(back_img, (720, 1280))
        backgroundUI = myUtills.BitwiseImage(back_img)

        # Take first frame and find corners in it
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        # Set tracking points
        p0 = np.array([[list(left_hand)], [list(right_hand)]]) #tracking하는 포인트 위치와 개수 설정
        print("p0 초기 설정 위치")
        print(p0)
        p0 = np.float32(p0)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        # body detector and overlayer

        body_detector = myUtills.detector()
        clothes_overlayer = myUtills.overlayer()
        right_overlayer = myUtills.arm_overlayer()

        countframe = 0

        clothesIndex = [0, 1]  # 더미변수 (checkPosition 에서 받을 예정)
        # 오버레이 루프
        while(True):
            countframe += 1
            if (countframe > 100000):
                countframe = 0
            if (countframe >= 0):
                # 1. optical flow
                ret, realframe = cap.read()
                realframe = cv2.flip(realframe, 1)  # 좌우반전

                frame_gray = cv2.cvtColor(realframe, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                print("p1 좌표는")
                print(p1)
                print("p0 좌표는")
                print(p0)
                print("st 값은")
                print(st)
                if (st.all() == 1):   # optical flow의 좌표를 가져와서 프레임 안에 있을 때만 돌아가게 하기

                    # draw the tracks
                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        a, b = new.ravel()  # 현재 프레임의 좌표값
                        c, d = old.ravel()  # 이전 프레임의 좌표값
                        #checkHandPosition(a,b)
                        #print(a,b)
                        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)  # 현재와 이전의 프레임을 이어줌
                        realframe = cv2.circle(realframe, (a, b), 5, color[i].tolist(), -1)


                    # Now update the previous frame and previous points
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)

                # 2.detect body

                realframe = body_detector.detect_body2(realframe)
                box_coordinate = body_detector.box_coordinate

                # 3. check hand position and run action

                btnCheckIdx = 0  # 배열의 원소 세개가 모두 같은지 확인하는 변수
                if (countframe == 0): # 3초 정도의 delay
                    print("손 위치를 확인해 볼까나")
                    #leftPosition = checkHandPosition(p1[0][0][1], p1[0][0][0])  # 왼손 위치 체크
                    rightPosition = checkHandPosition(p1[1][0][1], p1[1][0][0])  # 오른손 위치 체크
                    btnCheckArray[btnIdx%3] = rightPosition #배열에 손위치값 담음
                    btnIdx += 1
                    print("전 ")
                    print(btnCheckIdx)
                    for i, item in enumerate(btnCheckArray): #배열 원소 세개가 모두 같은지 확인
                        if (btnCheckArray[i] == btnCheckArray[2]):
                            btnCheckIdx += 1
                    print("후 ")
                    print(btnCheckIdx)
                    if(btnCheckIdx == 3): #배열의 원소 세개가 모두 같으면
                        print("배열 원소 세개가 같음!!꺅")
                        if (rightPosition == 0):
                            # 위쪽 화살표 클릭
                            print("위")
                            if (not clothesArrayIdx == 0):  # 0인 경우는 첫번째 배열이므로 더이상 위쪽으로 넘어갈 수 없음
                                clothesArrayIdx -= 1
                                backgroundUI = myUtills.BitwiseImage(cv2.imread('back_img2.png'))
                        elif (rightPosition == 1):
                            # 첫번째 아이템 클릭
                            print("첫")
                            clothesIndex = [clothesArrayIdx, 0]
                        elif (rightPosition == 2):
                            # 두번째 아이템 클릭
                            clothesIndex = [clothesArrayIdx, 1]
                        elif (rightPosition == 3):
                            # 세번째 아이템 클릭
                            clothesIndex = [clothesArrayIdx, 2]
                        elif (rightPosition == 4):
                            # 아래쪽 화살표 클릭
                            if (clothesArrayIdx < 2):  # 배열 크기가 1이므로 그 이상은 배열이 없으므로 더이상 아래로 내려갈 수 없음
                                clothesArrayIdx += 1
                                backgroundUI = myUtills.BitwiseImage(cv2.imread('back_img3.png'))
                        else:
                            print("100이다이노망")

                # 4.overlay clothes

                #print("left:",p1[0][0])
                #print("right:",p1[1][0])
                #realframe = right_overlayer.overlay(realframe, box_coordinate, p1, clothesIndex) #프레임, 얼굴좌표, (왼좌표. 오른좌표)
                #realframe = clothes_overlayer.overlay(realframe, box_coordinate,clothesIndex)
                #realframe = clothes_overlayer.changeClothes(realframe, box_coordinate)


                # 카메라 종료
                k = cv2.waitKey(30) & 0xff
                if k == 27:  # esc key
                    cap.release()
                    break


                #Background UI 삽입
                #backgroundUI.setImage(realframe, 0, 0)

                if (st.all() == 1): # st == 1 이면 프레임 안
                    img = cv2.add(realframe, mask)
                else: # 프레임 밖으로 벗어난 경우
                    img = realframe
                    break

                #img = cv2.resize(img, (360, 640))  # Resize image
                cv2.imshow('frame', img)


                # Detections

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()