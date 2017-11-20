import cv2
import numpy as np
import time


def draw_text(frame, text, x, y, color=(255, 255, 255), thickness=20, size=5):
    if x is not None and y is not None:
        cv2.putText(
            frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


def main():
    cap = cv2.VideoCapture(0)

    init_time = time.time()

    counter = 5 # 화면에 띄울 숫자
    end_time = init_time + counter + 1 # 타이머 끝나는 시간
    secondPassed = init_time + 1 # 1초가 지났는지 안지났는지 비교하는용

    timer_over = False # 타이머 끝났는지 확인용

    while (cap.isOpened()):
        ret, frame = cap.read()

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
                timer_over == True
                break
        else:
            break

    # while(timer_over == True):
        # optical flow
        # UI
        # Detections



    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()