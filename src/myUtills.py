import cv2

class BitwiseImage:
    def __init__(self, img):
        self.img = img

    def setImage(self, frame, y, x):
        rows,cols,channels = self.img.shape

        roi = frame[y:rows+y, x:cols+x]

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

        pre_path = 'c:\\opencv\\library\\opencv-master\\opencv-master\\data\\haarcascades\\'

        path_lowerbody = pre_path + 'haarcascade_lowerbody.xml'
        path_fullbody = pre_path + 'haarcascade_fullbody.xml'
        path_face = pre_path + 'haarcascade_frontalface_default.xml'
        path_upper = pre_path + 'haarcascade_upperbody.xml'
        path_eye = pre_path + 'haarcascade_eye.xml'
        path_hog = pre_path + 'hogcascade_pedestrians.xml'

        face_cascade = cv2.CascadeClassifier(path_upper)

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

    def overlay(self, frame_detected, box_coordinate):


        if box_coordinate is None :
            print("no!! box_coordinate")
            return frame_detected
        else :

            print(box_coordinate)
            frame = frame_detected

            x1 = box_coordinate[0][0]
            y1 = box_coordinate[0][1]
            x2 = box_coordinate[1][0]
            y2 = box_coordinate[1][1]

            if x1 is None :
                return frame

            else :

                # 1.크기 설정
                ratio = (x2 - x1) #4.3 기준
                r = ratio / self.clothes.img.shape[1]
                dim = (int(ratio), int(self.clothes.img.shape[0] * r))
                resized = cv2.resize(self.clothes.img, dim, interpolation=cv2.INTER_AREA)
                img2 = resized
                rows, cols, channels = img2.shape
                self.clothes.img = img2


                # 2. y축 위치 설정
                x_move = -80
                y_move = int((y2 - y1)*1.5)

                # 사람이 감지되었다고 가정

                self.clothes.setImage(frame,y1+y_move, x1+x_move)
                return frame