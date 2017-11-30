import cv2
import math

class BitwiseImage:
    def __init__(self, img):
        self.img = img
        self.origin = img

    def setImage(self, frame, y, x):
        rows,cols,channels = self.img.shape

        # y,x의 좌표 위치부터 ROI 지정
        roi = frame[y:rows+y, x:cols+x]
        #roi = frame[0:rows,0:cols]

        # create mask from logo
        img2gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # black out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi,roi, mask = mask_inv)

        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(self.img,self.img,mask = mask)

        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg,img2_fg)
        frame[y:rows+y, x:cols+x ] = dst
        #frame[0:rows, 0:cols] = dst


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

        #pre_path = 'c:\\opencv\\library\\opencv-master\\opencv-master\\data\\haarcascades\\'
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
        #print(found)

        for (x, y, w, h) in found:
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            detector.box_coordinate = [(x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h)]

        detector.draw_detections(frame, found)

        return frame


class listOfClothes :

    def __init__(self):
        self.array = [["blue","pink","brown"],["red","orange","suit"]] #옷 폴더 이름

    def getClothes (self,i,j) :
        return self.array[i][j]

class overlayer:
    isResize = False
    img_array = listOfClothes()


    def overlay(self,frame_detected, box_coordinate, clothesIndex):

        img = self.img_array.getClothes( clothesIndex[0],clothesIndex[1])
        clothesToWear = BitwiseImage(cv2.imread("clothes/"+img+"/body.png") )#입을 옷을 Bitewise 세팅

        if box_coordinate is None :
            print("no!! box_coordinate")
            return frame_detected
        else :

            #print(box_coordinate)
            frame = frame_detected

            x1,y1 = box_coordinate[0]
            x2,y2 = box_coordinate[1]

            if x1 is None :
                return frame
            else :
                #if (not self.isResize):
                # 1.크기 설정
                #if not self.isResize :
                ratio = math.ceil(x2 - x1)*6 #4.3 기준
                ratio = 312
                r = ratio / clothesToWear.img.shape[1]
                dim = (int(ratio), int(clothesToWear.img.shape[0] * r))
                resized = cv2.resize(clothesToWear.img, dim, interpolation=cv2.INTER_AREA)
                img2 = resized
               # rows, cols, channels = img2.shape
                clothesToWear.img = img2
                self.isResize = True

                # 2. y축 위치 설정
                x_move = -120
                y_move = int((y2 - y1))-20

                # 사람이 감지되었다고 가정
                cv2.imshow("body",clothesToWear.img)
                clothesToWear.setImage(frame,y1+y_move, x1+x_move)
                return frame
class arm_overlayer:
    img_array = listOfClothes()
    isResize = False

    def rotationDegree(self, x1, y1, x2, y2):

        return (-math.atan((y2 - y1) / (x2 - x1)) * (180 / 3.141592))

    def overlay(self, frame_detected, box_coordinate, hand, clothesIndex):

        left = hand[0][0] #왼손좌표
        right = hand[1][0] #오른손좌표

        img = self.img_array.getClothes(clothesIndex[0], clothesIndex[1])

        leftInstance =  BitwiseImage(cv2.imread("clothes/" + img + "/left.png")) # 입을 옷을 Bitewise 세팅
        rightInstance = BitwiseImage(cv2.imread("clothes/" + img + "/right.png"))

        leftToWear = leftInstance.img  # 입을 옷 이미지
        rightToWear = rightInstance.img
        armArray = [leftToWear,rightToWear] # for 문 돌리기 위해서 list화

        if box_coordinate is None :
            return frame_detected
        else :
            frame = frame_detected
            print(box_coordinate)
            x1 , y1 = box_coordinate[0]
            x2 , y2 = box_coordinate[1]

            for i in range(2) :
                #if (not self.isResize):
                # 1.크기 설정
                ratio = (x2 - x1)*5 #4.3 기준
                r = ratio / armArray[i].shape[1]
                dim = (int(ratio), int(armArray[i].shape[0] * r))
                resized = cv2.resize(armArray[i], dim, interpolation=cv2.INTER_AREA)
                armArray[i] = resized

                #회전
                num_rows, num_cols = armArray[i].shape[:2]
                degree = self.rotationDegree( x2, y2,right[0] ,right[1] )
                degree = degree if i==1 else degree*-1 # 왼손 오른손 구분
                rotation_matrix = cv2.getRotationMatrix2D((num_cols/2-20,num_rows/2-20),degree , 1)
                imgRotation = cv2.warpAffine(armArray[i], rotation_matrix , (num_cols, num_rows))
                armArray[i] = imgRotation

                # 2. y축 위치 설정
                x_move = -100
                y_move = -100

                #leftInstance.setImage(frame,y2+y_move, x2+x_move)

            cv2.imshow("left", armArray[0])
            cv2.imshow("right", armArray[1])
            return frame

