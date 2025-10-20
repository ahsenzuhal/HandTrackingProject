import cv2
import time
import os
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.75, trackCon=0.75):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw=True):
        if img is None:
            return None
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList

# ------------------ Finger Counter ------------------
def main():
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    folderPath = "FingerImages"
    myList = os.listdir(folderPath)
    overlayList = []
    for imPath in myList:
        image = cv2.imread(os.path.join(folderPath, imPath))
        if image is not None:
            overlayList.append(image)

    if not overlayList:
        print("Overlay resimleri bulunamadı!")
        return

    print(f"{len(overlayList)} overlay resmi yüklendi.")

    detector = handDetector(detectionCon=0.75)
    tipIds = [4, 8, 12, 16, 20]

    pTime = 0

    while True:
        success, img = cap.read()
        if not success or img is None:
            print("Görüntü alınamadı!")
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            fingers = []

            # Thumb
            if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # 4 fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)
            print(totalFingers)

            # Overlay security check
            if totalFingers > 0 and totalFingers <= len(overlayList):
                h, w, c = overlayList[totalFingers - 1].shape
                img[0:h, 0:w] = overlayList[totalFingers - 1]

            # Show the count in a box
            cv2.rectangle(img, (20, 225), (170, 425), (218, 160, 109), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 375),
                        cv2.FONT_HERSHEY_PLAIN, 10, (52, 52, 52), 25)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime - pTime > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (400, 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (52, 52, 52), 3)

        cv2.imshow("Finger Counter", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
