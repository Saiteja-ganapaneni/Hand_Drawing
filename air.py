import cv2
import mediapipe as mp
import numpy as np
h = mp.solutions.hands
d = mp.solutions.drawing_utils
c = cv2.VideoCapture(0)
v = None
with h.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hand:
    p = None
    while True:
        r, f = c.read()
        if not r:
            break
        f = cv2.flip(f, 1)
        if v is None:
            v = np.zeros_like(f)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = hand.process(rgb)
        if res.multi_hand_landmarks:
            hm = res.multi_hand_landmarks[0]
            H, W, _ = f.shape
            x, y = int(hm.landmark[8].x * W), int(hm.landmark[8].y * H)
            cv2.circle(f, (x, y), 10, (0, 0, 255), -1)
            if p:
                cv2.line(v, p, (x, y), (255, 0, 0), 5)
            p = (x, y)
            d.draw_landmarks(f, hm, h.HAND_CONNECTIONS)
        else:
            p = None
        f = cv2.addWeighted(f, 0.5, v, 0.5, 0)
        cv2.imshow('Draw', f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
c.release()
cv2.destroyAllWindows()
