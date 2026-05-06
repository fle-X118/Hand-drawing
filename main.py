import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

colors = [
    [0, 0, 255],    # Красный
    [0, 255, 0],    # Зеленый
    [255, 0, 0],    # Синий
    [0, 255, 255],  # Желтый
    [255, 0, 255],  # Маджента (Розовый)
    [255, 255, 0],  # Циан (Голубой)
    [255, 255, 255] # Белый
]
clens = len(colors)
chl = 0
# 1. Настройка размеров и модели
W_WIDTH, W_HEIGHT = 1280, 720
model_path = 'gesture_recognizer.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# 2. Камера (IP-камера или веб-камера)
# url = "http://10.120.243.239:8080/video"
# cap = cv2.VideoCapture(url)
cap = cv2.VideoCapture(0) # Раскомментируй, если используешь вебкамеру
cap.set(3, W_WIDTH)
cap.set(4, W_HEIGHT)

# Создаем холст и переменные состояния
canvas = np.zeros((W_HEIGHT, W_WIDTH, 3), np.uint8)
xp, yp = 0, 0

cv2.namedWindow('Air Drawing', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (W_WIDTH, W_HEIGHT))

    # Подготовка кадра для MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = recognizer.recognize(mp_image)

    # Логика рисования
    hand_detected = False
    if result.hand_landmarks:
        for i, landmarks in enumerate(result.hand_landmarks):
            hand_detected = True
            # Указательный палец (Landmark 8)
            finger_tip = landmarks[8]
            cx, cy = int(finger_tip.x * W_WIDTH), int(finger_tip.y * W_HEIGHT)

            # Проверка жеста (защита от пустого списка)
            gesture = "None"
            if i < len(result.gestures) and result.gestures[i]:
                gesture = result.gestures[i][0].category_name
            if gesture == "Thumb_Up":
                draw_color = (0, 0, 0)
                cv2.line(canvas, (xp, yp), (cx, cy), draw_color, 25)
            if gesture == "Open_Palm":
                chl = (chl + 1) % clens
            if gesture == "Victory":
                canvas = np.zeros((W_HEIGHT, W_WIDTH, 3), np.uint8)
            if gesture == "Thumb_Down":
                exit()
            if gesture == "Pointing_Up":
                current_color = colors[chl]
                cv2.circle(frame, (cx, cy), 10, (current_color), cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = cx, cy

                # Рисуем на холсте (плавная линия)
                cv2.line(canvas, (xp, yp), (cx, cy), (current_color), 5)
                xp, yp = cx, cy
            else:
                xp, yp = 0, 0

    if not hand_detected:
        xp, yp = 0, 0

    # Наложение холста на видео
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow('Air Drawing', frame)

    # Обработка клавиш (один вызов waitKey)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
