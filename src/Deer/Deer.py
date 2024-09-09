import cv2
import dlib
import numpy as np
import os

# Dlib의 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 사슴 뿔 이미지 로드
horns_image = cv2.imread('deer_horns.png', cv2.IMREAD_UNCHANGED)

# 웹캠 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임을 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        vertical_adjustment = 120  # 상단으로 20 픽셀 이동
        horizontal_adjustment = 0  # 수평 이동 없음

        # 랜드마크 포인트 중 머리 위치에 해당하는 포인트 사용
        forehead_x = (landmarks.part(19).x + landmarks.part(24).x) // 2 + horizontal_adjustment
        forehead_y = (landmarks.part(19).y + landmarks.part(24).y) // 2 - vertical_adjustment
        
        # 사슴 뿔 이미지 크기 조정 및 위치 계산
        scale_width_factor = 1.0  # 너비를 20% 증가
        scale_height_factor = 1.0  # 높이를 얼굴 높이의 40%로 조정
                
        horns_resized = cv2.resize(horns_image, 
                                   (int(face.width() * scale_width_factor), 
                                    int(face.height() * scale_height_factor)), 
                                    interpolation=cv2.INTER_AREA)
        horns_width = horns_resized.shape[1]
        horns_height = horns_resized.shape[0]

        # 오버레이할 위치 계산
        x1 = max(0, forehead_x - horns_width // 2)
        x2 = min(frame.shape[1], x1 + horns_width)
        y1 = max(0, forehead_y - horns_height // 2)
        y2 = min(frame.shape[0], y1 + horns_height)

        # 프레임 경계 내에서만 블렌딩
        horns_x1 = max(0, -x1)
        horns_y1 = max(0, -y1)
        horns_x2 = horns_width - max(0, (x1 + horns_width) - frame.shape[1])
        horns_y2 = horns_height - max(0, (y1 + horns_height) - frame.shape[0])

        # 사슴 뿔을 얼굴 위에 오버레이
        alpha_s = horns_resized[horns_y1:horns_y2, horns_x1:horns_x2, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1+horns_y1:y1+horns_y2, x1+horns_x1:x1+horns_x2, c] = (
                alpha_s * horns_resized[horns_y1:horns_y2, horns_x1:horns_x2, c] +
                alpha_l * frame[y1+horns_y1:y1+horns_y2, x1+horns_x1:x1+horns_x2, c])

    # 변형된 프레임 표시
    cv2.imshow('Deer Horns Filter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
