import cv2
import dlib
import numpy as np

# Dlib의 얼굴 검출기와 랜드마크 검출기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("src/deer/shape_predictor_68_face_landmarks.dat")

# 사슴 뿔 이미지 로드 (PNG 파일로 알파 채널 포함)
horns_image = cv2.imread('src/deer/Data/deer_horns.png', cv2.IMREAD_UNCHANGED)
 
# 입력 이미지 로드
input_image_path = 'src/deer/Data/input_img.jpg'
output_image_path = 'src/deer/Data/output_image.jpg'
input_image = cv2.imread(input_image_path)

# 그레이스케일로 변환
gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# 얼굴 검출
faces = detector(gray)

# 얼굴이 검출되었는지 확인
if len(faces) > 0:
    for face in faces:
        landmarks = predictor(gray, face)

        # 사슴 뿔 위치 계산
        vertical_adjustment = 280  # 수직 위치 조정
        horizontal_adjustment = 0  # 수평 위치 조정

        # 이마의 중앙 좌표 계산
        forehead_x = (landmarks.part(19).x + landmarks.part(24).x) // 2 + horizontal_adjustment
        forehead_y = (landmarks.part(19).y + landmarks.part(24).y) // 2 - vertical_adjustment

        # 사슴 뿔 크기 조정
        scale_width_factor = 1.0  # 너비를 얼굴 너비로 조정
        scale_height_factor = 1.0  # 높이를 얼굴 높이로 조정

        horns_resized = cv2.resize(horns_image, 
                                   (int(face.width() * scale_width_factor), 
                                    int(face.height() * scale_height_factor)), 
                                   interpolation=cv2.INTER_AREA)
        horns_width = horns_resized.shape[1]
        horns_height = horns_resized.shape[0]

        # 오버레이 위치 계산
        x1 = max(0, forehead_x - horns_width // 2)
        x2 = min(input_image.shape[1], x1 + horns_width)
        y1 = max(0, forehead_y - horns_height // 2)
        y2 = min(input_image.shape[0], y1 + horns_height)

        # 프레임 경계 내에서만 블렌딩
        horns_x1 = max(0, -x1)
        horns_y1 = max(0, -y1)
        horns_x2 = horns_width - max(0, (x1 + horns_width) - input_image.shape[1])
        horns_y2 = horns_height - max(0, (y1 + horns_height) - input_image.shape[0])

        # 사슴 뿔을 얼굴 위에 오버레이 (알파 채널 사용)
        alpha_s = horns_resized[horns_y1:horns_y2, horns_x1:horns_x2, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            input_image[y1+horns_y1:y1+horns_y2, x1+horns_x1:x1+horns_x2, c] = (
                alpha_s * horns_resized[horns_y1:horns_y2, horns_x1:horns_x2, c] +
                alpha_l * input_image[y1+horns_y1:y1+horns_y2, x1+horns_x1:x1+horns_x2, c])

    # 결과 이미지 저장
    cv2.imwrite(output_image_path, input_image)
    print(f"Output image saved as {output_image_path}")

else:
    print("No face detected in the input image.")

