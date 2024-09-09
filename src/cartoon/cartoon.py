import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 현재 파일(Cartoon.py)의 디렉토리 경로를 기준으로 모델 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.tflite')

# TensorFlow Lite 모델 로드
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 모델이 예상하는 입력 크기 가져오기
input_shape = input_details[0]['shape']

# 이미지 전처리 함수
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((input_shape[1], input_shape[2]))
    image = np.array(image).astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(image, axis=0)

# 이미지 후처리 함수
def postprocess_image(output_data):
    output_data = (output_data + 1.0) * 127.5
    output_data = np.clip(output_data, 0, 255).astype(np.uint8)
    return Image.fromarray(output_data[0])

# 이미지 변환 함수
def convert_image(input_image_path, output_image_path):
    input_data = preprocess_image(input_image_path, input_shape)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_image = postprocess_image(output_data)
    
    output_image.save(output_image_path)

# 변환할 이미지 경로 설정
input_image_path = os.path.join(current_dir, 'input.jpg')  # 변환할 인물 사진 경로
output_image_path = os.path.join(current_dir, 'output.jpg')  # 변환된 이미지 저장 경로

# 이미지 변환
convert_image(input_image_path, output_image_path)

print("이미지 변환 완료!")
