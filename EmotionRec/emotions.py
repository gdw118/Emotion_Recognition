import datetime
import tempfile
import os
import streamlit as st

import cv2
import numpy as np
import dlib
from imutils import face_utils
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input

USE_WEBCAM = False  # If false, loads video file source

# parameters for loading data and images
emotion_model_path = './models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')
# emotion_labels = get_class_to_arg('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []


# starting video streaming
def detect_attention(video_path='./test/speech.mp4'):
    distracted_ratio = 0
    last_print_frame_time = datetime.datetime.now()
    distracted_count = 0
    frame_count = 0
    current_frame = 0

    # Select video or webcam feed
    cap = None
    if USE_WEBCAM == True:
        cap = cv2.VideoCapture(0)  # Webcam source
    else:
        cap = cv2.VideoCapture(video_path)  # Video file source

    # 计算视频总帧数和帧率，以便确定结束前2秒的帧
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    threshold_frame = total_frames - 3 * fps

    with tempfile.TemporaryDirectory() as temp_dir:
        # 添加进度条
        st.write("视频处理中")
        progress_bar = st.progress(0)  # 初始化进度条

        while cap.isOpened():  # True:
            ret, bgr_image = cap.read()

            if not ret or bgr_image is None:
                print("无法读取视频帧，视频结束或读取出错")
                break

            if current_frame >= threshold_frame:
                break  # 倒数第3秒关闭视频

            current_frame += 1

            # 更新进度条
            progress_percentage = current_frame / threshold_frame  # 计算进度百分比
            if current_frame >= threshold_frame:
                progress_percentage = 100
            progress_bar.progress(progress_percentage)  # 更新进度条

            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            faces = detector(rgb_image)

            for face_coordinates in faces:

                x1, x2, y1, y2 = apply_offsets(face_utils.rect_to_bb(face_coordinates), emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, (emotion_target_size))
                except:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]
                emotion_window.append(emotion_text)

                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except:
                    continue

                color = emotion_probability * np.asarray((0, 255, 0))
                color = color.astype(int)
                color = color.tolist()

                draw_bounding_box(face_utils.rect_to_bb(face_coordinates), rgb_image, color)
                draw_text(face_utils.rect_to_bb(face_coordinates), rgb_image, emotion_mode,
                          color, 0, -45, 1, 1)

                print('EMOTION PROBABILITY -> ', emotion_probability)

                if emotion_text == 'NOT ATTENTIVE':
                    distracted_count += 1
                    now = datetime.datetime.now()
                    time_delta = now - last_print_frame_time
                    time_delta = time_delta.total_seconds()
                    if time_delta > 3:
                        ret, frame = cap.read()  # 读取视频帧
                        temp_image_path = os.path.join(temp_dir, f'distracted_frame_{frame_count}.jpg')
                        cv2.imwrite(temp_image_path, frame)
                        frame_count += 1
                        last_print_frame_time = now  # 更新最后保存帧的时间

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Attention Testing...', bgr_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # 列出临时目录中的所有图片
        dir_count = 0
        st.write("注意力不集中时刻：")
        for image_file in os.listdir(temp_dir):
            image_path = os.path.join(temp_dir, image_file)
            if os.path.isfile(image_path):
                # 显示图片
                st.image(image_path, caption=f"注意力不集中的时刻: {image_file}")
                dir_count += 1
        if dir_count == 0:
            st.success("本次表现很好，全程专注！")
        else:
            distracted_ratio = distracted_count / total_frames
            st.error("有注意力不集中的情况，占比为{:.2%}".format(distracted_ratio))

        return distracted_ratio


if __name__ == "__main__":
    detect_attention()
