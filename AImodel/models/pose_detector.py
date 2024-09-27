import csv
import copy
import itertools
import cv2
import numpy as np
import mediapipe as mp
import joblib
import os


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark[0:25]:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    return temp_landmark_list

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark[11:25]):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_bounding_rect(use_brect, image, brect, rect_color):
    if use_brect:
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)
    return image

def draw_info_text(image, brect, facial_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    if facial_text != "":
        info_text = 'Pose: ' + facial_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def load_model(model_path):
    return joblib.load(model_path)

def body(vid, model_path):
    pos = 0
    crossed = 0
    raised = 0
    explain = 0
    straight = 0
    face = 0
    count = 0
    cap_width = 1920
    cap_height = 1080
    output_frames = []

    mp_draw = mp.solutions.drawing_utils
    use_brect = True

    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    xg_boost_model = load_model(model_path)

    # CSV 파일 경로 수정
    csv_path = os.path.join(os.path.dirname(__file__), 'pose_keypoint_classifier_label.csv')
    try:
        with open(csv_path, encoding='utf-8-sig') as f:
            keypoint_classifier_labels = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return [], f"Error: CSV file not found at {csv_path}", 0

    while True:
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)

        height, width, _ = debug_image.shape
        scaling_factor = 800 / max(height, width)
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)
        debug_image = cv2.resize(debug_image, (new_width, new_height))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks is not None:
            count += 1
            brect = calc_bounding_rect(debug_image, results.pose_landmarks)
            landmark_list = calc_landmark_list(debug_image, results.pose_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            facial_emotion_id = xg_boost_model.predict([pre_processed_landmark_list])[0]

            if facial_emotion_id == 0:
                crossed += 1
            elif facial_emotion_id == 1:
                raised += 1
            elif facial_emotion_id == 2:
                explain += 1
            elif facial_emotion_id == 3:
                straight += 1
            elif facial_emotion_id == 4:
                face += 1

            if facial_emotion_id in [2, 3]:
                rect_color = (0, 255, 0)
                pos += 1
            else:
                rect_color = (0, 0, 255)

            debug_image = draw_bounding_rect(use_brect, debug_image, brect, rect_color)
            debug_image = draw_info_text(debug_image, brect, keypoint_classifier_labels[facial_emotion_id])

        output_frames.append(debug_image)

    cap.release()

    try:
        pos_score = (pos / count) * 100
        crosed_score = (crossed / count) * 100
        raised_score = (raised / count) * 100
        face_score = (face / count) * 100

        messagep = 'YOUR POSITIVE AREAS: '
        messagen = 'NEEDS IMPROVEMENT: '

        if pos_score >= 70:
            messagep += " Good job on sitting straight and using hand gestures."
        else:
            messagen += " Sit straight and use hand gestures."

        if crosed_score >= 10:
            messagen += " Don't cross your arms."

        if raised_score >= 10:
            messagen += " Don't raise your arms."

        if face_score >= 10:
            messagen += " Don't touch your face."

        if messagep == 'YOUR POSITIVE AREAS: ':
            messagep = ''
        if messagen == 'NEEDS IMPROVEMENT: ':
            messagen = ''

        message = messagep + '\n\n' + messagen

    except Exception as e:
        print(f"Error in pose analysis: {str(e)}")
        pos_score = 0
        message = f'Error in pose analysis: {str(e)}'

    return output_frames, message, pos_score
