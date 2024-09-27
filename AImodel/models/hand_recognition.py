# The-Interview-Buster-main\hand_recognition.py
import csv
import copy
import argparse
import itertools
from collections import deque
import os
import cv2 as cv
import numpy as np
import mediapipe as mp
import joblib

GREEN = (0, 255, 0)

#FUNCTIONS

# FUNCTIONS

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
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
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    temp_landmark_list = list(map(lambda n: n / max_value, temp_landmark_list))
    return temp_landmark_list

def draw_info_text(image, brect, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    cv.putText(image, hand_sign_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

# Load the XGBoost model
def load_model():
    return joblib.load(r'hand_xgboost_model.pkl')

# Load labels for hand keypoint classification
with open('hand_keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

# Hand gesture analysis function without Streamlit
def analyze_hand_video(vid):
    cap = cv.VideoCapture(vid)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    xgboost_model = load_model()
    
    opencount = 0
    closecount = 0
    pointcount = 0
    count = 0
    output_frames = []

    while True:
        ret, image = cap.read()
        if not ret:
            break

        debug_image = copy.deepcopy(image)
        debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
        height, width = debug_image.shape[:2]

        scaling_factor = 800 / max(height, width)
        new_height = int(height * scaling_factor)
        new_width = int(width * scaling_factor)

        debug_image = cv.resize(debug_image, (new_width, new_height))

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                count += 1
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = xgboost_model.predict([pre_processed_landmark_list])[0]

                if hand_sign_id == 0:
                    opencount += 1
                    color = GREEN
                elif hand_sign_id == 1:
                    closecount += 1
                    color = (0, 0, 255)
                elif hand_sign_id == 2:
                    pointcount += 1
                    color = (0, 0, 255)

                cv.putText(debug_image, str(hand_sign_id), (100, 250), cv.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)
                debug_image = draw_info_text(debug_image, brect, keypoint_classifier_labels[hand_sign_id])

        output_frames.append(debug_image)

    cap.release()

    try:
        open_score = (opencount / count) * 100
        close_score = (closecount / count) * 100
        point_score = (pointcount / count) * 100

        message = ""
        if open_score >= 70:
            message += "Good job on using open hand gestures.\n"
        else:
            message += "Practice using open hand gestures.\n"

        if close_score >= 10:
            message += "Refrain from using closed hand gestures.\n"

        if point_score >= 10:
            message += "Don't point your fingers too much.\n"

    except ZeroDivisionError:
        open_score = 0
        message = "No hand gestures detected."

    return output_frames, message, open_score