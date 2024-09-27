import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image


def analyze(vid):
    return head_eye(vid)
#landmarks

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

#iris
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_iris_center = [468]
R_iris_center = [473]

CHIN = [167, 393]

THAADI = [200]

NOSE = [4]

LH_LEFT = [33]
LH_RIGHT = [133]
RH_LEFT = [362]
RH_RIGHT = [263]

#colors

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)

#HEAD-EYE FUNCTIONS

def find_leftmost_rightmost(coordinates):
    leftmost = (float('inf'), float('inf'))
    rightmost = (-float('inf'), -float('inf'))
    
    for x, y in coordinates:
        leftmost = (min(leftmost[0], x), min(leftmost[1], y))
        rightmost = (max(rightmost[0], x), max(rightmost[1], y))
    
    return leftmost, rightmost

def transform_coordinates(coordinates):
    leftmost, rightmost = find_leftmost_rightmost(coordinates)
    
    # Calculate the scaling factor
    scaling_factor = 100 / (rightmost[0] - leftmost[0])
    
    transformed_coordinates = []
    for x, y in coordinates:
        # Translate to (50, 50)
        translated_x = x - leftmost[0] + 50
        translated_y = y - leftmost[1] + 50
        
        # Scale the coordinates
        scaled_x = translated_x * scaling_factor
        scaled_y = translated_y * scaling_factor
        
        transformed_coordinates.append([int(scaled_x), int(scaled_y)])
    
    return [np.array(transformed_coordinates)]

def landmarkdet(img, results):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

    return mesh_coord

#euclidean dist
def eucli(p1, p2):
    x, y = p1
    x1, y1 = p2
    dist = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return dist

def head_pose_estimate(model_points, landmarks, K):
    dist_coef = np.zeros((4, 1))
    ret, rvec, tvec = cv2.solvePnP(model_points, landmarks, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)

    rot_mat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rot_mat, np.zeros((3, 1), dtype=np.float64)))
    eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
    yaw   = int(eulerAngles[1, 0]*360)
    pitch = int(eulerAngles[0, 0]*360)
    roll  = eulerAngles[2, 0]*360
    return roll, yaw, pitch

def newirispos2(transformed_eye_coordinates, image):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p1 = flat_cords[0]
    p4 = flat_cords[8]
    iris = flat_cords[17]

    p = (p1+p4)/2

    con = p-iris
    con = (abs(con[0]), abs(con[1]))

    point1_int = (int(p[0]), int(p[1]))
    point2_int = (int(iris[0]), int(iris[1]))

    cv2.circle(image, point1_int, 5, (0, 0, 255), -1)  # Red color for point1
    cv2.circle(image, point2_int, 5, (0, 255, 0), -1)  # Green color for point2

    return con

#blink ratio
def newbratio(transformed_eye_coordinates):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p2 = flat_cords[13]
    p6 = flat_cords[3]
    p3 = flat_cords[11]
    p5 = flat_cords[5]
    right = flat_cords[0]
    left = flat_cords[8]

    earclosed = ((5.385164807134504 + 4.47213595499958) / 2*(eucli(right, left)))
    earopen = ((35.12833614050059 + 31.400636936215164) / 2*(eucli(right, left)))

    ear = (eucli(p2,p6) + eucli(p3,p5))/2*(eucli(right, left))

    thresh = (earopen + earclosed)/2

    if ear<=thresh:
        return True
    else:
        return False

def draw_bounding_rect(use_brect, image, brect, rect_color):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# def draw_info_text(image, brect, facial_text):
#     info_text =''
#     cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
#                  (0, 0, 0), -1)

#     if facial_text != "":
#         info_text = facial_text
#     cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
#                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

#     return image
def put_korean_text(image, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)


def head_eye(vid):
    count = 0
    text=''

    eyecount = 0
    headcount = 0
    straight = 0
    blinkcount = 0
    blinklist = []
    fps=0
    prev = 0
    consecutive_blink = 0
    blink_too_long =0
    output_frames = []

    map_face_mesh = mp.solutions.face_mesh
    rect_color = (0, 255, 0)  # Green
    font_path = r"The-Interview-Buster-main\app\utils\NanumBarunGothic.ttf"

    cap = cv2.VideoCapture(vid)

    output_file = r'D:\CIR\teamProj\test2\The-Interview-Buster-main\media\eye-contact.mp4'

    # head_eye.py에서 exit() 제거 또는 적절한 예외 처리로 수정
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return [], "Could not open video source", 0, 0  # 에러 메시지와 함께 빈 리스트 반환

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
    fps = 30.0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    with map_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape
            scaling_factor = 800 / max(height, width)
            new_height = int(height * scaling_factor)
            new_width = int(width * scaling_factor)
            frame = cv2.resize(frame, (new_width, new_height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fps += 1

            face_3d = []
            face_2d = []
            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    count += 1

                    if fps % 1441 == 0:
                        blinklist.append(blinkcount)
                        blinkcount = 0
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            if idx == 1:
                                nose_2d = (lm.x * width, lm.y * height)
                                nose_3d = (lm.x * width, lm.y * height, lm.z * 3000)
                            x, y = int(lm.x * width), int(lm.y * height)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    focal_length = 1 * width
                    cam_matrix = np.array([[focal_length, 0, height / 2],
                                           [0, focal_length, width / 2],
                                           [0, 0, 1]])

                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rot_matrix, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_matrix)

                    x = angles[0] * 360  # pitch
                    y = angles[1] * 360  # yaw

                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                    p1 = (int(nose_2d[0]) - 100, int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y * 10) - 100, int(nose_2d[1] - x * 10))

                    if not (-5 < int(y) < 5 and -5 < int(x) < 5):
                        straight = 0
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Look straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                    else:
                        straight = 1

                mesh_coords = landmarkdet(frame, results)
                mesh_points = np.array(mesh_coords)
                fhead = tuple(mesh_points[151])
                chin = tuple(mesh_points[175])
                threshold = 10

                if straight == 1:
                    if abs(fhead[0] - chin[0]) < threshold:
                        straight = 1
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Head straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                        headcount += 1
                    else:
                        straight = 0
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Look straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                x1, y1 = mesh_points[LOWER_LIPS][0]
                x2, y2 = mesh_points[LOWER_LIPS][10]
                x3, y3 = mesh_points[LIPS][25]
                x4, y4 = mesh_points[THAADI][0]
                x5, y5 = mesh_points[CHIN][0]
                x6, y6 = mesh_points[CHIN][1]
                eye_coordinates = []
                eye_cont_coordinates = []
                r_eye_cont_coordinates = []

                for i in LEFT_EYE:
                    eye_coordinates.append(tuple(mesh_points[i]))

                LEFT_EYE_and_IRIS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 468, 473]
                RIGHT_EYE_and_IRIS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 473, 468]

                for i in RIGHT_EYE_and_IRIS:
                    r_eye_cont_coordinates.append(tuple(mesh_points[i]))
                
                for i in LEFT_EYE_and_IRIS:
                    eye_cont_coordinates.append(tuple(mesh_points[i]))

                transformed_eye_coordinates = transform_coordinates(eye_coordinates)
                transformed_eyecont_coordinates = transform_coordinates(eye_cont_coordinates)
                rtransformed_eyecont_coordinates = transform_coordinates(r_eye_cont_coordinates)
                blink = newbratio(transformed_eye_coordinates)

                if blink:
                    if prev == 1:
                        if consecutive_blink <= 72:
                            consecutive_blink += 1
                        else:
                            blink_too_long = 1
                    prev = 1
                else:
                    if prev == 1:
                        blinkcount += 1
                    prev = 0
                    consecutive_blink = 0

                cont = newirispos2(transformed_eyecont_coordinates, frame)
                rcont = newirispos2(rtransformed_eyecont_coordinates, frame)

                if 0 <= ((cont[0] + rcont[0]) / 2) <= 2.5 and 0 <= ((cont[1] + rcont[1]) / 2) <= 3.5:
                    contact = True
                else:
                    contact = False

                if not blink:
                    if contact:
                        text = 'Eye Contact'
                        rect_color = (0, 255, 0)  # Green
                        eyecount += 1
                    else:
                        text = 'No Eye Contact'
                        rect_color = (0, 0, 255)  # Red
                else:
                    text = 'Blinking'
                    rect_color = (0, 0, 255)  # Red

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    brect = calc_bounding_rect(frame, face_landmarks)
                    frame = draw_bounding_rect(True, frame, brect, rect_color)
                    # Draw English text using OpenCV
                    cv2.putText(frame, text, (brect[0] + 5, brect[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
            out.write(frame)  # Write the frame to the output video
            output_frames.append(frame)

    try:
        head_score = ((headcount / count) * 100)
        eye_score = ((eyecount / count) * 100)

        messagep = '긍정적인 부분: '
        messagen = '개선이 필요한 부분: '

        if head_score <= 50:
            messagen += f"머리를 대부분 똑바로 유지하지 못했습니다. 머리를 똑바로 유지하지 않으면 면접관에게 불안정하거나 자신감이 부족하다는 인상을 줄 수 있습니다. 평균적으로 {int(head_score)}%의 시간 동안 머리를 똑바로 유지했습니다. 자세를 똑바로 유지하는 연습이 필요합니다."
        elif 50 < head_score <= 90:
            messagen += f"머리 자세는 비교적 안정적이었지만, 조금 더 일관되게 유지하는 것이 좋습니다. 면접 중에는 집중력과 자신감을 나타낼 수 있도록 머리를 똑바로 유지하는 것이 중요합니다. 평균적으로 {int(head_score)}%의 시간 동안 머리를 똑바로 유지했습니다."
        elif head_score > 90:
            messagep += f"머리를 매우 잘 유지했습니다! 전체 시간의 {int(head_score)}% 동안 머리를 똑바로 유지하며, 이는 면접관에게 안정적이고 신뢰감을 주는 인상을 남겼습니다."

        if eye_score <= 25:
            messagen += f"눈맞춤이 부족했습니다. 눈맞춤은 면접 중 자신감과 신뢰감을 전달하는 중요한 요소입니다. 평균적으로 {int(eye_score)}%의 시간 동안만 면접관과 눈을 맞추셨습니다. 조금 더 오랜 시간 동안 눈맞춤을 유지하는 연습이 필요합니다."
        elif 25 < eye_score <= 50:
            messagen += f"눈맞춤이 다소 부족한 편입니다. 면접관과 눈을 맞추는 시간은 면접 태도에 중요한 역할을 하므로, 평균적으로 {int(eye_score)}%의 시간 동안 눈맞춤을 유지했습니다. 조금 더 오랜 시간 동안 시선을 유지하는 것이 좋습니다."
        elif 50 < eye_score <= 75:
            messagen += f"눈맞춤이 괜찮지만, 더 오랜 시간 동안 유지할 필요가 있습니다. 평균적으로 {int(eye_score)}%의 시간 동안 눈맞춤을 유지했습니다. 면접관과의 눈맞춤을 통해 더 집중된 모습을 보여주세요."
        elif 75 < eye_score <= 90:
            messagep += f"눈맞춤을 상당히 잘 유지했습니다! 평균적으로 {int(eye_score)}%의 시간 동안 눈을 맞추셨으며, 이는 면접관에게 당신의 집중력과 준비성을 잘 전달하는 요소입니다."
        elif eye_score > 90:
            messagep += f"훌륭합니다! 평균적으로 {int(eye_score)}%의 시간 동안 눈맞춤을 유지했습니다. 이는 매우 강력한 자신감을 전달하며 면접에서 중요한 역할을 합니다."

        try:
            total_blink = sum(blinklist) / len(blinklist)
        except ZeroDivisionError:
            total_blink = 0

        if total_blink > 20:
            messagen += f"너무 많이 깜빡이고 있습니다. 평균적으로 분당 {int(total_blink)}번씩 깜빡였습니다. 깜빡임이 많으면 집중력이 부족하다는 신호일 수 있습니다. 깜빡임을 줄이는 연습을 통해 면접에서 좀 더 자신감 있는 모습을 보여주세요."

        if messagep == '긍정적인 부분: ':
            messagep = ''
        if messagen == '개선이 필요한 부분: ':
            messagen = ''

        message = messagep + "\n\n" + messagen

    except ZeroDivisionError:
        head_score, eye_score = 0, 0
        message = 'No face detected.'

    return output_frames, message, head_score, eye_score