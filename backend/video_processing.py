# video_processing.py: 包含影片處理和姿勢分析邏輯，負責從影片中提取關鍵點、計算關節角度並生成建議運動範圍。

import cv2
import numpy as np
import os
import logging
from config import POSE_PROCESSOR, JOINT_PAIRS

logger = logging.getLogger(__name__)

# 計算由三個關鍵點形成的夾角。返回夾角的度數 (float)，若關鍵點無效則返回 0.0。
def calculate_angle(a, b, c):
    if not all([a, b, c]): return 0.0
    a = np.array([a.x, a.y]); b = np.array([b.x, b.y]); c = np.array([c.x, c.y])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# 從 MediaPipe 檢測到的姿勢關鍵點中提取預定義的關節角度。返回包含各關節角度的字典。
def get_angles_from_landmarks(landmarks):
    lm = landmarks.landmark
    pl = POSE_PROCESSOR.PoseLandmark 
    return {
        "L Shoulder": calculate_angle(lm[pl.LEFT_ELBOW], lm[pl.LEFT_SHOULDER], lm[pl.RIGHT_SHOULDER]),
        "R Shoulder": calculate_angle(lm[pl.RIGHT_ELBOW], lm[pl.RIGHT_SHOULDER], lm[pl.LEFT_SHOULDER]),
        "L Elbow": calculate_angle(lm[pl.LEFT_SHOULDER], lm[pl.LEFT_ELBOW], lm[pl.LEFT_WRIST]),
        "R Elbow": calculate_angle(lm[pl.RIGHT_SHOULDER], lm[pl.RIGHT_ELBOW], lm[pl.RIGHT_WRIST]),
        "L Armpit": calculate_angle(lm[pl.LEFT_ELBOW], lm[pl.LEFT_SHOULDER], lm[pl.LEFT_HIP]),
        "R Armpit": calculate_angle(lm[pl.RIGHT_ELBOW], lm[pl.RIGHT_SHOULDER], lm[pl.RIGHT_HIP]),
        "L Waist": calculate_angle(lm[pl.LEFT_SHOULDER], lm[pl.LEFT_HIP], lm[pl.LEFT_KNEE]),
        "R Waist": calculate_angle(lm[pl.RIGHT_SHOULDER], lm[pl.RIGHT_HIP], lm[pl.RIGHT_KNEE]),
        "L Knee": calculate_angle(lm[pl.LEFT_HIP], lm[pl.LEFT_KNEE], lm[pl.LEFT_ANKLE]),
        "R Knee": calculate_angle(lm[pl.RIGHT_HIP], lm[pl.RIGHT_KNEE], lm[pl.RIGHT_ANKLE]),
    }

# 根據平均角度和範圍寬度，計算建議的最小和最大角度範圍。返回包含 "min" 和 "max" 鍵的字典。
def calculate_suggested_range(angle, range_width=20, round_to=5):
    delta = range_width / 2
    if (angle - delta) < 0:
        raw_lower = 0
        raw_upper = range_width
    elif (angle + delta) > 180:
        raw_lower = 180 - range_width
        raw_upper = 180
    else:
        raw_lower = angle - delta
        raw_upper = angle + delta
    final_lower = round(raw_lower / round_to) * round_to
    final_upper = round(raw_upper / round_to) * round_to
    return {"min": int(final_lower), "max": int(final_upper)}

 # 處理上傳影片，執行姿勢分析，返回每個捕獲狀態的平均關節角度和建議範圍。
 # 若影片無法打開或未檢測到姿勢則拋出 IOError/ValueError。
def process_video_data(video_file, captures, mirror_settings, range_width):
    temp_video_path = f"temp_{video_file.filename}"
    video_file.save(temp_video_path)

    raw_angles_by_status = {}
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        logger.error("Could not open video file.")
        os.remove(temp_video_path)
        raise IOError("Could not open video file")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    for capture in captures:
        status_name = capture['statusName']
        timestamp = capture['time']
        frame_number = int(timestamp * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = POSE_PROCESSOR.process(image)
            if results.pose_landmarks:
                angles = get_angles_from_landmarks(results.pose_landmarks)
                if status_name not in raw_angles_by_status:
                    raw_angles_by_status[status_name] = []
                raw_angles_by_status[status_name].append(angles)
            else:
                logger.warning(f"No pose detected in frame {frame_number} for status '{status_name}'.")
    cap.release()
    os.remove(temp_video_path)

    if not raw_angles_by_status:
        raise ValueError("No poses could be detected in any of the captured frames.")

    processed_angles = {}
    for status, captured_frames in raw_angles_by_status.items():
        if not captured_frames: continue

        status_averages = {}
        for pair_name, (l_joint, r_joint) in JOINT_PAIRS.items():
            l_angles = [frame[l_joint] for frame in captured_frames]
            r_angles = [frame[r_joint] for frame in captured_frames]
            if mirror_settings.get(pair_name, False):
                combined_angles = l_angles + r_angles
                if combined_angles:
                    unified_avg = np.mean(combined_angles)
                    status_averages[l_joint] = unified_avg
                    status_averages[r_joint] = unified_avg
            else:
                if l_angles: status_averages[l_joint] = np.mean(l_angles)
                if r_angles: status_averages[r_joint] = np.mean(r_angles)

        suggested_ranges = {}
        for joint_name, avg_angle in status_averages.items():
            suggested_ranges[joint_name] = calculate_suggested_range(avg_angle, range_width=range_width)

        processed_angles[status] = {"angles": suggested_ranges}

    return processed_angles

