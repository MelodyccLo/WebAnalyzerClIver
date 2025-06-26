import mediapipe as mp

# MediaPipe 姿勢處理器初始化：配置 MediaPipe Pose 模型，用於從圖像中檢測和追蹤人體關鍵點。
mp_pose = mp.solutions.pose
POSE_PROCESSOR = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 關節對定義：定義了身體各部位的關節對，每個關節對包含左右兩個關節的名稱。
JOINT_PAIRS = {
    "Shoulder": ("L Shoulder", "R Shoulder"),
    "Elbow": ("L Elbow", "R Elbow"),
    "Armpit": ("L Armpit", "R Armpit"),
    "Waist": ("L Waist", "R Waist"),
    "Knee": ("L Knee", "R Knee"),
}

# 所有關節列表 (從 JOINT_PAIRS 派生)：包含了所有定義的關節名稱，用於後續處理。
ALL_JOINTS = [item for pair in JOINT_PAIRS.values() for item in pair]
