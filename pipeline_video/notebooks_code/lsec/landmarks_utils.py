import cv2
import torch
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# --- MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

ALL_FACE_INDICES = list(range(468))

# --- Landmark generation ---
def generate_landmarks(face_roi):
    rgb_frame = cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return np.zeros((len(ALL_FACE_INDICES),2))
    
    landmarks = []
    h, w = face_roi.shape[:2]
    face_landmarks = results.multi_face_landmarks[0]
    for idx in ALL_FACE_INDICES:
        if idx < len(face_landmarks.landmark):
            x = face_landmarks.landmark[idx].x * w
            y = face_landmarks.landmark[idx].y * h
            landmarks.append([x, y])
        else:
            landmarks.append([0,0])
    return np.array(landmarks)

# --- Extract frames + landmarks ---
def extract_landmarks_from_video(video_path, target_size=256):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    frames, landmarks_list = [], []
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
            padding = int(0.2 * min(w,h))
            x, y, w, h = max(0,x-padding), max(0,y-padding), min(rgb_frame.shape[1]-x, w+2*padding), min(rgb_frame.shape[0]-y, h+2*padding)
            face_roi = rgb_frame[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (target_size, target_size))
            landmarks = generate_landmarks(face_roi_resized)
            frames.append(torch.from_numpy(face_roi_resized).permute(2,0,1).float()/255.0)
            landmarks_list.append(landmarks)
    
    cap.release()
    print(f"Extracted {len(frames)} frames and landmarks from {video_path}")
    return frames, landmarks_list

# --- Function to draw landmarks only on black background ---
def draw_landmarks_only(landmarks, img_size=(256, 256)):
    """Draw landmarks on black background without face image"""
    h, w = img_size
    # Create black background
    landmark_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw landmarks as colored circles
    for x, y in landmarks:
        if x > 0 and y > 0 and 0 <= x < w and 0 <= y < h:  # Only draw valid landmarks
            cv2.circle(landmark_img, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green circles
    
    return landmark_img

# --- Visualisasi perbandingan dengan landmarks only ---
def visualize_landmarks_comparison(frames_real, landmarks_real, frames_gen, landmarks_gen, n=5):
    # Check if both lists have frames
    if len(frames_real) == 0 or len(frames_gen) == 0:
        print("Error: One or both video lists are empty")
        return
    
    # Use minimum of both frame counts to avoid index errors
    min_frames = min(len(frames_real), len(frames_gen))
    n = min(n, min_frames)
    
    if n == 0:
        print("Tidak ada frame untuk divisualisasikan")
        return
    
    print(f"Visualizing {n} frames (Real: {len(frames_real)}, Gen: {len(frames_gen)}, Min: {min_frames})")
    
    # Create indices based on the minimum frame count for both videos
    indices = np.linspace(0, min_frames-1, num=n, dtype=int)
    
    plt.figure(figsize=(20, 12))
    
    for i in range(n):
        idx = indices[i]
        
        # Original Real Frame
        plt.subplot(4, n, i+1)
        img_real = (frames_real[idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        plt.imshow(img_real)
        plt.axis('off')
        plt.title(f"Real Frame {idx}", fontweight='bold')
        
        # Original Generated Frame
        plt.subplot(4, n, i+1+n)
        img_gen = (frames_gen[idx].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        plt.imshow(img_gen)
        plt.axis('off')
        plt.title(f"Gen Frame {idx}", fontweight='bold')
        
        # Real Landmarks Only (no face background)
        plt.subplot(4, n, i+1+2*n)
        landmarks_img_real = draw_landmarks_only(landmarks_real[idx])
        plt.imshow(landmarks_img_real)
        plt.axis('off')
        plt.title(f"Real Landmarks {idx}", fontweight='bold', color='green')
        
        # Generated Landmarks Only (no face background)
        plt.subplot(4, n, i+1+3*n)
        landmarks_img_gen = draw_landmarks_only(landmarks_gen[idx])
        plt.imshow(landmarks_img_gen)
        plt.axis('off')
        plt.title(f"Gen Landmarks {idx}", fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.show()