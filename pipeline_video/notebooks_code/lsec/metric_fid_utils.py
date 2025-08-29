import cv2
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import mediapipe as mp

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

# --- Landmark extraction (opsional) ---
def generate_landmarks(face_roi):
    rgb_frame = cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        return np.zeros((len(ALL_FACE_INDICES), 2))
    
    landmarks = []
    h, w = face_roi.shape[:2]
    face_landmarks = results.multi_face_landmarks[0]
    for idx in ALL_FACE_INDICES:
        if idx < len(face_landmarks.landmark):
            x = face_landmarks.landmark[idx].x * w
            y = face_landmarks.landmark[idx].y * h
            landmarks.append([x, y])
        else:
            landmarks.append([0, 0])
    return np.array(landmarks)

# --- Ekstraksi frame ROI wajah ---
def extract_face_frames(video_path, target_size=256):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
            padding = int(0.2 * min(w,h))
            x, y, w, h = max(0,x-padding), max(0,y-padding), min(rgb_frame.shape[1]-x, w+2*padding), min(rgb_frame.shape[0]-y, h+2*padding)
            face_roi = rgb_frame[y:y+h, x:x+w]
            face_roi_resized = cv2.resize(face_roi, (target_size, target_size))
            face_tensor = torch.from_numpy(face_roi_resized).permute(2,0,1).float()/255.0
            frames.append(face_tensor)
    
    cap.release()
    print(f"Extracted {len(frames)} frames from {video_path}")
    return frames

# --- Resample frames agar jumlah sama ---
def resample_frames(frames, target_count):
    if len(frames) <= target_count:
        return frames
    indices = np.linspace(0, len(frames)-1, num=target_count, dtype=int)
    return [frames[i] for i in indices]

# --- Convert frames ke tensor FID ---
def frames_to_tensor_fid(frames):
    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.Lambda(lambda x: (x*255).to(torch.uint8))
    ])
    return torch.stack([transform(f) for f in frames])

# --- Hitung FID ---
def calculate_fid(video_real, video_gen, device="cuda"):
    frames_real = extract_face_frames(video_real)
    frames_gen  = extract_face_frames(video_gen)

    if len(frames_real)==0 or len(frames_gen)==0:
        raise ValueError("Tidak ada frame yang terdeteksi di salah satu video.")

    target_frames = min(len(frames_real), len(frames_gen))
    print(f"Resampling frames to {target_frames}")

    frames_real = resample_frames(frames_real, target_frames)
    frames_gen  = resample_frames(frames_gen, target_frames)

    tensor_real = frames_to_tensor_fid(frames_real)
    tensor_gen  = frames_to_tensor_fid(frames_gen)

    fid = FrechetInceptionDistance().to(device)
    fid.update(tensor_real.to(device), real=True)
    fid.update(tensor_gen.to(device), real=False)
    score = fid.compute().item()
    print(f"FID score: {score}")
    return score
