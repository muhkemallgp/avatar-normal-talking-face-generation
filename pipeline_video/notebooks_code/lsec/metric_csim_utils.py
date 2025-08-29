import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0, 
    min_detection_confidence=0.5
)

# Initialize FaceNet model for identity embeddings
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_face_embeddings(video_path, target_size=160):
    """Extract face embeddings from video frames using FaceNet"""
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            # Get the largest face detection
            detection = max(results.detections, 
                          key=lambda x: x.location_data.relative_bounding_box.width * 
                                      x.location_data.relative_bounding_box.height)
            
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = rgb_frame.shape
            
            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add padding
            padding = int(0.2 * min(width, height))
            x = max(0, x - padding)
            y = max(0, y - padding)
            width = min(w - x, width + 2 * padding)
            height = min(h - y, height + 2 * padding)
            
            # Extract face ROI
            face_roi = rgb_frame[y:y+height, x:x+width]
            
            if face_roi.size > 0:
                # Transform and get embedding
                face_tensor = transform(face_roi).unsqueeze(0)
                
                with torch.no_grad():
                    embedding = facenet(face_tensor)
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    print(f"Extracted {len(embeddings)} face embeddings from {video_path}")
    return np.array(embeddings) if embeddings else np.array([])

def calculate_csim(video_real, video_gen):
    """Calculate Cosine Similarity between identity embeddings of real and generated videos"""
    print("Extracting embeddings from real video...")
    embeddings_real = extract_face_embeddings(video_real)
    
    print("Extracting embeddings from generated video...")
    embeddings_gen = extract_face_embeddings(video_gen)
    
    if len(embeddings_real) == 0 or len(embeddings_gen) == 0:
        raise ValueError("No face embeddings extracted from one or both videos.")
    
    # Resample to same number of frames
    min_frames = min(len(embeddings_real), len(embeddings_gen))
    print(f"Resampling embeddings to {min_frames} frames")
    
    if len(embeddings_real) > min_frames:
        indices = np.linspace(0, len(embeddings_real)-1, min_frames, dtype=int)
        embeddings_real = embeddings_real[indices]
    
    if len(embeddings_gen) > min_frames:
        indices = np.linspace(0, len(embeddings_gen)-1, min_frames, dtype=int)
        embeddings_gen = embeddings_gen[indices]
    
    # Calculate cosine similarity for each frame pair
    similarities = []
    for i in range(min_frames):
        emb_real = embeddings_real[i]
        emb_gen = embeddings_gen[i]
        
        # Normalize embeddings
        emb_real_norm = emb_real / np.linalg.norm(emb_real)
        emb_gen_norm = emb_gen / np.linalg.norm(emb_gen)
        
        # Calculate cosine similarity
        cosine_sim = np.dot(emb_real_norm, emb_gen_norm)
        similarities.append(cosine_sim)
    
    # Return average similarity
    avg_csim = np.mean(similarities)
    print(f"Average CSIM score: {avg_csim:.4f}")
    return avg_csim, similarities