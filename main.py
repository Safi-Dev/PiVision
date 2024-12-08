import cv2
from deepface import DeepFace
import os
import pickle
import uuid
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import time

# Reminder to change minsize for face when available
# Add more comments, please future self, I'm already struggling, man

'''
TAKE YOLO, improve object tracking
IMPLEMENT DWELL AND HEATMAP ANALYSIS, ADJUSTABLE PARAMETER FOR HEATMAP FADE TIME. 
ADJUST BASED UPON FOOT TRAFFIC IN FUTURE FIND GOOD BALANCE BASED UPON TRAFFIC
'''

@dataclass
class UserProfile:
    id: str
    face_encoding: np.ndarray
    face_image: np.ndarray
    age: int
    gender: str
    race: str
    timestamp: datetime

class FaceRecognitionSystem:
    def __init__(self):
        self.KNOWN_USERS_FILE = 'known_users.pkl'
        self.SNAPSHOTS_DIR = 'face_snapshots'
        self.MODEL_NAME = "Facenet512"
        self.PROCESS_EVERY_N_FRAMES = 5  
        self.SIMILARITY_THRESHOLD = 0.6  # Lower values are more strict, 0-1 range, tested 0.6 strikes good balance, especially because my laptop webcam sucks
        self.NO_DETECTION_TIMEOUT = 3  # Time in seconds before clearing overlay
        
        # Frame counter 
        self.fps = 0
        self.fps_start_time = time.time()
        self.frame_times = []
        
        # Cache for face recognition results
        self.last_recognition_result = None
        self.recognition_cache_frames = 0
        self.MAX_CACHE_FRAMES = 10  # Number of frames to keep the same recognition result, can be adjusted, untested performance benefits

        # Timer for clearing the overlay
        self.last_detection_time = None

        if not os.path.exists(self.SNAPSHOTS_DIR):
            os.makedirs(self.SNAPSHOTS_DIR)

        self.known_users = self.load_known_users()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def load_known_users(self) -> Dict[str, UserProfile]:
        try:
            if os.path.exists(self.KNOWN_USERS_FILE):
                with open(self.KNOWN_USERS_FILE, 'rb') as file:
                    return pickle.load(file)
            return {}
        except Exception as e:
            print(f"Error loading known users: {e}")
            return {}

    def save_known_users(self):
        try:
            with open(self.KNOWN_USERS_FILE, 'wb') as file:
                pickle.dump(self.known_users, file)
        except Exception as e:
            print(f"Error saving known users: {e}")

    @staticmethod
    def calculate_cosine_similarity(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        return np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))

    def analyse_face_snapshot(self, face_image: np.ndarray) -> Optional[Dict]:  # Changed analyze to analyse
        try:
            analysis = DeepFace.analyze(
                face_image,
                actions=['age', 'gender', 'race'],
                enforce_detection=False,
                align=True,
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]

            return {
                'age': analysis.get('age'),
                'gender': analysis.get('dominant_gender', 'Unknown'),
                'race': analysis.get('dominant_race'),
            }
        except Exception as e:
            print(f"Error analysing face: {e}")  # Changed analyzing to analysing
            return None

    def get_face_encoding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        try:
            representation = DeepFace.represent(
                face_image,
                model_name=self.MODEL_NAME,
                enforce_detection=False,
            )
            
            if representation and isinstance(representation, list):
                return np.array(representation[0].get('embedding'))
            return None
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return None

    def recognise_face(self, face_image: np.ndarray, force_new_recognition: bool = False) -> Optional[UserProfile]:  # Changed recognize to recognise
        # Use cached result if available and not forced to do new recognition and cache limit not reached
        if not force_new_recognition and self.last_recognition_result and self.recognition_cache_frames < self.MAX_CACHE_FRAMES:
            self.recognition_cache_frames += 1
            return self.last_recognition_result

        try:
            current_encoding = self.get_face_encoding(face_image)
            if current_encoding is None:
                return None

            max_similarity = 0
            best_match = None

            for user in self.known_users.values():
                similarity = self.calculate_cosine_similarity(
                    current_encoding, 
                    user.face_encoding
                )
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = user

            if max_similarity > self.SIMILARITY_THRESHOLD:
                # Cache the result
                self.last_recognition_result = best_match
                self.recognition_cache_frames = 0
                return best_match
            
            self.last_recognition_result = None
            return None

        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None

    def add_new_user(self, face_image: np.ndarray) -> Optional[UserProfile]:
        try:
            face_encoding = self.get_face_encoding(face_image)
            if face_encoding is None:
                return None

            analysis = self.analyse_face_snapshot(face_image)  # Changed analyze to analyse
            if analysis is None:
                return None

            user_id = str(uuid.uuid4())
            new_user = UserProfile(
                id=user_id,
                face_encoding=face_encoding,
                face_image=face_image,
                age=analysis['age'],
                gender=analysis['gender'],
                race=analysis['race'],
                timestamp=datetime.now()
            )
            
            # Save face snapshot in a separate thread to avoid blocking 
            snapshot_path = os.path.join(self.SNAPSHOTS_DIR, f"{user_id}.jpg")
            cv2.imwrite(snapshot_path, cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            
            self.known_users[user_id] = new_user
            self.save_known_users()
            
            return new_user
            
        except Exception as e:
            print(f"Error adding new user: {e}")
            return None

    def update_fps(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only the last 30 frame times for FPS calculation
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
            
        # Calculate FPS based on the last 30 frames
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        # Set optimal camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Starting webcam feed... Press 'q' to quit.")
        
        last_face_location = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.update_fps()

            # Display FPS
            cv2.putText(frame, f"FPS: {self.fps:.1f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Process every Nth frame for face detection
            if self.recognition_cache_frames % self.PROCESS_EVERY_N_FRAMES == 0:
                grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Changed gray to grey
                faces = self.face_cascade.detectMultiScale(
                    grey_frame,  # Changed gray to grey
                    scaleFactor=1.3,  # Increased for better performance
                    minNeighbors=5,
                    minSize=(60, 60)  # Increased minimum face size
                )
                
                if len(faces) > 0:
                    last_face_location = faces
                    self.last_detection_time = time.time()  # Update detection time when a face is detected
                else:
                    # If no faces detected in the current frame, check time
                    if self.last_detection_time is not None:
                        time_since_last_detection = time.time() - self.last_detection_time
                        if time_since_last_detection > self.NO_DETECTION_TIMEOUT:
                            last_face_location = None  # Clear face location if timeout exceeds
                            self.last_detection_time = None  # Reset detection time after clearing

            # If we have a face location (current or cached), process it
            if last_face_location is not None:
                for (x, y, w, h) in last_face_location:
                    face_image = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    
                    # Force new recognition periodically
                    force_new_recognition = self.recognition_cache_frames >= self.MAX_CACHE_FRAMES
                    user = self.recognise_face(face_image, force_new_recognition)  # Changed recognize to recognise
                    
                    if user:
                        colour = (0, 255, 0)  # Changed color to colour
                        display_text = [
                            f"ID: {user.id[:8]}",
                            f"Age: {user.age}",
                            f"Gender: {user.gender}",
                            f"Race: {user.race}",
                        ]
                    else:
                        colour = (0, 0, 255)  # Changed color to colour
                        new_user = self.add_new_user(face_image)
                        
                        if new_user:
                            colour = (255, 165, 0)  # Changed color to colour
                            display_text = [
                                f"New User!",
                                f"ID: {new_user.id[:8]}",
                                f"Age: {new_user.age}",
                                f"Gender: {new_user.gender}",
                                f"Race: {new_user.race}",
                            ]
                        else:
                            display_text = ["Failed to add user"]

                    cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)  # Changed color to colour
                    
                    y_offset = y - 10
                    for line in display_text:
                        cv2.putText(frame, line, (x, y_offset), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1,  # Changed color to colour
                                    cv2.LINE_AA)
                        y_offset -= 20

            # Clear overlay if no face detected for more than NO_DETECTION_TIMEOUT seconds
            if self.last_detection_time is not None:
                time_since_last_detection = time.time() - self.last_detection_time
                if time_since_last_detection > self.NO_DETECTION_TIMEOUT:
                    last_face_location = None  # Clear face location
                    self.last_detection_time = None  # Reset detection time

            # Always keep checking for new faces, even after overlay is cleared
            if self.recognition_cache_frames % self.PROCESS_EVERY_N_FRAMES == 0:
                grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Changed gray to grey
                faces = self.face_cascade.detectMultiScale(
                    grey_frame,  # Changed gray to grey
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(60, 60)
                )
                if len(faces) > 0:
                    last_face_location = faces
                    self.last_detection_time = time.time()  # Restart detection time if a face reappears

            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()