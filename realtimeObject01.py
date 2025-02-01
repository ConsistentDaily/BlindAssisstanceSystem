import cv2
import torch
import pyttsx3
import numpy as np
from ultralytics import YOLO
from collections import deque
from threading import Thread
import time
import queue
import pygame.mixer


class BlindAssistanceSystem:
    def __init__(self):
        # Initialize audio
        pygame.mixer.init()
        self.speech_queue = queue.Queue()
        self.last_speech_time = {}  # Dictionary to track last speech time for each object
        self.speech_cooldown = 2.0  # Cooldown period in seconds

        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 200)  # Faster speech rate

        # Initialize YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt').to(self.device)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Constants
        self.KNOWN_HEIGHT = {
            'person': 1.7,
            'car': 1.5,
            'bicycle': 1.0,
            'dog': 0.5,
            # Add more objects with their typical heights
        }
        self.FOCAL_LENGTH = 615
        self.confidence_threshold = 0.35  # Increased confidence threshold

        # Object tracking
        self.object_tracker = {}  # Dictionary to store object tracking info
        self.tracking_history = 5  # Number of frames to track

        # Priority objects that require immediate attention
        self.priority_objects = {'person', 'car', 'truck', 'motorcycle', 'bicycle', 'dog'}

        # Start speech thread
        self.speech_thread = Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

        # Initialize object tracking deques
        self.tracked_objects = {}

    def _speech_worker(self):
        """Background thread for handling speech queue"""
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                self.engine.say(text)
                self.engine.runAndWait()
                self.speech_queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text, object_id=None):
        """Add speech to queue with cooldown check"""
        current_time = time.time()
        if object_id is None or (
                object_id not in self.last_speech_time or
                current_time - self.last_speech_time[object_id] >= self.speech_cooldown
        ):
            self.speech_queue.put(text)
            if object_id is not None:
                self.last_speech_time[object_id] = current_time

    def estimate_distance(self, bbox_height, object_type):
        """Estimate distance using object-specific known heights"""
        known_height = self.KNOWN_HEIGHT.get(object_type, 1.0)
        distance = (known_height * self.FOCAL_LENGTH) / bbox_height
        return distance

    def smooth_detection(self, detection, object_id):
        """Smooth detection coordinates using moving average"""
        if object_id not in self.tracked_objects:
            self.tracked_objects[object_id] = deque(maxlen=5)

        self.tracked_objects[object_id].append(detection)
        if len(self.tracked_objects[object_id]) < 3:
            return detection

        return np.mean(self.tracked_objects[object_id], axis=0)

    def process_frame(self, frame):
        """Process a single frame and return detections"""
        # Resize frame while maintaining aspect ratio
        height, width = frame.shape[:2]
        scale = 640 / max(height, width)
        if scale < 1:
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # Run detection with batch processing
        results = self.model(frame, stream=True)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                label = self.model.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                # Calculate center and create detection object
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                bbox_height = y2 - y1

                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'label': label,
                    'confidence': confidence,
                    'center': (center_x, center_y),
                    'bbox_height': bbox_height
                }
                detections.append(detection)

        return detections

    def generate_guidance(self, detection, frame_width):
        """Generate detailed guidance message based on object position and distance"""
        label = detection['label']
        center_x = detection['center'][0]
        distance = self.estimate_distance(detection['bbox_height'], label)

        # Position detection
        if center_x < frame_width * 0.33:
            position = "on your left"
        elif center_x > frame_width * 0.66:
            position = "on your right"
        else:
            position = "directly ahead"

        # Distance-based message
        if distance < 1.5:
            urgency = "very close"
            action = "please stop or change direction immediately"
        elif distance < 3.0:
            urgency = "nearby"
            action = "proceed with caution"
        else:
            urgency = f"approximately {distance:.1f} meters away"
            action = "noted"

        return f"{label} {position}, {urgency}, {action}"

    def run(self):
        """Main loop for the assistance system"""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_width = frame.shape[1]

                # Process frame and get detections
                detections = self.process_frame(frame)

                # Sort detections by priority and distance
                detections.sort(key=lambda x: (
                    x['label'] not in self.priority_objects,
                    self.estimate_distance(x['bbox_height'], x['label'])
                ))

                # Process each detection
                for detection in detections:
                    # Generate unique object ID based on position and label
                    object_id = f"{detection['label']}_{int(detection['center'][0])}_{int(detection['center'][1])}"

                    # Smooth detection coordinates
                    smoothed_detection = self.smooth_detection(detection['bbox'], object_id)

                    # Generate and speak guidance
                    guidance = self.generate_guidance(detection, frame_width)
                    self.speak(guidance, object_id)

                    # Optional: Display detection (for debugging)
                    if frame is not None:
                        x1, y1, x2, y2 = detection['bbox']
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{detection['label']} {detection['confidence']:.2f}",
                                    (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Optional: Display frame (for debugging)
                cv2.imshow('YOLO Object Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    system = BlindAssistanceSystem()
    system.run()