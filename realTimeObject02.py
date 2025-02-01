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
        self.last_speech_time = {}
        self.speech_cooldown = 1.5  # Reduced cooldown for more frequent updates

        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 200)

        # Initialize YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt').to(self.device)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Navigation constants
        self.SAFE_DISTANCE = 2.0  # meters
        self.DANGER_DISTANCE = 1.2  # meters
        self.LEFT_ZONE = 0.35
        self.RIGHT_ZONE = 0.65

        # Path tracking
        self.clear_path = True
        self.last_guidance = ""
        self.path_history = deque(maxlen=5)

        # Initialize other components
        self.KNOWN_HEIGHT = {
            'person': 1.7,
            'car': 1.5,
            'bicycle': 1.0,
            'dog': 0.5,
            'chair': 0.8,
            'table': 0.75,
            'bench': 0.5,
        }
        self.FOCAL_LENGTH = 615
        self.confidence_threshold = 0.35
        self.priority_objects = {'person', 'car', 'truck', 'motorcycle', 'bicycle', 'dog', 'chair', 'table', 'bench'}

        # Initialize zones for path analysis
        self.zones = {
            'left': {'clear': True, 'objects': []},
            'center': {'clear': True, 'objects': []},
            'right': {'clear': True, 'objects': []}
        }

        # Start speech thread
        self.speech_thread = Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()

        # Initialize object tracking
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

    def analyze_path(self, detections, frame_width):
        """Analyze the path ahead and determine the best route"""
        # Reset zones
        for zone in self.zones.values():
            zone['clear'] = True
            zone['objects'] = []

        # Analyze each detection and categorize by zone
        for detection in detections:
            center_x = detection['center'][0]
            distance = self.estimate_distance(detection['bbox_height'], detection['label'])

            # Determine zone
            if center_x < frame_width * self.LEFT_ZONE:
                zone = 'left'
            elif center_x > frame_width * self.RIGHT_ZONE:
                zone = 'right'
            else:
                zone = 'center'

            # Mark zone as occupied if object is within danger distance
            if distance < self.SAFE_DISTANCE:
                self.zones[zone]['clear'] = False
                self.zones[zone]['objects'].append({
                    'label': detection['label'],
                    'distance': distance
                })

    def generate_navigation_guidance(self):
        """Generate precise navigation instructions based on zone analysis"""
        # Check if center path is clear
        if self.zones['center']['clear']:
            if self.zones['left']['clear'] and self.zones['right']['clear']:
                return "Path is clear, keep walking straight"
            elif not self.zones['left']['clear']:
                return "Objects on your left, stay to the right while walking straight"
            elif not self.zones['right']['clear']:
                return "Objects on your right, stay to the left while walking straight"
        else:
            # Center path is blocked, find alternative
            center_objects = self.zones['center']['objects']
            if not center_objects:  # Safety check
                return "Caution, unable to determine path"

            nearest_object = min(center_objects, key=lambda x: x['distance'])

            if self.zones['left']['clear'] and self.zones['right']['clear']:
                return f"Stop. {nearest_object['label']} ahead at {nearest_object['distance']:.1f} meters. You can go either left or right"
            elif self.zones['left']['clear']:
                return f"Stop. {nearest_object['label']} ahead. Take a step to your left and proceed"
            elif self.zones['right']['clear']:
                return f"Stop. {nearest_object['label']} ahead. Take a step to your right and proceed"
            else:
                return f"Stop. Path blocked in all directions. Wait for path to clear"

    def generate_detailed_guidance(self, detection, frame_width):
        """Generate detailed guidance for specific objects"""
        label = detection['label']
        distance = self.estimate_distance(detection['bbox_height'], label)
        center_x = detection['center'][0]

        # Determine exact position
        if center_x < frame_width * self.LEFT_ZONE:
            position = "on your left"
        elif center_x > frame_width * self.RIGHT_ZONE:
            position = "on your right"
        else:
            position = "directly ahead"

        # Generate specific guidance based on object type and distance
        if distance < self.DANGER_DISTANCE:
            if label in ['person', 'dog']:
                return f"Caution! {label} very close {position}, stop immediately"
            elif label in ['car', 'truck', 'motorcycle']:
                return f"Danger! {label} very close {position}, stop immediately"
            else:
                return f"Warning! {label} very close {position}, stop immediately"
        elif distance < self.SAFE_DISTANCE:
            return f"{label} {position}, {distance:.1f} meters away, proceed with caution"
        else:
            return f"{label} {position}, {distance:.1f} meters away"

    def run(self):
        """Main loop with enhanced navigation guidance"""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_width = frame.shape[1]
                detections = self.process_frame(frame)

                # Analyze path and generate navigation guidance
                self.analyze_path(detections, frame_width)
                nav_guidance = self.generate_navigation_guidance()

                # Speak navigation guidance if it's different from last guidance
                if nav_guidance != self.last_guidance:
                    self.speak(nav_guidance)
                    self.last_guidance = nav_guidance

                # Process individual detections for specific warnings
                for detection in detections:
                    distance = self.estimate_distance(detection['bbox_height'], detection['label'])

                    # Only provide specific object warnings for close objects or priority objects
                    if distance < self.SAFE_DISTANCE or detection['label'] in self.priority_objects:
                        guidance = self.generate_detailed_guidance(detection, frame_width)
                        self.speak(guidance, f"{detection['label']}_{int(detection['center'][0])}")

                # Optional: Display frame for debugging
                if frame is not None:
                    # Draw zones for visualization
                    h, w = frame.shape[:2]
                    cv2.line(frame, (int(w * self.LEFT_ZONE), 0), (int(w * self.LEFT_ZONE), h), (0, 255, 0), 2)
                    cv2.line(frame, (int(w * self.RIGHT_ZONE), 0), (int(w * self.RIGHT_ZONE), h), (0, 255, 0), 2)

                    # Draw detections
                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{detection['label']}", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    cv2.imshow('YOLO Object Detection', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    system = BlindAssistanceSystem()
    system.run()