# src/utils/image_utils.py
import cv2
import time
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from src.config.settings import AppConfig
from src.processors.face_detection_processor import FaceDetection, DetectionResult

logger = logging.getLogger(__name__)

@dataclass
class DisplayMetrics:
    """Metrics for display performance"""
    fps: float
    frame_count: int
    detection_count: int
    processing_time: float

class ImageUtils:
    """Enhanced utility functions for image processing and display with face detection"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Display settings
        self.display_width = getattr(config.display, 'display_width', 1280)
        self.display_height = getattr(config.display, 'display_height', 720)
        self.show_fps = getattr(config.display, 'show_fps', True)
        self.show_detection_info = getattr(config.display, 'show_detection_info', True)
        self.show_emotion_probabilities = getattr(config.display, 'show_emotion_probabilities', False)
        
        # Performance tracking
        self._frame_times = []
        self._max_frame_history = 30
        
        # Window management
        self._windows: Dict[str, bool] = {}
        
        logger.info("Enhanced ImageUtils initialized")
    
    def display_frame_with_detections(self, frame: np.ndarray, detection_result: Optional[DetectionResult],
                                    start_time: float, window_name: str) -> DisplayMetrics:
        """Display frame with face detection annotations and performance metrics"""
        try:
            # Calculate frame processing time
            current_time = time.time()
            processing_time = current_time - start_time
            
            # Update frame time history for FPS calculation
            self._frame_times.append(current_time)
            if len(self._frame_times) > self._max_frame_history:
                self._frame_times.pop(0)
            
            # Calculate FPS
            fps = self._calculate_fps()
            
            # Resize frame for display
            display_frame, scale_x, scale_y = self._prepare_display_frame(frame)
            
            # Draw face detections if available
            detection_count = 0
            if detection_result and detection_result.faces:
                display_frame = self._draw_face_detections(
                    display_frame, detection_result.faces, scale_x, scale_y
                )
                detection_count = len(detection_result.faces)
            
            # Draw performance overlay
            display_frame = self._draw_performance_overlay(
                display_frame, fps, detection_count, processing_time, detection_result
            )
            
            # Show frame
            self._show_frame(display_frame, window_name)
            
            return DisplayMetrics(
                fps=fps,
                frame_count=len(self._frame_times),
                detection_count=detection_count,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Display failed for {window_name}: {e}")
            return DisplayMetrics(fps=0, frame_count=0, detection_count=0, processing_time=0)
    
    def _prepare_display_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Prepare frame for display with proper scaling"""
        frame_h, frame_w = frame.shape[:2]
        
        # Calculate display dimensions while maintaining aspect ratio
        aspect_ratio = frame_w / frame_h
        display_aspect = self.display_width / self.display_height
        
        if display_aspect > aspect_ratio:
            new_height = self.display_height
            new_width = int(self.display_height * aspect_ratio)
        else:
            new_width = self.display_width
            new_height = int(self.display_width / aspect_ratio)
        
        # Resize frame
        display_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Calculate scaling factors
        scale_x = new_width / frame_w
        scale_y = new_height / frame_h
        
        return display_frame, scale_x, scale_y
    
    def _draw_face_detections(self, frame: np.ndarray, faces: List[FaceDetection], 
                            scale_x: float, scale_y: float) -> np.ndarray:
        """Draw face detection annotations on frame"""
        for face in faces:
            # Scale coordinates
            x = int(face.x * scale_x)
            y = int(face.y * scale_y)
            w = int(face.width * scale_x)
            h = int(face.height * scale_y)
            
            # Get emotion color
            color = self._get_emotion_color(face.emotion)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label_parts = []
            if face.face_id:
                label_parts.append(f"ID:{face.face_id[-3:]}")  # Show last 3 chars of ID
            
            label_parts.append(f"{face.confidence:.2f}")
            
            if face.emotion:
                label_parts.append(f"{face.emotion}")
                if face.emotion_confidence:
                    label_parts.append(f"({face.emotion_confidence:.2f})")
            
            label = " ".join(label_parts)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0] + 5, y),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame, label, (x + 2, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
            
            # Draw emotion probabilities if enabled
            if self.show_emotion_probabilities and face.emotion_probabilities:
                self._draw_emotion_probabilities(frame, face.emotion_probabilities, x, y + h)
        
        return frame
    
    def _draw_emotion_probabilities(self, frame: np.ndarray, emotion_probs: Dict[str, float], 
                                  x: int, y: int):
        """Draw emotion probability bars"""
        sorted_emotions = sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True)
        
        bar_width = 100
        bar_height = 12
        spacing = 2
        
        for i, (emotion, prob) in enumerate(sorted_emotions[:4]):  # Top 4 emotions
            bar_y = y + i * (bar_height + spacing) + 5
            
            # Draw background bar
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Draw probability bar
            prob_width = int((prob / 100.0) * bar_width)
            color = self._get_emotion_color(emotion)
            cv2.rectangle(frame, (x, bar_y), (x + prob_width, bar_y + bar_height), color, -1)
            
            # Draw text
            text = f"{emotion}: {prob:.1f}%"
            cv2.putText(frame, text, (x + bar_width + 5, bar_y + bar_height - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_performance_overlay(self, frame: np.ndarray, fps: float, detection_count: int,
                                processing_time: float, detection_result: Optional[DetectionResult]) -> np.ndarray:
        """Draw performance metrics overlay"""
        if not self.show_fps and not self.show_detection_info:
            return frame
        
        overlay_height = 0
        line_height = 25
        
        # Prepare overlay text
        overlay_lines = []
        
        if self.show_fps:
            overlay_lines.append(f"FPS: {fps:.1f}")
            
        if self.show_detection_info:
            overlay_lines.append(f"Faces: {detection_count}")
            overlay_lines.append(f"Process: {processing_time*1000:.1f}ms")
            
            if detection_result:
                overlay_lines.append(f"Frame: {detection_result.frame_number}")
        
        # Calculate overlay dimensions
        overlay_height = len(overlay_lines) * line_height + 10
        max_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] 
                        for line in overlay_lines]) + 20
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + max_width, 10 + overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text lines
        for i, line in enumerate(overlay_lines):
            y_pos = 35 + i * line_height
            cv2.putText(frame, line, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def _show_frame(self, frame: np.ndarray, window_name: str):
        """Show frame in window with proper window management"""
        try:
            # Create window if it doesn't exist
            if window_name not in self._windows:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])
                self._windows[window_name] = True
                logger.debug(f"Created window: {window_name}")
            
            # Show frame
            cv2.imshow(window_name, frame)
            
        except Exception as e:
            logger.error(f"Error showing frame in window {window_name}: {e}")
    
    def _calculate_fps(self) -> float:
        """Calculate FPS based on frame time history"""
        if len(self._frame_times) < 2:
            return 0.0
        
        time_span = self._frame_times[-1] - self._frame_times[0]
        if time_span <= 0:
            return 0.0
        
        return (len(self._frame_times) - 1) / time_span
    
    def _get_emotion_color(self, emotion: Optional[str]) -> Tuple[int, int, int]:
        """Get BGR color for emotion visualization"""
        emotion_colors = {
            'happy': (0, 255, 0),       # Green
            'sad': (255, 0, 0),         # Blue
            'angry': (0, 0, 255),       # Red
            'surprise': (0, 255, 255),  # Yellow
            'fear': (128, 0, 128),      # Purple
            'disgust': (0, 128, 0),     # Dark Green
            'neutral': (128, 128, 128), # Gray
            'unknown': (255, 255, 255)  # White
        }
        return emotion_colors.get(emotion or 'unknown', (255, 255, 255))
    
    # def handle_key_events(self, window_name: str) -> str:
    #     """Handle keyboard events for the display window"""
    #     try:
    #         key = cv2.waitKey(1) & 0xFF
            
    #         if key == ord('q'):
    #             return 'quit'
    #         elif key == ord('s'):
    #             return