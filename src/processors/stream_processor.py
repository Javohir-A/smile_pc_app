# src/processors/stream_processor.py - FIXED VERSION
import cv2
import threading
import queue
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from src.config import AppConfig
from src.api.fastapi_server import FastAPIWebSocketServer   
# logging.getLogger("src.api.fastapi_server").setLevel(logging.DEBUG)
# logging.getLogger("src.processors.stream_processor").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

class StreamStatus(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    ACTIVE = "active"
    ERROR = "error"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"

@dataclass
class StreamInfo:
    """Information about a stream"""
    stream_id: str
    url: str
    name: str
    status: StreamStatus
    last_frame_time: float
    error_count: int
    reconnect_count: int
    fps: float = 0.0
    resolution: tuple = (0, 0)

@dataclass
class FrameData:
    """Frame data with metadata"""
    stream_id: str
    frame: np.ndarray
    timestamp: float
    frame_number: int
    stream_info: StreamInfo
    processed_frame: Optional[np.ndarray] = None  # For display with annotations

@dataclass
class DisplayFrame:
    """Frame prepared for display"""
    stream_id: str
    frame: np.ndarray
    window_name: str
    timestamp: float

class StreamReader:
    """Individual stream reader - CAPTURE ONLY, NO GUI"""
    
    def __init__(self, stream_id: str, url: str, config: AppConfig):
        self.stream_id = stream_id
        self.url = url
        self.config = config
        self.cap = None
        self.running = False
        self.thread = None
        
        # Frame communication - no GUI operations here
        self.frame_queue = queue.Queue(maxsize=2)
        self.info = StreamInfo(
            stream_id=stream_id,
            url=url,
            name=f"Camera-{stream_id}",
            status=StreamStatus.IDLE,
            last_frame_time=0,
            error_count=0,
            reconnect_count=0
        )
        self._frame_count = 0
        self._lock = threading.Lock()
        
        # FPS calculation
        self._fps_window = []
        self._fps_window_size = 10
        self._color_format_warned = False
        self._force_color_conversion = True
        self.face_detector = None
        
    def set_face_detector(self, face_detector):
        """Set the face detection processor"""
        self.face_detector = face_detector
    
    def _read_frames(self):
        """Frame reading loop - NO GUI OPERATIONS"""
        consecutive_failures = 0
        last_frame_time = time.time()

        while self.running:
            try:
                if not self.cap or not self.cap.isOpened():
                    if not self._reconnect():
                        time.sleep(self.config.camera.reconnect_delay)
                        continue
                
                frame_start = time.time()
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame from stream {self.stream_id} (attempt {consecutive_failures})")
                    
                    if consecutive_failures >= self.config.camera.max_errors:
                        logger.error(f"Max consecutive failures reached for stream {self.stream_id}")
                        if not self._reconnect():
                            break
                        consecutive_failures = 0
                    continue
                
                # Handle color format conversion
                if self._force_color_conversion:
                    try:
                        frame = self._convert_frame_color(frame)
                    except Exception as e:
                        if not self._color_format_warned:
                            logger.warning(f"Frame color conversion failed for stream {self.stream_id}: {e}")
                            self._color_format_warned = True
                
                consecutive_failures = 0
                current_time = time.time()
                
                # Calculate FPS
                frame_interval = current_time - last_frame_time
                if frame_interval > 0:
                    instant_fps = 1.0 / frame_interval
                    self._fps_window.append(instant_fps)
                    if len(self._fps_window) > self._fps_window_size:
                        self._fps_window.pop(0)
                    
                    with self._lock:
                        self.info.last_frame_time = current_time
                        self.info.status = StreamStatus.ACTIVE
                        self._frame_count += 1
                        self.info.fps = sum(self._fps_window) / len(self._fps_window)
                
                last_frame_time = current_time
                
                # Create frame data - NO GUI OPERATIONS HERE
                frame_data = FrameData(
                    stream_id=self.stream_id,
                    frame=frame,
                    timestamp=current_time,
                    frame_number=self._frame_count,
                    stream_info=self.info
                )
                
                # Add to queue - drop old frames for real-time processing
                try:
                    while not self.frame_queue.empty():
                        try:
                            old_frame = self.frame_queue.get_nowait()
                            del old_frame
                        except queue.Empty:
                            break
                    
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    pass
                
                # Minimal processing delay
                processing_time = time.time() - frame_start
                if processing_time < 0.001:
                    time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in stream reader {self.stream_id}: {e}")
                consecutive_failures += 1
                if consecutive_failures >= self.config.camera.max_errors:
                    break
                time.sleep(self.config.camera.reconnect_delay)
        
        logger.info(f"Stream reader {self.stream_id} thread ended")
    
    def connect(self) -> bool:
        """Connect to the stream with optimized settings"""
        try:
            self.info.status = StreamStatus.CONNECTING
            logger.info(f"Connecting to stream {self.stream_id}: {self.url}")
            
            if not self.url or self.url == "0" or self.url == "":
                logger.warning(f"Invalid URL for stream {self.stream_id}: {self.url}")
                raise Exception("Invalid stream URL")
            
            # Try different backends
            backends_to_try = [
                (cv2.CAP_FFMPEG, {}),
                (cv2.CAP_GSTREAMER, {}),
                (cv2.CAP_ANY, {})
            ]
            
            self.cap = None
            for backend, options in backends_to_try:
                try:
                    self.cap = cv2.VideoCapture(self.url, backend)
                    if self.cap.isOpened():
                        logger.info(f"Successfully opened stream {self.stream_id} with backend {backend}")
                        break
                    else:
                        if self.cap:
                            self.cap.release()
                        self.cap = None
                except Exception as e:
                    logger.debug(f"Backend {backend} failed for stream {self.stream_id}: {e}")
                    if self.cap:
                        self.cap.release()
                    self.cap = None
                    continue
            
            if not self.cap:
                self.cap = cv2.VideoCapture(self.url)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open video capture with any backend")
            
            # Optimize capture properties
            self._optimize_capture_properties()
            
            # Test read a frame
            ret, test_frame = self.cap.read()
            if not ret:
                raise Exception("Failed to read test frame from stream")
            
            if test_frame is not None:
                logger.info(f"Stream {self.stream_id} frame shape: {test_frame.shape}, dtype: {test_frame.dtype}")
            
            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.info.resolution = (width, height)
            self.info.status = StreamStatus.ACTIVE
            logger.info(f"Stream {self.stream_id} connected successfully. Resolution: {width}x{height}, FPS: {actual_fps}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to stream {self.stream_id}: {e}")
            self.info.status = StreamStatus.ERROR
            self.info.error_count += 1
            return False
        
    def _optimize_capture_properties(self):
        """Apply capture optimizations safely"""
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass
        
        try:
            if self.url.startswith(('rtsp://', 'rtmp://')):
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            else:
                self.cap.set(cv2.CAP_PROP_FPS, 25)
        except:
            pass
        
        try:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            logger.debug(f"Enabled RGB conversion for stream {self.stream_id}")
        except:
            pass
        
        # Network stream optimizations
        if self.url.startswith(('rtsp://', 'rtmp://', 'http://')):
            timeout_props = [
                (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000),
                (cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
            ]
            
            for prop, value in timeout_props:
                try:
                    self.cap.set(prop, value)
                except:
                    pass
        
    def _convert_frame_color(self, frame):
        """Convert frame to proper color format if needed"""
        try:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                b, g, r = cv2.split(frame)
                if np.array_equal(b, g) and np.array_equal(g, r):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                return frame
            elif len(frame.shape) == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                return frame
        except Exception as e:
            if not self._color_format_warned:
                logger.warning(f"Color conversion error for stream {self.stream_id}: {e}")
                self._color_format_warned = True
            return frame
        
    def start(self):
        """Start reading frames"""
        if self.running:
            return
        
        if not self.connect():
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        logger.info(f"Stream reader {self.stream_id} started")
    
    def stop(self):
        """Stop reading frames"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.info.status = StreamStatus.STOPPED
        logger.info(f"Stream reader {self.stream_id} stopped")
    
    def get_frame(self) -> Optional[FrameData]:
        """Get the latest frame (non-blocking)"""
        try:
            latest_frame = None
            while True:
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            return latest_frame
        except:
            return None

    def _reconnect(self) -> bool:
        """Attempt to reconnect to the stream"""
        if self.info.reconnect_count >= self.config.camera.max_reconnects:
            logger.error(f"Max reconnection attempts reached for stream {self.stream_id}")
            self.info.status = StreamStatus.ERROR
            return False
        
        self.info.status = StreamStatus.RECONNECTING
        self.info.reconnect_count += 1
        
        logger.info(f"Attempting to reconnect stream {self.stream_id} (attempt {self.info.reconnect_count})")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        time.sleep(min(self.config.camera.reconnect_delay, 2.0))
        return self.connect()


class CentralizedDisplayManager:
    """Manages all camera displays in a single thread - THREAD SAFE"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.running = False
        self.display_thread = None
        self.display_queue = queue.Queue(maxsize=50)
        self.windows_created = set()
        self.face_detector = None
        # ADD: FastAPI WebSocket server
        self.fastapi_server = None
        if config.ENABLE_WEBSOCKET:
            self.fastapi_server = FastAPIWebSocketServer(
                config, 
                host="0.0.0.0",
                port=config.WEBSOCKET_PORT
            )
            logger.info("FastAPI WebSocket server initialized")
    
    async def start_fastapi_server(self):
        """Start FastAPI server"""
        if self.fastapi_server:
            await self.fastapi_server.start_server()

    async def stop_fastapi_server(self):
        """Stop FastAPI server"""
        if self.fastapi_server:
            await self.fastapi_server.stop_server()

    def set_camera_manager(self, camera_manager):
        """Set camera manager for FastAPI server"""
        if self.fastapi_server:
            self.fastapi_server.set_camera_manager(camera_manager)
          
    def set_face_detector(self, face_detector):
        """Set face detector for drawing annotations"""
        self.face_detector = face_detector
        
        if self.fastapi_server:
            self.fastapi_server.set_face_detector(face_detector)
            
    def add_frame_for_display(self, frame_data: FrameData):
        """Add frame to display queue (called from processing thread)"""
        
        # CV2 Display Logic - ONLY if GUI is enabled
        if self.config.enable_gui:
            try:
                # Prepare display frame with annotations
                display_frame = self._prepare_display_frame(frame_data)
                
                # Add to display queue, drop old frames if full
                try:
                    self.display_queue.put_nowait(display_frame)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put_nowait(display_frame)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Error preparing frame for display: {e}")
        
        # WebSocket Streaming Logic - COMPLETELY SEPARATE from CV2
        if self.fastapi_server:
            try:
                # Throttle WebSocket frames
                current_time = time.time()
                if not hasattr(self, '_last_websocket_frame_time'):
                    self._last_websocket_frame_time = {}
                
                stream_id = frame_data.stream_id
                last_time = self._last_websocket_frame_time.get(stream_id, 0)
                min_interval = 1.0 / getattr(self.config, 'WEBSOCKET_MAX_FPS', 15)
                
                if current_time - last_time < min_interval:
                    return
                
                self._last_websocket_frame_time[stream_id] = current_time
                
                # ALWAYS use WebSocket-specific frame preparation (no CV2 windows)
                annotated_frame = self._prepare_websocket_frame(frame_data)
                
                # Send frame to WebSocket clients (thread-safe)
                self.fastapi_server.add_frame_threadsafe(
                    frame_data.stream_id,
                    annotated_frame
                )
                
            except Exception as e:
                logger.error(f"Error adding frame to FastAPI stream: {e}")
            
    def _prepare_websocket_frame(self, frame_data: FrameData):
        """Prepare frame for WebSocket streaming WITHOUT creating CV2 windows"""
        frame = frame_data.frame.copy()
        
        # Apply face detection annotations if available
        if self.face_detector:
            try:
                detections = self.face_detector.get_stream_detections(frame_data.stream_id)
                if detections and detections.faces:
                    # Draw face annotations directly on frame
                    frame = self._draw_face_annotations_directly(frame, detections.faces)
            except Exception as e:
                logger.debug(f"Error drawing detections for WebSocket: {e}")
        
        # Resize frame for WebSocket (no CV2 display sizing)
        height, width = frame.shape[:2]
        if width > 640:  # WebSocket-specific sizing
            scale = 640 / width
            new_width = 640
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        # Add minimal stream info overlay for WebSocket
        info_text = f"Stream: {frame_data.stream_id} | FPS: {frame_data.stream_info.fps:.1f}"
        cv2.putText(frame, info_text, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame

    def _draw_face_annotations_directly(self, frame, faces):
        """Draw face annotations directly on frame (no display manager)"""
        for face in faces:
            x, y, w, h = int(face.x), int(face.y), int(face.width), int(face.height)
            
            # Get emotion color
            color = self._get_emotion_color_simple(getattr(face, 'emotion', None))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare label
            label_parts = []
            if hasattr(face, 'face_id') and face.face_id:
                label_parts.append(f"ID:{face.face_id[-3:]}")
            if hasattr(face, 'confidence'):
                label_parts.append(f"{face.confidence:.2f}")
            if hasattr(face, 'emotion') and face.emotion:
                label_parts.append(f"{face.emotion}")
                if hasattr(face, 'emotion_confidence') and face.emotion_confidence:
                    label_parts.append(f"({face.emotion_confidence:.2f})")
            
            label = " ".join(label_parts)
            
            # Draw label background and text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0] + 5, y), color, -1)
            cv2.putText(frame, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        return frame

    def _get_emotion_color_simple(self, emotion):
        """Get BGR color for emotion (simple version)"""
        emotion_colors = {
            'happy': (0, 255, 0), 'sad': (255, 0, 0), 'angry': (0, 0, 255),
            'surprise': (0, 255, 255), 'fear': (128, 0, 128), 'disgust': (0, 128, 0),
            'neutral': (128, 128, 128), 'unknown': (255, 255, 255)
        }
        return emotion_colors.get(emotion or 'unknown', (255, 255, 255))
        
        
    def _prepare_annotated_frame(self, frame_data: FrameData):
        """Prepare frame with annotations for FastAPI streaming"""
        frame = frame_data.frame.copy()
        
        # if self.face_detector:
        #     try:
        #         detections = self.face_detector.get_stream_detections(frame_data.stream_id)
        #         if detections and detections.faces:
        #             # Use same drawing logic as CV2 display
        #             frame = self._prepare_display_frame(frame_data).frame
        #     except Exception as e:
        #         logger.debug(f"Error drawing detections: {e}")
        
        return frame   
         
    def _prepare_display_frame(self, frame_data: FrameData) -> DisplayFrame:
        """Prepare frame for display with annotations"""
        frame = frame_data.frame.copy()
        
        # Apply face detection annotations if available
        if self.face_detector:
            try:
                detections = self.face_detector.get_stream_detections(frame_data.stream_id)
                if detections and detections.faces:
                    frame = self.face_detector.draw_detections(
                        frame, 
                        detections.faces,
                        show_emotions=True,
                        show_probabilities=False
                    )
            except Exception as e:
                logger.debug(f"Error drawing detections for {frame_data.stream_id}: {e}")
        
        # Resize frame to display dimensions
        frame = self._resize_frame_for_display(frame)
        
        # Add stream info overlay
        if hasattr(self.config, 'show_frame_info') and getattr(self.config.display, 'show_frame_info', True):
            info_text = f"Stream: {frame_data.stream_id} | FPS: {frame_data.stream_info.fps:.1f} | Frame: {frame_data.frame_number}"
            cv2.putText(frame, info_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        window_name = f"Camera {frame_data.stream_id}"
        
        return DisplayFrame(
            stream_id=frame_data.stream_id,
            frame=frame,
            window_name=window_name,
            timestamp=frame_data.timestamp
        )
    
    def _resize_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to match display configuration"""
        display_width = getattr(self.config.display, 'display_width', 960)
        display_height = getattr(self.config.display, 'display_height', 540)
        
        h, w = frame.shape[:2]
        scale_w = display_width / w
        scale_h = display_height / h
        scale = min(scale_w, scale_h)
        
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        y_offset = (display_height - new_height) // 2
        x_offset = (display_width - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        return canvas
    
    def _display_loop(self):
        """Main display loop - ONLY THREAD THAT CALLS OpenCV GUI"""
        logger.info("Centralized display thread started")
        
        while self.running:
            try:
                # Collect all available frames
                frames_to_display = []
                try:
                    # Get frames with timeout
                    frame = self.display_queue.get(timeout=0.1)
                    frames_to_display.append(frame)
                    
                    # Collect additional frames without blocking
                    while True:
                        try:
                            frame = self.display_queue.get_nowait()
                            frames_to_display.append(frame)
                        except queue.Empty:
                            break
                            
                except queue.Empty:
                    # No frames to display, just handle events
                    pass
                
                # Display all collected frames
                for display_frame in frames_to_display:
                    try:
                        # Create window if not exists
                        if display_frame.window_name not in self.windows_created:
                            cv2.namedWindow(display_frame.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                            display_width = getattr(self.config.display, 'display_width', 960)
                            display_height = getattr(self.config.display, 'display_height', 540)
                            cv2.resizeWindow(display_frame.window_name, display_width, display_height)
                            self.windows_created.add(display_frame.window_name)
                            logger.info(f"Created display window: {display_frame.window_name}")
                        
                        # Display frame
                        cv2.imshow(display_frame.window_name, display_frame.frame)
                        
                    except Exception as e:
                        logger.error(f"Error displaying frame for {display_frame.window_name}: {e}")
                
                # Handle keyboard events - CRITICAL: Only this thread calls waitKey
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit key pressed - stopping display")
                    self.running = False
                    break
                elif key == ord('c'):
                    logger.info("Clearing all windows")
                    cv2.destroyAllWindows()
                    self.windows_created.clear()
                
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
                time.sleep(0.01)
        
        # Cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        logger.info("Centralized display thread ended")
    
    def start(self):
        """Start the display manager"""
        # Convert string to boolean if needed
        gui_enabled = self.config.enable_gui
        if isinstance(gui_enabled, str):
            gui_enabled = gui_enabled.lower() in ('true', '1', 'yes', 'on')
        
        logger.critical(f"GUI enabled check: {gui_enabled} (original: {self.config.enable_gui}, type: {type(self.config.enable_gui)})")
        
        # ONLY start if GUI is enabled - USE THE CONVERTED VARIABLE
        if not gui_enabled:  # ✅ Fixed: use gui_enabled instead of self.config.enable_gui
            logger.info("🚫 GUI DISABLED - CV2 display manager will NOT start")
            logger.info(f"🔧 enable_gui setting: {self.config.enable_gui}")
            return
            
        if self.running:
            logger.info("⚠️  Display manager already running")
            return
            
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        logger.info("✅ CV2 display manager started (GUI enabled)")
    
    def stop(self):
        """Stop the display manager"""
        self.running = False
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=5)
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        logger.info("Centralized display manager stopped")