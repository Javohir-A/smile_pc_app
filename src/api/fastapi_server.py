# src/api/fastapi_server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import asyncio
import json
import cv2
import base64
import logging
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import uvicorn
from pathlib import Path
from src.config import AppConfig
import queue
import threading
import time
import websockets
import websockets.exceptions
from fastapi import WebSocket, WebSocketDisconnect, HTTPException

logger = logging.getLogger(__name__)

@dataclass
class CameraInfo:
    """Camera information for frontend"""
    stream_id: str
    name: str
    status: str
    fps: float
    resolution: tuple
    url: str

@dataclass
class StreamStats:
    """Streaming statistics"""
    active_connections: int
    total_streams: int
    active_streams: int
    websocket_fps: float

class FastAPIWebSocketServer:
    """FastAPI-based WebSocket server for camera streaming"""
    
    def __init__(self, config: AppConfig, host: str = "0.0.0.0", port: int = 8765):
        self.config = config
        self.host = host
        self.port = port
        self.app = FastAPI(
            title="Smile Mini PC Camera Streaming API",
            description="Real-time camera streaming with face detection and emotion recognition",
            version="1.0.0"
        )
        
        # Connection management
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # stream_id -> websockets
        self.all_streams_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}  # websocket -> metadata
        
        # Camera and streaming data
        self.camera_manager = None
        self.face_detector = None
        self.camera_info: Dict[str, CameraInfo] = {}
        self.stream_frames: Dict[str, bytes] = {}
        self.stream_stats = StreamStats(0, 0, 0, 0.0)
        
        self.frame_queue = queue.Queue(maxsize=50)  # Smaller queue
        self.frame_processor_task = None
        self.processing_frames = False
        # Server instance
        self.server = None
        self.server_task = None
        
        self._setup_routes()
        self._setup_static_files()
        
    def _setup_static_files(self):
        """Setup static file serving for frontend"""
        # Serve static files from frontend directory
        frontend_dir = Path("frontend")
        if frontend_dir.exists():
            self.app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve main frontend page"""
            frontend_file = Path("frontend/index.html")
            if frontend_file.exists():
                return FileResponse(frontend_file)
            else:
                return HTMLResponse("""
                <html>
                    <head><title>Smile Mini PC Camera Streams</title></head>
                    <body>
                        <h1>Smile Mini PC Camera System</h1>
                        <p>Frontend not found. Please create frontend/index.html</p>
                        <p>WebSocket endpoint: ws://localhost:8765/ws/</p>
                        <p>API documentation: <a href="/docs">/docs</a></p>
                    </body>
                </html>
                """)
        
        @self.app.get("/api/cameras", response_model=List[CameraInfo])
        async def get_cameras():
            """Get list of available cameras"""
            return list(self.camera_info.values())
        
        @self.app.get("/api/cameras/{stream_id}", response_model=CameraInfo)
        async def get_camera(stream_id: str):
            """Get specific camera information"""
            if stream_id not in self.camera_info:
                raise HTTPException(status_code=404, detail="Camera not found")
            return self.camera_info[stream_id]
        
        @self.app.get("/api/stats", response_model=StreamStats)
        async def get_stats():
            """Get streaming statistics"""
            self.stream_stats.active_connections = sum(len(conns) for conns in self.active_connections.values()) + len(self.all_streams_connections)
            self.stream_stats.total_streams = len(self.camera_info)
            self.stream_stats.active_streams = len([cam for cam in self.camera_info.values() if cam.status == "active"])
            return self.stream_stats
        
        @self.app.websocket("/ws/stream/{stream_id}")
        async def websocket_single_stream(websocket: WebSocket, stream_id: str):
            """WebSocket endpoint for single camera stream"""
            await self._handle_single_stream(websocket, stream_id)
        
        @self.app.websocket("/ws/streams/all")
        async def websocket_all_streams(websocket: WebSocket):
            """WebSocket endpoint for all camera streams"""
            await self._handle_all_streams(websocket)
        
        @self.app.websocket("/ws/")
        async def websocket_general(websocket: WebSocket):
            """General WebSocket endpoint with command support"""
            await self._handle_general_connection(websocket)
        
        @self.app.post("/api/cameras/{stream_id}/start")
        async def start_camera_stream(stream_id: str):
            """Start streaming for specific camera"""
            if self.camera_manager:
                # Add logic to start specific camera if needed
                return {"message": f"Stream {stream_id} start requested"}
            raise HTTPException(status_code=503, detail="Camera manager not available")
        
        @self.app.post("/api/cameras/{stream_id}/stop")
        async def stop_camera_stream(stream_id: str):
            """Stop streaming for specific camera"""
            if self.camera_manager:
                # Add logic to stop specific camera if needed
                return {"message": f"Stream {stream_id} stop requested"}
            raise HTTPException(status_code=503, detail="Camera manager not available")
    
    async def _handle_single_stream(self, websocket: WebSocket, stream_id: str):
        """Handle single stream WebSocket connection"""
        await websocket.accept()
        
        # Add to connections
        if stream_id not in self.active_connections:
            self.active_connections[stream_id] = set()
        self.active_connections[stream_id].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "type": "single_stream",
            "stream_id": stream_id,
            "connected_at": asyncio.get_event_loop().time()
        }
        
        logger.info(f"Client connected to stream {stream_id}")
        
        try:
            # Send initial camera info
            if stream_id in self.camera_info:
                await websocket.send_text(json.dumps({
                    "type": "camera_info",
                    "data": asdict(self.camera_info[stream_id])
                }))
            
            # Handle client messages
            while True:
                message = await websocket.receive_text()
                await self._handle_client_message(websocket, stream_id, message)
                
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from stream {stream_id}")
        except Exception as e:
            logger.error(f"Error in single stream WebSocket: {e}")
        finally:
            await self._cleanup_connection(websocket)
    
    async def _handle_all_streams(self, websocket: WebSocket):
        """Handle all streams WebSocket connection"""
        await websocket.accept()
        
        # Add to connections
        self.all_streams_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "type": "all_streams",
            "connected_at": asyncio.get_event_loop().time()
        }
        
        logger.info("Client connected to all streams")
        
        try:
            # Send initial camera list
            await websocket.send_text(json.dumps({
                "type": "camera_list",
                "data": [asdict(info) for info in self.camera_info.values()]
            }))
            
            # Handle client messages
            while True:
                message = await websocket.receive_text()
                await self._handle_client_message(websocket, "all", message)
                
        except WebSocketDisconnect:
            logger.info("Client disconnected from all streams")
        except Exception as e:
            logger.error(f"Error in all streams WebSocket: {e}")
        finally:
            await self._cleanup_connection(websocket)
    
    async def _handle_general_connection(self, websocket: WebSocket):
        """Handle general WebSocket connection with command support"""
        await websocket.accept()
        
        self.connection_metadata[websocket] = {
            "type": "general",
            "connected_at": asyncio.get_event_loop().time()
        }
        
        logger.info("General WebSocket client connected")
        
        try:
            # Send welcome message
            await websocket.send_text(json.dumps({
                "type": "welcome",
                "message": "Connected to Smile Mini PC Camera System",
                "available_commands": ["get_cameras", "subscribe_stream", "subscribe_all", "get_stats"]
            }))
            
            while True:
                message = await websocket.receive_text()
                await self._handle_general_message(websocket, message)
                
        except WebSocketDisconnect:
            logger.info("General WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in general WebSocket: {e}")
        finally:
            await self._cleanup_connection(websocket)
    
    async def _handle_client_message(self, websocket: WebSocket, stream_context: str, message: str):
        """Handle messages from WebSocket clients"""
        try:
            data = json.loads(message)
            command = data.get("command")
            
            if command == "get_cameras":
                await websocket.send_text(json.dumps({
                    "type": "camera_list",
                    "data": [asdict(info) for info in self.camera_info.values()]
                }))
            
            elif command == "get_frame":
                if stream_context in self.stream_frames:
                    await websocket.send_text(json.dumps({
                        "type": "frame",
                        "stream_id": stream_context,
                        "data": self.stream_frames[stream_context].decode('utf-8')
                    }))
            
            elif command == "get_stats":
                stats = await self.get_current_stats()
                await websocket.send_text(json.dumps({
                    "type": "stats",
                    "data": asdict(stats)
                }))
                
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Invalid JSON message"
            }))
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def _handle_general_message(self, websocket: WebSocket, message: str):
        """Handle general WebSocket messages with extended commands"""
        try:
            data = json.loads(message)
            command = data.get("command")
            
            if command == "subscribe_stream":
                stream_id = data.get("stream_id")
                if stream_id and stream_id in self.camera_info:
                    # Move connection to specific stream
                    if stream_id not in self.active_connections:
                        self.active_connections[stream_id] = set()
                    self.active_connections[stream_id].add(websocket)
                    self.connection_metadata[websocket]["subscribed_stream"] = stream_id
                    
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "stream_id": stream_id,
                        "camera_info": asdict(self.camera_info[stream_id])
                    }))
            
            elif command == "subscribe_all":
                self.all_streams_connections.add(websocket)
                self.connection_metadata[websocket]["subscribed_all"] = True
                
                await websocket.send_text(json.dumps({
                    "type": "subscribed_all",
                    "camera_list": [asdict(info) for info in self.camera_info.values()]
                }))
            
            else:
                # Handle as regular client message
                await self._handle_client_message(websocket, "general", message)
                
        except Exception as e:
            logger.error(f"Error handling general message: {e}")
    
    async def _cleanup_connection(self, websocket: WebSocket):
        """Clean up WebSocket connection"""
        try:
            # Remove from all connection sets
            for stream_connections in self.active_connections.values():
                stream_connections.discard(websocket)
            self.all_streams_connections.discard(websocket)
            
            # Remove metadata
            self.connection_metadata.pop(websocket, None)
            import websockets
            # Close the websocket if still open
            if websocket.client_state != websockets.exceptions.ConnectionState.DISCONNECTED:
                try:
                    await websocket.close()
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Error in cleanup: {e}")
    
    def set_camera_manager(self, camera_manager):
        """Set camera manager reference"""
        self.camera_manager = camera_manager
        
    def set_face_detector(self, face_detector):
        """Set face detector reference"""
        self.face_detector = face_detector
        
    def update_camera_info(self, stream_id: str, stream_info, name: str = None):
        """Update camera information"""
        self.camera_info[stream_id] = CameraInfo(
            stream_id=stream_id,
            name=name or f"Camera {stream_id}",
            status=stream_info.status.value,
            fps=stream_info.fps,
            resolution=stream_info.resolution,
            url=f"/ws/stream/{stream_id}"
        )
        
        try:
            camera_list_data = {
                'type': 'camera_list_update',
                'data': [asdict(info) for info in self.camera_info.values()]
            }
            self.frame_queue.put_nowait(camera_list_data)
        except queue.Full:
            pass  # Skip if queue is full
        
    def add_frame(self, stream_id: str, frame: np.ndarray, detection_result=None):
        """Add frame for streaming (async context)"""
        self.add_frame_threadsafe(stream_id, frame, detection_result)
    
    def _draw_face_annotations(self, frame: np.ndarray, faces) -> np.ndarray:
        """Draw face detection annotations"""
        for face in faces:
            x, y, w, h = int(face.x), int(face.y), int(face.width), int(face.height)
            
            # Get emotion color
            color = self._get_emotion_color(getattr(face, 'emotion', None))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
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
    
    def _get_emotion_color(self, emotion: Optional[str]):
        """Get BGR color for emotion"""
        emotion_colors = {
            'happy': (0, 255, 0), 'sad': (255, 0, 0), 'angry': (0, 0, 255),
            'surprise': (0, 255, 255), 'fear': (128, 0, 128), 'disgust': (0, 128, 0),
            'neutral': (128, 128, 128), 'unknown': (255, 255, 255)
        }
        return emotion_colors.get(emotion or 'unknown', (255, 255, 255))
    
    async def _broadcast_camera_list(self):
        """Broadcast updated camera list to all clients"""
        camera_list_msg = json.dumps({
            "type": "camera_list_update",
            "data": [asdict(info) for info in self.camera_info.values()]
        })
        
        # Send to all streams clients
        disconnected = []
        for websocket in self.all_streams_connections.copy():
            try:
                await websocket.send_text(camera_list_msg)
            except:
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.all_streams_connections.discard(ws)
    
    async def _broadcast_frame(self, stream_id: str, frame_data: str):
        """Broadcast frame to interested clients"""
        frame_msg = json.dumps({
            "type": "frame",
            "stream_id": stream_id,
            "data": frame_data,
            "timestamp": time.time()
        })
        
        disconnected = []
        sent_count = 0
        
        # Send to specific stream clients
        if stream_id in self.active_connections:
            for websocket in self.active_connections[stream_id].copy():
                # CHECK if websocket is still open
                if websocket.client_state == websockets.exceptions.ConnectionState.CONNECTED:
                    try:
                        await asyncio.wait_for(websocket.send_text(frame_msg), timeout=0.5)
                        sent_count += 1
                    except:
                        disconnected.append(websocket)
                else:
                    disconnected.append(websocket)
        
        # Same check for all streams clients
        for websocket in self.all_streams_connections.copy():
            if websocket.client_state == websockets.exceptions.ConnectionState.CONNECTED:
                try:
                    await asyncio.wait_for(websocket.send_text(frame_msg), timeout=0.5)
                    sent_count += 1
                except:
                    disconnected.append(websocket)
            else:
                disconnected.append(websocket)
        
        # Same check for subscribed clients
        for websocket, metadata in self.connection_metadata.items():
            if (metadata.get("subscribed_stream") == stream_id or 
                metadata.get("subscribed_all")):
                if websocket.client_state == websockets.exceptions.ConnectionState.CONNECTED:
                    try:
                        await asyncio.wait_for(websocket.send_text(frame_msg), timeout=0.5)
                        sent_count += 1
                    except:
                        disconnected.append(websocket)
                else:
                    disconnected.append(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            await self._cleanup_connection(ws)
        
        if sent_count > 0:
            logger.debug(f"Sent frame to {sent_count} clients")
        
    async def get_current_stats(self) -> StreamStats:
        """Get current streaming statistics"""
        active_connections = sum(len(conns) for conns in self.active_connections.values()) + len(self.all_streams_connections)
        return StreamStats(
            active_connections=active_connections,
            total_streams=len(self.camera_info),
            active_streams=len([cam for cam in self.camera_info.values() if cam.status == "active"]),
            websocket_fps=15.0  # You can calculate actual FPS if needed
        )
    
    async def start_server(self):
        """Start FastAPI server"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False
        )
        
        self.server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(self.server.serve())
        
        # ADD: Start frame processing
        self.processing_frames = True
        self.frame_processor_task = asyncio.create_task(self._process_frame_queue())
        
        logger.info(f"FastAPI WebSocket server started on http://{self.host}:{self.port}")
        logger.info(f"API documentation: http://{self.host}:{self.port}/docs")
        logger.info(f"WebSocket endpoints:")
        logger.info(f"  - Single stream: ws://{self.host}:{self.port}/ws/stream/{{stream_id}}")
        logger.info(f"  - All streams: ws://{self.host}:{self.port}/ws/streams/all")
        logger.info(f"  - General: ws://{self.host}:{self.port}/ws/")
            
    async def stop_server(self):
        """Stop FastAPI server"""
        # ADD: Stop frame processing
        self.processing_frames = False
        if self.frame_processor_task:
            self.frame_processor_task.cancel()
            try:
                await self.frame_processor_task
            except asyncio.CancelledError:
                pass
        
        if self.server:
            self.server.should_exit = True
            if self.server_task:
                await self.server_task
        logger.info("FastAPI WebSocket server stopped")
        
    def add_frame_threadsafe(self, stream_id: str, frame: np.ndarray, detection_result=None):
        """Thread-safe version - NO asyncio operations"""
        # Check if anyone is watching this stream
        has_clients = (
            (stream_id in self.active_connections and self.active_connections[stream_id]) or 
            self.all_streams_connections
        )
        
        if not has_clients:
            return
            
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame for stream {stream_id}")
                return
                
            # Process frame completely synchronously
            display_frame = frame.copy()
            
            # Validate frame dimensions
            if len(display_frame.shape) != 3 or display_frame.shape[2] != 3:
                logger.warning(f"Invalid frame format for stream {stream_id}: {display_frame.shape}")
                return
            
            # REDUCE FRAME SIZE for WebSocket streaming
            height, width = display_frame.shape[:2]
            if width > 640:  # Smaller size for better performance
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            # DON'T apply face detection again - frame already has annotations
            
            # Encode frame as JPEG with lower quality for WebSocket
            quality = getattr(self.config, 'WEBSOCKET_QUALITY', 100)  # Lower quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            
            success, buffer = cv2.imencode('.jpg', display_frame, encode_params)
            if not success:
                logger.warning(f"Failed to encode frame for stream {stream_id}")
                return
                
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            
            # Validate encoded frame
            if len(frame_bytes) == 0:
                logger.warning(f"Empty encoded frame for stream {stream_id}")
                return
            
            # CHECK FRAME SIZE
            frame_size_kb = len(frame_bytes) / 1024
            if frame_size_kb > 80:  # If larger than 80KB, reduce quality further
                quality = 30
                success, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if success:
                    frame_bytes = base64.b64encode(buffer).decode('utf-8')
                    logger.debug(f"Reduced frame quality to {quality} for stream {stream_id}")
            
            logger.debug(f"Frame size: {frame_size_kb:.1f}KB for stream {stream_id}")
            
            # Store frame - this is all we do, no broadcasting
            self.stream_frames[stream_id] = frame_bytes.encode('utf-8')
            
            # Put in queue for async processing
            frame_data = {
                'stream_id': stream_id,
                'frame_bytes': frame_bytes,
                'timestamp': time.time()
            }
            
            try:
                self.frame_queue.put_nowait(frame_data)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()  # Remove old frame
                    self.frame_queue.put_nowait(frame_data)  # Add new frame
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            
    async def _process_frame_queue(self):
        """Process frames from queue in async context"""
        logger.info("Frame queue processor started")
        
        while self.processing_frames:
            try:
                # Get frame from queue with timeout
                try:
                    frame_data = self.frame_queue.get(timeout=0.1)
                    logger.debug(f"Got frame from queue: {type(frame_data)}")
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue
                    
                # Check if it's a camera list update or frame data
                if isinstance(frame_data, dict) and frame_data.get('type') == 'camera_list_update':
                    # Handle camera list update
                    logger.debug("Processing camera list update")
                    await self._broadcast_camera_list_data(frame_data['data'])
                else:
                    # Handle frame data
                    stream_id = frame_data['stream_id']
                    frame_bytes = frame_data['frame_bytes']
                    
                    logger.debug(f"Broadcasting frame for stream {stream_id}")
                    # Broadcast to clients - this is the only async operation
                    await self._broadcast_frame(stream_id, frame_bytes)
                    
            except Exception as e:
                logger.error(f"Error in frame queue processor: {e}")
                await asyncio.sleep(0.01)
        
        logger.info("Frame queue processor stopped")

    async def _broadcast_camera_list_data(self, camera_data):
        """Broadcast camera list data to clients"""
        camera_list_msg = json.dumps({
            "type": "camera_list_update",
            "data": camera_data
        })
        
        # Send to all streams clients
        disconnected = []
        for websocket in self.all_streams_connections.copy():
            try:
                await websocket.send_text(camera_list_msg)
            except:
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            self.all_streams_connections.discard(ws)
    
    async def _handle_single_stream(self, websocket: WebSocket, stream_id: str):
        """Handle single stream WebSocket connection"""
        await websocket.accept()
        
        # Add to connections
        if stream_id not in self.active_connections:
            self.active_connections[stream_id] = set()
        self.active_connections[stream_id].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "type": "single_stream",
            "stream_id": stream_id,
            "connected_at": time.time()
        }
        
        logger.info(f"Client connected to stream {stream_id}")
        
        try:
            # Send initial camera info
            if stream_id in self.camera_info:
                await websocket.send_text(json.dumps({
                    "type": "camera_info",
                    "data": asdict(self.camera_info[stream_id])
                }))
            
            # ADD: Send current frame if available
            if stream_id in self.stream_frames:
                await websocket.send_text(json.dumps({
                    "type": "frame",
                    "stream_id": stream_id,
                    "data": self.stream_frames[stream_id].decode('utf-8'),
                    "timestamp": time.time()
                }))
            
            # Handle client messages
            while True:
                try:
                    # ADD timeout to prevent hanging
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    await self._handle_client_message(websocket, stream_id, message)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_text(json.dumps({
                        "type": "ping",
                        "timestamp": time.time()
                    }))
                    
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from stream {stream_id}")
        except Exception as e:
            logger.error(f"Error in single stream WebSocket: {e}")
        finally:
            await self._cleanup_connection(websocket)
            
    async def _broadcast_frame(self, stream_id: str, frame_data: str):
        """Broadcast frame to interested clients"""
        frame_msg = json.dumps({
            "type": "frame",
            "stream_id": stream_id,
            "data": frame_data,
            "timestamp": time.time()
        })
        
        disconnected = []
        sent_count = 0
        
        # Send to specific stream clients
        if stream_id in self.active_connections:
            for websocket in self.active_connections[stream_id].copy():
                try:
                    await asyncio.wait_for(websocket.send_text(frame_msg), timeout=0.5)
                    sent_count += 1
                except:
                    disconnected.append(websocket)
        
        # Send to all streams clients  
        for websocket in self.all_streams_connections.copy():
            try:
                await asyncio.wait_for(websocket.send_text(frame_msg), timeout=0.5)
                sent_count += 1
            except:
                disconnected.append(websocket)
        
        # Send to subscribed clients
        for websocket, metadata in self.connection_metadata.items():
            if (metadata.get("subscribed_stream") == stream_id or 
                metadata.get("subscribed_all")):
                try:
                    await asyncio.wait_for(websocket.send_text(frame_msg), timeout=0.5)
                    sent_count += 1
                except:
                    disconnected.append(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            await self._cleanup_connection(ws)
        
        if sent_count > 0:
            logger.debug(f"Sent frame to {sent_count} clients")

    # UPDATE the cleanup method:
    async def _cleanup_connection(self, websocket: WebSocket):
        """Clean up WebSocket connection"""
        try:
            # Remove from all connection sets
            for stream_connections in self.active_connections.values():
                stream_connections.discard(websocket)
            self.all_streams_connections.discard(websocket)
            
            # Remove metadata
            self.connection_metadata.pop(websocket, None)
            
        except Exception as e:
            logger.debug(f"Error in cleanup: {e}")
                
    def update_camera_info_once(self, stream_id: str, stream_info, name: str = None):
        """Update camera information once (no broadcasting)"""
        self.camera_info[stream_id] = CameraInfo(
            stream_id=stream_id,
            name=name or f"Camera {stream_id}",
            status=stream_info.status.value,
            fps=stream_info.fps,
            resolution=stream_info.resolution,
            url=f"/ws/stream/{stream_id}"
        )
        # Don't broadcast - just update the info