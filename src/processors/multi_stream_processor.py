# src/processors/multi_stream_processor.py - UPDATED WITH MINI PC INTEGRATION
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any, Set
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from src.config import AppConfig
from src.di.dependencies import DependencyContainer
from .stream_processor import StreamReader, StreamInfo, StreamStatus, CentralizedDisplayManager
from src.helpers.system import get_system_mac_address
from src.models.stream import FrameData

logger = logging.getLogger(__name__)

class MultiStreamProcessor:
    """Processes multiple camera streams with centralized display management and Mini PC integration"""
    
    def __init__(self, config: AppConfig, dependency_container: DependencyContainer):
        self.config = config
        self.container = dependency_container
        self.stream_readers: Dict[str, StreamReader] = {}
        self.running = False
        self.processing_thread = None
        self.frame_processors: List[Callable[[FrameData], Any]] = []
        self._stats_thread = None
        self.face_detector = None
        self._emotion_processor = None
        
        # NEW: Centralized display manager
        self.display_manager = CentralizedDisplayManager(config)
        self.display_manager.set_camera_manager(self)
        # Camera discovery
        self._discovery_thread = None
        self._last_discovery_time = None
        self._discovery_interval = 60
        self._discovery_lock = threading.Lock()
        self._known_camera_urls = set()
        
        # Track exact camera-to-stream mappings
        self._assigned_camera_mapping: Dict[str, str] = {}  # camera_guid -> stream_id
        self._stream_camera_mapping: Dict[str, str] = {}   # stream_id -> camera_guid
        self._last_assigned_cameras: Set[str] = set()      # Previous assignment for comparison
        
        # Mini PC integration
        self._mini_pc_info = None
        self._system_mac_address = None
        
        logger.info(f"MultiStreamProcessor initialized with centralized display and Mini PC integration")
        
        # Initialize Mini PC info
        self._initialize_mini_pc_info()
    
    def _initialize_mini_pc_info(self):
        """Initialize Mini PC information for camera filtering"""
        try:
            self._system_mac_address = get_system_mac_address()
            logger.info(f"System MAC address: {self._system_mac_address}")
            
            # Get Mini PC info
            mini_pc_usecase = self.container.get_mini_pc_usecase()
            self._mini_pc_info = mini_pc_usecase.get_mini_pc_by_mac(self._system_mac_address)
            
            if self._mini_pc_info:
                logger.info(f"Mini PC found: {self._mini_pc_info.device_name} (ID: {self._mini_pc_info.guid})")
            else:
                logger.warning(f"Mini PC not found for MAC: {self._system_mac_address}")
                
        except Exception as e:
            logger.error(f"Error initializing Mini PC info: {e}")
            self._mini_pc_info = None
    
    def add_frame_processor(self, processor: Callable[[FrameData], Any]):
        """Add a frame processor function"""
        self.frame_processors.append(processor)
        logger.info(f"Added frame processor: {processor.__name__}")
    
    def set_face_detector(self, face_detector):
        """Set the face detection processor for all streams"""
        self.face_detector = face_detector
        # Set face detector for display manager
        self.display_manager.set_face_detector(face_detector)
        # Set face detector for all existing stream readers
        for reader in self.stream_readers.values():
            reader.set_face_detector(face_detector)
    
    def set_discovery_interval(self, hours: float):
        """Set camera discovery interval in hours"""
        self._discovery_interval = hours * 3600
        logger.info(f"Camera discovery interval set to {hours} hours")
    
    def initialize_streams(self, force_refresh: bool = False):
        """Initialize all stream readers with optional force refresh"""
        with self._discovery_lock:
            if force_refresh:
                self._smart_stream_refresh()
            else:
                self.stop_all_streams()
                self._initialize_all_cameras()
            
            self._last_discovery_time = datetime.now()
    
    def _get_assigned_cameras(self) -> List:
        """Get cameras assigned to this Mini PC"""
        try:
            if not self._mini_pc_info:
                logger.warning("No Mini PC info available - cannot get assigned cameras")
                return []
            
            mini_pc_usecase = self.container.get_mini_pc_usecase()
            cameras = mini_pc_usecase.get_mini_pc_cameras(mini_pc_id=self._mini_pc_info.guid)
            
            logger.info(f"Found {len(cameras)} cameras assigned to Mini PC {self._mini_pc_info.device_name}")
            return cameras
            
        except Exception as e:
            logger.error(f"Failed to get assigned cameras: {e}")
            return []
    
    def _get_all_cameras_fallback(self) -> List:
        """Fallback: Get all cameras from database if Mini PC assignment fails"""
        try:
            camera_usecase = self.container.get_camera_usecase()
            cameras = camera_usecase.list_cameras()
            logger.info(f"Fallback: Found {len(cameras)} total cameras from database")
            return cameras
        except Exception as e:
            logger.error(f"Failed to get cameras from database: {e}")
            return []
    
    def _smart_stream_refresh(self):
        """Smart refresh - ONLY use assigned cameras"""
        logger.info("Performing smart camera refresh...")
        
        # ONLY get cameras assigned to this Mini PC
        cameras = self._get_assigned_cameras()
        
        # REMOVED: No fallback to all cameras
        if not cameras:
            logger.info("No assigned cameras found - waiting for camera assignment")
            logger.info("💡 Admin can assign cameras to this Mini PC in the admin panel")
            # Don't load any cameras - just wait
            
            # Clear existing streams
            streams_to_remove = list(self.stream_readers.keys())
            for stream_id in streams_to_remove:
                if not stream_id.startswith("fallback") and not stream_id.startswith("dev"):
                    logger.info(f"Removing stream {stream_id} - no assigned cameras")
                    reader = self.stream_readers[stream_id]
                    reader.stop()
                    del self.stream_readers[stream_id]
                    self._known_camera_urls.discard(reader.url)
            
            # Add development cameras only if no database cameras at all
            dev_cameras = self._get_development_cameras()
            if dev_cameras:
                logger.info("Adding development cameras for testing")
                current_urls = set()
                camera_map = {}
                for url, camera_info in dev_cameras.items():
                    current_urls.add(url)
                    camera_map[url] = camera_info
            else:
                current_urls = set()
                camera_map = {}
        else:
            # Process assigned cameras
            current_urls = set()
            camera_map = {}
            
            for camera in cameras:
                try:
                    url = camera.generate_rtsp_url()
                    if url and url.strip():
                        current_urls.add(url)
                        camera_map[url] = camera
                except Exception as e:
                    logger.error(f"Error processing camera {getattr(camera, 'guid', 'unknown')}: {e}")
        
        # Find cameras to remove
        streams_to_remove = []
        for stream_id, reader in self.stream_readers.items():
            if reader.url not in current_urls and not stream_id.startswith("fallback"):
                streams_to_remove.append(stream_id)
        
        # Remove old streams
        for stream_id in streams_to_remove:
            logger.info(f"Removing stream {stream_id} - camera no longer assigned")
            reader = self.stream_readers[stream_id]
            reader.stop()
            del self.stream_readers[stream_id]
            self._known_camera_urls.discard(reader.url)
        
        # Find new cameras to add
        new_cameras = []
        for url, camera in camera_map.items():
            if url not in self._known_camera_urls:
                new_cameras.append((url, camera))
        
        # Add new streams
        for url, camera in new_cameras:
            try:
                stream_id = self._get_next_stream_id()
                camera_name = getattr(camera, 'guid', getattr(camera, 'name', 'unknown'))
                logger.info(f"Adding new assigned camera stream {stream_id}: {camera_name}")
                
                reader = StreamReader(stream_id, url, self.config)
                
                if self.face_detector:
                    reader.set_face_detector(self.face_detector)
                
                self.stream_readers[stream_id] = reader
                self._known_camera_urls.add(url)
                
                if self.running:
                    reader.start()
                    logger.info(f"Started new stream {stream_id} for assigned camera: {camera_name}")
                    
            except Exception as e:
                logger.error(f"Failed to add new camera {getattr(camera, 'guid', 'unknown')}: {e}")
        
        self._update_fallback_stream(current_urls)
        logger.info(f"Smart refresh complete - Added: {len(new_cameras)}, Removed: {len(streams_to_remove)}, Active: {len(self.stream_readers)}")
   
    def _remove_unassigned_cameras(self, current_assigned_guids: Set[str]):
        """Remove cameras that are no longer assigned to this Mini PC"""
        streams_to_remove = []
        
        for stream_id, camera_guid in self._stream_camera_mapping.items():
            # If this camera is no longer in the assigned list, remove it
            if camera_guid not in current_assigned_guids:
                # Skip development and fallback streams
                if not stream_id.startswith(("dev_", "fallback_")):
                    streams_to_remove.append((stream_id, camera_guid))
        
        for stream_id, camera_guid in streams_to_remove:
            logger.info(f"Removing camera {camera_guid} (stream {stream_id}) - no longer assigned to this Mini PC")
            self._remove_stream(stream_id, camera_guid)
    
    def _log_current_mappings(self):
        """Log current camera to stream mappings for debugging"""
        logger.debug("=== Current Camera Mappings ===")
        logger.debug(f"Assigned cameras: {len(self._assigned_camera_mapping)}")
        for camera_guid, stream_id in self._assigned_camera_mapping.items():
            logger.debug(f"  Camera {camera_guid} -> Stream {stream_id}")
        
        logger.debug(f"All streams: {len(self.stream_readers)}")
        for stream_id in self.stream_readers.keys():
            camera_guid = self._stream_camera_mapping.get(stream_id, "unknown")
            logger.debug(f"  Stream {stream_id} -> Camera {camera_guid}")
    
    def _remove_stream(self, stream_id: str, camera_guid: str):
        """Remove a specific stream and update tracking"""
        try:
            if stream_id in self.stream_readers:
                reader = self.stream_readers[stream_id]
                reader.stop()
                del self.stream_readers[stream_id]
                self._known_camera_urls.discard(reader.url)
                
                # Update tracking mappings
                if camera_guid in self._assigned_camera_mapping:
                    del self._assigned_camera_mapping[camera_guid]
                if stream_id in self._stream_camera_mapping:
                    del self._stream_camera_mapping[stream_id]
                
                logger.info(f"Successfully removed stream {stream_id} for camera {camera_guid}")
                logger.debug(f"Removed mapping: Camera {camera_guid} -> Stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove stream {stream_id}: {e}")
    
    
    def _get_development_cameras(self) -> Dict[str, Any]:
        """Get hardcoded development camera URLs"""
        dev_cameras = {}
        
        # Development cameras (keep your existing ones)
        urls = [
            "rtsp://localhost:8554/mystream",
            # "rtsp://10.10.0.37:8554/mystream"
        ]
        
        for i, url in enumerate(urls):
            dev_cameras[url] = {
                'guid': f'dev-camera-{i}',
                'name': f'Development Camera {i+1}',
                'ip_address': url.split('@')[-1].split(':')[0] if '@' in url else url.split('://')[1].split(':')[0],
                'port': 554
            }
        
        return dev_cameras
    
    def _get_next_stream_id(self) -> str:
        """Get the next available stream ID"""
        existing_ids = set()
        for stream_id in self.stream_readers.keys():
            if stream_id.startswith("stream_"):
                try:
                    num = int(stream_id.split("_")[1])
                    existing_ids.add(num)
                except:
                    pass
        
        for i in range(100):
            if i not in existing_ids:
                return f"stream_{i:02d}"
        
        return f"stream_{len(self.stream_readers):02d}"
    
    def _initialize_all_cameras(self):
        """Initialize ONLY assigned cameras with tracking"""
        logger.info("Initializing assigned cameras with precise tracking...")
        
        # Clear previous mappings
        self._assigned_camera_mapping.clear()
        self._stream_camera_mapping.clear()
        self._known_camera_urls.clear()
        
        # Get cameras assigned to this Mini PC
        cameras = self._get_assigned_cameras()
        
        if cameras:
            logger.info(f"Initializing {len(cameras)} assigned cameras for Mini PC {self._mini_pc_info.device_name}")
            
            for camera in cameras:
                try:
                    camera_guid = str(camera.guid)
                    url = camera.generate_rtsp_url()
                    
                    if url and url.strip():
                        self._add_assigned_camera_stream(camera, camera_guid, url)
                    else:
                        logger.warning(f"Skipping camera {camera_guid} - invalid URL")
                        
                except Exception as e:
                    logger.error(f"Failed to initialize camera {getattr(camera, 'guid', 'unknown')}: {e}")
        else:
            logger.info("No assigned cameras found")
            logger.info("💡 Assign cameras to this Mini PC in the admin panel to start processing")
            
            # Add development cameras for testing
            dev_cameras = self._get_development_cameras()
            if dev_cameras:
                logger.info("Adding development cameras for testing")
                self._add_development_streams(dev_cameras)
        
        self._update_fallback_stream(self._known_camera_urls)
        self._log_current_mappings()
        logger.info(f"Initialized {len(self.stream_readers)} stream readers total")
    
    def _add_development_streams(self, dev_cameras: Dict[str, Any]):
        """Add development camera streams"""
        for i, (url, camera_info) in enumerate(dev_cameras.items()):
            try:
                stream_id = f"dev_{i:02d}"
                
                # Remove existing dev stream if present
                if stream_id in self.stream_readers:
                    self._remove_stream(stream_id, f"dev-camera-{i}")
                
                reader = StreamReader(stream_id, url, self.config)
                
                if self.face_detector:
                    reader.set_face_detector(self.face_detector)
                
                self.stream_readers[stream_id] = reader
                self._known_camera_urls.add(url)
                
                # Track development cameras separately
                dev_camera_id = f"dev-camera-{i}"
                self._stream_camera_mapping[stream_id] = dev_camera_id
                
                if self.running:
                    reader.start()
                    logger.info(f"Started development stream {stream_id}")
                    
            except Exception as e:
                logger.error(f"Failed to add development camera {i}: {e}")
    
    def _add_assigned_camera_stream(self, camera, camera_guid: str, url: str):
        """Add a new assigned camera stream with tracking"""
        try:
            stream_id = self._get_next_stream_id()
            camera_name = getattr(camera, 'guid', getattr(camera, 'name', 'unknown'))
            
            logger.info(f"Adding assigned camera {camera_guid} as stream {stream_id}: {camera_name}")
            
            reader = StreamReader(stream_id, url, self.config)
            
            if self.face_detector:
                reader.set_face_detector(self.face_detector)
            
            # Add to streams
            self.stream_readers[stream_id] = reader
            self._known_camera_urls.add(url)
            
            # Update tracking mappings
            self._assigned_camera_mapping[camera_guid] = stream_id
            self._stream_camera_mapping[stream_id] = camera_guid
            
            if self.running:
                reader.start()
                logger.info(f"Started stream {stream_id} for assigned camera {camera_guid}")
            
            logger.debug(f"Added mapping: Camera {camera_guid} -> Stream {stream_id}")
            
        except Exception as e:
            logger.error(f"Failed to add assigned camera {camera_guid}: {e}")
    
    def _update_fallback_stream(self, current_urls: set):
        """Update fallback stream if configured"""
        if (hasattr(self.config.camera, 'fallback_url') and 
            self.config.camera.fallback_url and 
            self.config.camera.fallback_url.strip() and
            self.config.camera.fallback_url not in ["0", ""]):
            
            if self.config.camera.fallback_url not in current_urls:
                fallback_id = "fallback_stream"
                
                if fallback_id not in self.stream_readers:
                    try:
                        fallback_reader = StreamReader(fallback_id, self.config.camera.fallback_url, self.config)
                        if self.face_detector:
                            fallback_reader.set_face_detector(self.face_detector)
                        self.stream_readers[fallback_id] = fallback_reader
                        
                        if self.running:
                            fallback_reader.start()
                        
                        logger.info(f"Added fallback stream: {fallback_id}")
                    except Exception as e:
                        logger.error(f"Failed to initialize fallback stream: {e}")
    
    def _camera_discovery_loop(self):
        """Background thread for periodic camera discovery"""
        logger.info("Camera discovery thread started")
        
        while self.running:
            try:
                if self._last_discovery_time:
                    next_discovery = self._last_discovery_time + timedelta(seconds=self._discovery_interval)
                    sleep_time = (next_discovery - datetime.now()).total_seconds()
                else:
                    sleep_time = 0
                
                if sleep_time <= 0:
                    logger.info("Performing scheduled camera discovery...")
                    
                    # Re-check Mini PC assignment in case cameras were added
                    if self._mini_pc_info:
                        assigned_cameras = self._get_assigned_cameras()
                        if assigned_cameras:
                            logger.info(f"Found {len(assigned_cameras)} newly assigned cameras")
                        else:
                            logger.info("No assigned cameras found - continuing with discovery mode")
                    
                    self.initialize_streams(force_refresh=True)
                    sleep_time = self._discovery_interval
                else:
                    hours_remaining = sleep_time / 3600
                    logger.debug(f"Next camera discovery in {hours_remaining:.1f} hours")
                
                sleep_intervals = min(60, sleep_time)
                for _ in range(int(sleep_time / sleep_intervals)):
                    if not self.running:
                        break
                    time.sleep(sleep_intervals)
                
            except Exception as e:
                logger.error(f"Error in camera discovery loop: {e}")
                time.sleep(60)
        
        logger.info("Camera discovery thread ended")
        
    def get_camera_assignment_status(self) -> Dict[str, Any]:
        """Get detailed camera assignment status"""
        try:
            assigned_cameras = self._get_assigned_cameras() if self._mini_pc_info else []
            
            assignment_details = {
                'mini_pc_registered': self._mini_pc_info is not None,
                'mini_pc_id': str(self._mini_pc_info.guid) if self._mini_pc_info else None,
                'mini_pc_name': self._mini_pc_info.device_name if self._mini_pc_info else None,
                'system_mac': self._system_mac_address,
                'assigned_cameras_count': len(assigned_cameras),
                'active_streams_count': len(self.stream_readers),
                'camera_mappings': {},
                'stream_mappings': {},
                'cameras': []
            }
            
            # Camera to stream mappings
            for camera_guid, stream_id in self._assigned_camera_mapping.items():
                assignment_details['camera_mappings'][camera_guid] = stream_id
            
            # Stream to camera mappings
            for stream_id, camera_guid in self._stream_camera_mapping.items():
                assignment_details['stream_mappings'][stream_id] = camera_guid
            
            # Detailed camera info
            for camera in assigned_cameras:
                try:
                    camera_guid = str(camera.guid)
                    url = camera.generate_rtsp_url()
                    stream_id = self._assigned_camera_mapping.get(camera_guid, "not_mapped")
                    
                    # Get stream status
                    stream_status = "not_started"
                    if stream_id != "not_mapped" and stream_id in self.stream_readers:
                        reader = self.stream_readers[stream_id]
                        stream_status = reader.info.status.value
                    
                    assignment_details['cameras'].append({
                        'guid': camera_guid,
                        'stream_id': stream_id,
                        'ip_address': camera.ip_address,
                        'port': camera.port,
                        'rtsp_url': url,
                        'stream_status': stream_status,
                        'is_mapped': stream_id != "not_mapped",
                        'capabilities': {
                            'emotion': camera.detect_emotion,
                            'hands': camera.detect_hands,
                            'voice': camera.voice_detect,
                            'mask': camera.detect_mask,
                            'uniform': camera.detect_uniform
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing camera {camera.guid}: {e}")
            
            return assignment_details
            
        except Exception as e:
            logger.error(f"Error getting camera assignment status: {e}")
            return {'error': str(e)}
    
    # Keep all your existing methods but update the discovery status method
    def get_discovery_status(self) -> Dict[str, Any]:
        """Get camera discovery status with Mini PC info and precise tracking"""
        next_discovery = None
        if self._last_discovery_time:
            next_discovery = self._last_discovery_time + timedelta(seconds=self._discovery_interval)
        
        # Get current camera status
        assigned_cameras = self._get_assigned_cameras() if self._mini_pc_info else []
        
        return {
            'last_discovery': self._last_discovery_time.isoformat() if self._last_discovery_time else None,
            'next_discovery': next_discovery.isoformat() if next_discovery else None,
            'interval_hours': self._discovery_interval / 3600,
            'known_cameras': len(self._known_camera_urls),
            'active_streams': len(self.get_active_streams()),
            'total_streams': len(self.stream_readers),
            'mini_pc_id': str(self._mini_pc_info.guid) if self._mini_pc_info else None,
            'mini_pc_name': self._mini_pc_info.device_name if self._mini_pc_info else None,
            'assigned_cameras_count': len(assigned_cameras),
            'mapped_cameras_count': len(self._assigned_camera_mapping),
            'discovery_mode': not bool(assigned_cameras),
            'camera_mappings': dict(self._assigned_camera_mapping),
            'last_assigned_cameras': list(self._last_assigned_cameras)
        }

    def start_all_streams(self):
        """Start all stream readers, display manager, and processing threads"""
        logger.info("Starting all stream readers...")
        
        # Log Mini PC status
        if self._mini_pc_info:
            logger.info(f"Starting streams for Mini PC: {self._mini_pc_info.device_name}")
            assigned_cameras = self._get_assigned_cameras()
            logger.info(f"Assigned cameras: {len(assigned_cameras)}")
        else:
            logger.info("Starting streams in discovery mode - Mini PC not registered")
        
        # Start display manager first
        self.display_manager.start()
        
        if hasattr(self, 'display_manager') and self.display_manager.fastapi_server:
            for stream_id, reader in self.stream_readers.items():
                try:
                    self.display_manager.fastapi_server.update_camera_info_once(
                        stream_id,
                        reader.info,
                        f"Camera {stream_id}"
                    )
                except Exception as e:
                    logger.error(f"Error updating camera info for {stream_id}: {e}")
        # Start all stream readers
        for stream_id, reader in self.stream_readers.items():
            try:
                reader.start()
                logger.info(f"Started stream reader: {stream_id}")
            except Exception as e:
                logger.error(f"Failed to start stream reader {stream_id}: {e}")
        
        time.sleep(1)
        
        # Start processing threads
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
        self._stats_thread = threading.Thread(target=self._monitor_stats, daemon=True)
        self._stats_thread.start()
        
        self._discovery_thread = threading.Thread(target=self._camera_discovery_loop, daemon=True, name="camera_discovery")
        self._discovery_thread.start()
        
        logger.info("All streams started with centralized display and Mini PC integration")
    
    def stop_all_streams(self):
        """Stop all stream readers, display manager, and threads"""
        logger.info("Stopping all stream readers...")
        
        self.running = False
        
        # Stop all threads
        for task, name in [
            (self.processing_thread, "processing"),
            (self._stats_thread, "stats"),
            (self._discovery_thread, "discovery")
        ]:
            if task and task.is_alive():
                logger.info(f"Waiting for {name} thread to stop...")
                task.join(timeout=5)
        
        # Stop all stream readers
        for stream_id, reader in self.stream_readers.items():
            try:
                reader.stop()
                logger.info(f"Stopped stream reader: {stream_id}")
            except Exception as e:
                logger.error(f"Error stopping stream reader {stream_id}: {e}")
        
        # Stop display manager
        self.display_manager.stop()
        
        self.stream_readers.clear()
        self._known_camera_urls.clear()
        logger.info("All streams stopped")
    
    def get_stream_info(self) -> Dict[str, StreamInfo]:
        """Get information about all streams"""
        return {stream_id: reader.info for stream_id, reader in self.stream_readers.items()}
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs"""
        return [
            stream_id for stream_id, reader in self.stream_readers.items()
            if reader.info.status == StreamStatus.ACTIVE
        ]
    
    def force_camera_discovery(self):
        """Force immediate camera discovery"""
        logger.info("Forcing camera discovery...")
        self.initialize_streams(force_refresh=True)
    
    def get_discovery_status(self) -> Dict[str, Any]:
        """Get camera discovery status with Mini PC info"""
        next_discovery = None
        if self._last_discovery_time:
            next_discovery = self._last_discovery_time + timedelta(seconds=self._discovery_interval)
        
        # Get current camera status
        assigned_cameras = self._get_assigned_cameras() if self._mini_pc_info else []
        
        return {
            'last_discovery': self._last_discovery_time.isoformat() if self._last_discovery_time else None,
            'next_discovery': next_discovery.isoformat() if next_discovery else None,
            'interval_hours': self._discovery_interval / 3600,
            'known_cameras': len(self._known_camera_urls),
            'active_streams': len(self.get_active_streams()),
            'total_streams': len(self.stream_readers),
            'mini_pc_id': str(self._mini_pc_info.guid) if self._mini_pc_info else None,
            'mini_pc_name': self._mini_pc_info.device_name if self._mini_pc_info else None,
            'assigned_cameras_count': len(assigned_cameras),
            'discovery_mode': not bool(assigned_cameras)
        }
    
    def get_mini_pc_camera_status(self) -> Dict[str, Any]:
        """Get detailed Mini PC camera status"""
        try:
            assigned_cameras = self._get_assigned_cameras() if self._mini_pc_info else []
            
            camera_details = []
            for camera in assigned_cameras:
                try:
                    url = camera.generate_rtsp_url()
                    # Find corresponding stream
                    stream_status = "not_started"
                    for stream_id, reader in self.stream_readers.items():
                        if reader.url == url:
                            stream_status = reader.info.status.value
                            break
                    
                    camera_details.append({
                        'guid': str(camera.guid),
                        'ip_address': camera.ip_address,
                        'port': camera.port,
                        'rtsp_url': url,
                        'stream_status': stream_status,
                        'capabilities': {
                            'emotion': camera.detect_emotion,
                            'hands': camera.detect_hands,
                            'voice': camera.voice_detect,
                            'mask': camera.detect_mask,
                            'uniform': camera.detect_uniform
                        }
                    })
                except Exception as e:
                    logger.error(f"Error processing camera {camera.guid}: {e}")
            
            return {
                'mini_pc_registered': self._mini_pc_info is not None,
                'mini_pc_id': str(self._mini_pc_info.guid) if self._mini_pc_info else None,
                'mini_pc_name': self._mini_pc_info.device_name if self._mini_pc_info else None,
                'system_mac': self._system_mac_address,
                'assigned_cameras_count': len(assigned_cameras),
                'active_streams_count': len(self.get_active_streams()),
                'cameras': camera_details,
                'discovery_mode': len(assigned_cameras) == 0
            }
            
        except Exception as e:
            logger.error(f"Error getting Mini PC camera status: {e}")
            return {'error': str(e)}
    
    # Keep all your existing methods (_process_frames, _process_frame_batch, etc.)
    def _process_frames(self):
        """Main frame processing loop with centralized display"""
        logger.info("Frame processing thread started")
        
        while self.running:
            try:
                # Collect frames from all active streams
                active_frames = []
                
                for stream_id, reader in list(self.stream_readers.items()):
                    frame_data = reader.get_frame()
                    if frame_data:
                        active_frames.append(frame_data)
                
                # Process frames if any are available
                if active_frames:
                    self._process_frame_batch(active_frames)
                else:
                    time.sleep(0.005)
                
            except Exception as e:
                logger.error(f"Error in frame processing loop: {e}")
                time.sleep(0.01)
        
        logger.info("Frame processing thread ended")
        
    def _process_frame_batch(self, frames: List[FrameData]):
        """Process frames and pass to emotion processor for video creation"""
        try:
            if len(self.frame_processors) == 1 and len(frames) <= 6:
                processor = self.frame_processors[0]
                for frame_data in frames:
                    # Process frame and get detection result
                    detection_result = self._safe_process_frame_with_result(processor, frame_data)
                    
                    # Send to display manager
                    self.display_manager.add_frame_for_display(frame_data)
                    
                    # IMPORTANT: Pass frame to emotion processor for video generation
                    if detection_result and hasattr(self, '_emotion_processor') and self._emotion_processor:
                        try:
                            self._emotion_processor.process_detections(detection_result, frame_data.frame)
                        except Exception as e:
                            logger.error(f"Error in emotion processor: {e}")
                            
            elif len(self.frame_processors) > 1 or len(frames) > 4:
                with ThreadPoolExecutor(max_workers=min(2, len(frames))) as executor:
                    futures = []
                    for frame_data in frames:
                        for processor in self.frame_processors:
                            future = executor.submit(self._safe_process_frame_with_result, processor, frame_data)
                            futures.append((future, frame_data))
                    
                    for future, frame_data in futures:
                        try:
                            detection_result = future.result(timeout=2)
                            self.display_manager.add_frame_for_display(frame_data)
                            
                            # Pass to emotion processor
                            if detection_result and hasattr(self, '_emotion_processor') and self._emotion_processor:
                                try:
                                    self._emotion_processor.process_detections(detection_result, frame_data.frame)
                                except Exception as e:
                                    logger.error(f"Error in emotion processor: {e}")
                                    
                        except Exception as e:
                            logger.error(f"Frame processor error: {e}")
            else:
                if self.frame_processors:
                    processor = self.frame_processors[0]
                    for frame_data in frames:
                        detection_result = self._safe_process_frame_with_result(processor, frame_data)
                        self.display_manager.add_frame_for_display(frame_data)
                        
                        # Pass to emotion processor
                        if detection_result and hasattr(self, '_emotion_processor') and self._emotion_processor:
                            try:
                                self._emotion_processor.process_detections(detection_result, frame_data.frame)
                            except Exception as e:
                                logger.error(f"Error in emotion processor: {e}")
                else:
                    # No processors, just display frames
                    for frame_data in frames:
                        self.display_manager.add_frame_for_display(frame_data)
                        
        except Exception as e:
            logger.error(f"Error processing frame batch: {e}")
    
    def _safe_process_frame_with_result(self, processor: Callable, frame_data: FrameData):
        """Execute frame processor and return result"""
        try:
            # If processor returns result, use it; otherwise return None
            result = processor(frame_data)
            return result
        except Exception as e:
            logger.error(f"Frame processor {processor.__name__} error: {e}")
            return None
    def _safe_process_frame(self, processor: Callable, frame_data: FrameData):
        """Safely execute a frame processor"""
        try:
            processor(frame_data)
        except Exception as e:
            logger.error(f"Frame processor {processor.__name__} error: {e}")
    
    def _monitor_stats(self):
        """Monitor and log stream statistics with Mini PC and discovery info"""
        logger.info("Stream statistics monitoring started")
        
        last_detailed_log = time.time()
        detailed_log_interval = 300  
        
        while self.running:
            try:
                time.sleep(max(getattr(self.config.camera, 'discovery_interval_hours', 30), 30))
                
                stats = self.get_stream_info()
                active_count = len(self.get_active_streams())
                discovery_status = self.get_discovery_status()
                
                mini_pc_info = f" | Mini PC: {discovery_status.get('mini_pc_name', 'Unregistered')}"
                mode_info = f" | Mode: {'Discovery' if discovery_status.get('discovery_mode', False) else 'Assigned'}"
                
                logger.info(f"Stream Status - Active: {active_count}/{len(stats)}{mini_pc_info}{mode_info} | "
                          f"Next discovery in {discovery_status['interval_hours']:.1f} hours")
                
                # Detailed stats every 5 minutes
                current_time = time.time()
                if current_time - last_detailed_log >= detailed_log_interval:
                    last_detailed_log = current_time
                    
                    logger.info("=== Detailed Stream Statistics ===")
                    logger.info(f"Mini PC Status: {discovery_status}")
                    
                    for stream_id, info in stats.items():
                        if info.status == StreamStatus.ACTIVE:
                            logger.info(f"Stream {stream_id}: {info.name} - FPS={info.fps:.1f}, "
                                      f"Resolution={info.resolution}")
                        elif info.status in [StreamStatus.ERROR, StreamStatus.RECONNECTING]:
                            logger.warning(f"Stream {stream_id}: {info.name} - Status={info.status.value}, "
                                         f"Errors={info.error_count}, Reconnects={info.reconnect_count}")
                    
                    logger.info(f"Display Manager: {'Running' if self.display_manager.running else 'Stopped'}")
                
            except Exception as e:
                logger.error(f"Error in stats monitoring: {e}")
        
        logger.info("Stream statistics monitoring ended")