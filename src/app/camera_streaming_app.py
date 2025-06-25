# Enhanced main.py with Mini PC MAC address validation and camera discovery

import asyncio
from dotenv import load_dotenv
import sys
import logging
import signal
import psutil
import os
import subprocess
from src.processors.stream_factory import StreamProcessorFactory
from src.config.settings import AppConfig
from src.config import AppConfig
from src.di.dependencies import initialize_dependencies, DependencyContainer
from .monitoring import ResourceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamApplication:
    """Main application class for handling stream processing with auto camera discovery and face recognition"""
    
    def __init__(self, config: AppConfig, dependency_container: DependencyContainer):
        self.config = config
        self.dependency_container = dependency_container
        self.processor = None
        self.shutdown_event = asyncio.Event()
        self.memory_monitor_task = None
        self.command_handler_task = None
        self.validation_results = {}
        
        # Mini PC information
        self.mini_pc_info = None
        self.assigned_cameras = []
        
        self.resource_monitor = ResourceMonitor()
        
        self.fastapi_task = None
    
    def get_system_mac_address(self) -> str:
        """Get the MAC address of the primary network interface"""
        try:
            # Get MAC address of primary network interface
            result = subprocess.check_output(
                "ip link show | grep 'link/ether' | awk '{print $2}' | head -1", 
                shell=True
            ).decode().strip()
            
            if result:
                logger.info(f"🔍 Detected system MAC address: {result}")
                return result
            else:
                # Fallback method
                import uuid
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff)
                               for elements in range(0,2*6,2)][::-1])
                logger.info(f"🔍 Fallback MAC address: {mac}")
                return mac
                
        except Exception as e:
            logger.error(f"❌ Error getting MAC address: {e}")
            # Last resort fallback
            import uuid
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                           for elements in range(0,2*6,2)][::-1])
            logger.warning(f"⚠️  Using fallback MAC address: {mac}")
            return mac
    
    def get_system_info(self) -> dict:
        """Get comprehensive system information"""
        try:
            hostname = subprocess.check_output("hostname", shell=True).decode().strip()
            ip_address = subprocess.check_output(
                "hostname -I | awk '{print $1}'", shell=True
            ).decode().strip()
            
            return {
                "device_name": hostname,
                "mac_address": self.get_system_mac_address(),
                "ip_address": ip_address,
                "port": 8080  # Default API port
            }
        except Exception as e:
            logger.error(f"❌ Error getting system info: {e}")
            return {
                "device_name": "unknown-device",
                "mac_address": self.get_system_mac_address(),
                "ip_address": "127.0.0.1",
                "port": 8080
            }
    
    async def validate_mini_pc_registration(self):
        """Validate that this Mini PC is registered in the system and get its cameras"""
        logger.info("🖥️  Validating Mini PC registration...")
        
        try:
            system_info = self.get_system_info()
            logger.info(f"📋 System Info: {system_info['device_name']} ({system_info['mac_address']})")
            
            mini_pc_usecase = self.dependency_container.get_mini_pc_usecase()
            
            mini_pc = mini_pc_usecase.get_mini_pc_by_mac(system_info['mac_address'])
            
            if mini_pc:
                logger.info(f"✅ Mini PC found in database: {mini_pc.device_name}")
                logger.info(f"   ID: {mini_pc.guid}")
                logger.info(f"   Branch: {mini_pc.branch_id}")
                logger.info(f"   Company: {mini_pc.company_id}")
                logger.info(f"   Status: {'Active' if mini_pc.is_active else 'Inactive'}")
                
                if not mini_pc.is_active:
                    logger.warning("⚠️  Mini PC is marked as inactive in database")
                    return False
                
                self.mini_pc_info = mini_pc
                
                # Get assigned cameras
                logger.info("📹 Retrieving assigned cameras...")
                cameras = mini_pc_usecase.get_mini_pc_cameras(mini_pc_id=mini_pc.guid)
                
                if cameras:
                    logger.info(f"✅ Found {len(cameras)} assigned cameras:")
                    for i, camera in enumerate(cameras, 1):
                        logger.info(f"   {i}. Camera {camera.ip_address}:{camera.port}")
                        logger.info(f"      - Capabilities: Emotion={camera.detect_emotion}, "
                                  f"Hands={camera.detect_hands}, Voice={camera.voice_detect}")
                        logger.info(f"      - RTSP URL: {camera.generate_rtsp_url()}")
                    
                    self.assigned_cameras = cameras
                    
                    # Update config with discovered cameras
                    await self._update_config_with_cameras(cameras)
                    
                else:
                    logger.warning("⚠️  No cameras assigned to this Mini PC")
                    logger.info("💡 Add cameras to this Mini PC in the admin panel")
                
                return True
                
            else:
                logger.warning(f"⚠️  Mini PC with MAC {system_info['mac_address']} not found in database")
                logger.info("💡 Register this Mini PC in the admin panel:")
                logger.info(f"   - Device Name: {system_info['device_name']}")
                logger.info(f"   - MAC Address: {system_info['mac_address']}")
                logger.info(f"   - IP Address: {system_info['ip_address']}")
                logger.info(f"   - Port: {system_info['port']}")
                
                # Option to auto-register (if you want this feature)
                if hasattr(self.config, 'auto_register_mini_pc') and self.config.auto_register_mini_pc:
                    return await self._auto_register_mini_pc(system_info, mini_pc_usecase)
                
                return False
                
        except Exception as e:
            logger.error(f"❌ Error validating Mini PC registration: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def _auto_register_mini_pc(self, system_info: dict, mini_pc_usecase) -> bool:
        """Auto-register this Mini PC (optional feature)"""
        logger.info("🤖 Auto-registering Mini PC...")
        
        try:
            # You need to provide default branch_id and company_id
            # These could come from environment variables or config
            default_branch_id = os.getenv('DEFAULT_BRANCH_ID')
            default_company_id = os.getenv('DEFAULT_COMPANY_ID')
            
            if not default_branch_id or not default_company_id:
                logger.error("❌ Auto-registration requires DEFAULT_BRANCH_ID and DEFAULT_COMPANY_ID environment variables")
                return False
            
            from src.models.mini_pc import MiniPC
            from uuid import UUID
            
            new_mini_pc = MiniPC(
                device_name=system_info['device_name'],
                mac_address=system_info['mac_address'],
                ip_address=system_info['ip_address'],
                port=system_info['port'],
                branch_id=UUID(default_branch_id),
                company_id=UUID(default_company_id),
                is_active=True
            )
            
            created_mini_pc = mini_pc_usecase.create_mini_pc(new_mini_pc)
            logger.info(f"✅ Auto-registered Mini PC: {created_mini_pc.guid}")
            
            self.mini_pc_info = created_mini_pc
            return True
            
        except Exception as e:
            logger.error(f"❌ Error auto-registering Mini PC: {e}")
            return False
    
    async def _update_config_with_cameras(self, cameras):
        """Update application config with discovered cameras"""
        try:
            # Convert cameras to RTSP URLs
            rtsp_urls = []
            for camera in cameras:
                rtsp_url = camera.generate_rtsp_url()
                rtsp_urls.append(rtsp_url)
            
            # Update config
            if hasattr(self.config, 'camera') and hasattr(self.config.camera, 'rtsp_urls'):
                self.config.camera.rtsp_urls = rtsp_urls
                logger.info(f"📺 Updated config with {len(rtsp_urls)} camera URLs")
            else:
                logger.warning("⚠️  Config doesn't support dynamic camera URLs")
            
        except Exception as e:
            logger.error(f"❌ Error updating config with cameras: {e}")
        
    async def validate_system(self):
        """Validate system dependencies and configuration"""
        logger.info("🔍 Validating system dependencies...")
        
        # 1. Validate Mini PC registration first
        mini_pc_valid = await self.validate_mini_pc_registration()
        if not mini_pc_valid:
            logger.error("❌ Mini PC validation failed - cannot start application")
            logger.info("💡 Please register this Mini PC in the admin panel or check network connectivity")
            return False
        
        # 2. Validate configuration
        if not self.config.validate():
            logger.error("❌ Configuration validation failed")
            return False
        
        logger.info("✅ Configuration validation passed")
        
        # 3. Validate dependencies
        self.validation_results = StreamProcessorFactory.validate_dependencies(self.dependency_container)
        
        # Check critical dependencies
        critical_deps = ['camera_usecase']
        face_recognition_deps = ['face_usecase', 'face_repository', 'milvus_client']
        
        # Check if critical dependencies are available
        critical_available = all(self.validation_results.get(dep, False) for dep in critical_deps)
        if not critical_available:
            logger.error("❌ Critical dependencies not available - cannot start application")
            missing_deps = [dep for dep in critical_deps if not self.validation_results.get(dep, False)]
            logger.error(f"Missing critical dependencies: {missing_deps}")
            return False
        
        # Check face recognition dependencies
        face_recognition_available = all(self.validation_results.get(dep, False) for dep in face_recognition_deps)
        if face_recognition_available:
            logger.info("✅ Face recognition system fully available")
        else:
            logger.warning("⚠️  Face recognition system partially available or disabled")
            missing_deps = [dep for dep in face_recognition_deps if not self.validation_results.get(dep, False)]
            logger.warning(f"Missing face recognition dependencies: {missing_deps}")
        
        # 4. Log camera summary
        logger.info("📹 Camera Summary:")
        logger.info(f"   - Assigned cameras: {len(self.assigned_cameras)}")
        if self.assigned_cameras:
            active_cameras = sum(1 for c in self.assigned_cameras if getattr(c, 'is_active', True))
            emotion_cameras = sum(1 for c in self.assigned_cameras if c.detect_emotion)
            logger.info(f"   - Active cameras: {active_cameras}")
            logger.info(f"   - Emotion detection cameras: {emotion_cameras}")
        
        logger.info("✅ System validation completed successfully")
        return True
    
    async def get_mini_pc_status(self) -> dict:
        """Get current Mini PC status for monitoring"""
        try:
            system_info = self.get_system_info()
            
            status = {
                "mini_pc_id": str(self.mini_pc_info.guid) if self.mini_pc_info else None,
                "device_name": system_info['device_name'],
                "mac_address": system_info['mac_address'],
                "ip_address": system_info['ip_address'],
                "is_registered": self.mini_pc_info is not None,
                "is_active": self.mini_pc_info.is_active if self.mini_pc_info else False,
                "assigned_cameras_count": len(self.assigned_cameras),
                "active_cameras_count": sum(1 for c in self.assigned_cameras if getattr(c, 'is_active', True)),
                "branch_id": str(self.mini_pc_info.branch_id) if self.mini_pc_info else None,
                "company_id": str(self.mini_pc_info.company_id) if self.mini_pc_info else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting Mini PC status: {e}")
            return {"error": str(e)}

    # Rest of your existing methods remain the same...
    async def start_memory_monitor(self):
        """Monitor memory usage and log warnings"""
        while not self.shutdown_event.is_set():
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                if memory_mb > 1000:  # Warn if over 1GB
                    logger.warning(f"🧠 High memory usage: {memory_mb:.1f} MB")
                
                if memory_mb > 2000:  # Critical if over 2GB
                    logger.critical(f"🚨 Critical memory usage: {memory_mb:.1f} MB - consider restarting")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring memory: {e}")
                await asyncio.sleep(30)
    
    async def start_command_handler(self):
        """Handle runtime commands for camera management"""
        logger.info("⌨️  Command handler started")
        logger.info("Commands available:")
        logger.info("  - Press Ctrl+C to shutdown gracefully")
        logger.info("  - Check logs for camera discovery and face recognition status")
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in command handler: {e}")
                await asyncio.sleep(1)

    async def start(self):
        """Start the stream processing application with validation and auto discovery"""
        try:
            logger.info("🚀 Starting Mini PC application with camera discovery...")
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring(interval=15)
            
            # Validate system first (includes Mini PC and camera discovery)
            if not await self.validate_system():
                logger.error("❌ System validation failed - cannot start application")
                return
            
            # Log Mini PC status
            mini_pc_status = await self.get_mini_pc_status()
            logger.info(f"🖥️  Mini PC Status: {mini_pc_status}")
            
            # Start monitoring tasks
            self.memory_monitor_task = asyncio.create_task(self.start_memory_monitor())
            self.command_handler_task = asyncio.create_task(self.start_command_handler())
            
            # Create and configure the stream processor
            logger.info("🏭 Creating stream processor with face recognition...")
            factory = StreamProcessorFactory()
            self.processor = factory.create_processor(self.config, self.dependency_container)
            
            if not self.processor:
                logger.error("❌ Failed to create stream processor")
                return
            
            # Configure camera discovery
            discovery_interval = getattr(self.config.camera, 'discovery_interval_hours', 1.0)
            self.processor.set_discovery_interval(discovery_interval)
            logger.info(f"📡 Camera discovery interval set to {discovery_interval} hours")
            
            # Start all streams
            logger.info("▶️  Starting all camera streams...")
            self.processor.start_all_streams()
            
            if hasattr(self.processor, 'display_manager') and self.processor.display_manager.fastapi_server:
                try:
                    self.fastapi_task = asyncio.create_task(
                        self.processor.display_manager.start_fastapi_server()
                    )
                    logger.info(f"🌐 FastAPI server starting on port {getattr(self.config, 'WEBSOCKET_PORT', 8765)}")
                    await asyncio.sleep(0.5)  # Give it time to start
                except Exception as e:
                    logger.error(f"Failed to start FastAPI server: {e}")
                    
            # Log initial status
            discovery_status = self.processor.get_discovery_status()
            active_streams = self.processor.get_active_streams()
            
            logger.info("✅ Mini PC application started successfully!")
            logger.info(f"📊 Initial Status:")
            logger.info(f"  - Mini PC: {self.mini_pc_info.device_name if self.mini_pc_info else 'Unknown'}")
            logger.info(f"  - Assigned cameras: {len(self.assigned_cameras)}")
            logger.info(f"  - Active streams: {len(active_streams)}")
            logger.info(f"  - Face recognition: {'Enabled' if self.validation_results.get('face_usecase', False) else 'Disabled'}")
            
            if self.fastapi_task and not self.fastapi_task.done():
                port = self.config.WEBSOCKET_PORT
                logger.info(f"🌐 FastAPI server starting on port {self.config.WEBSOCKET_PORT}")
                logger.info(f"  - FastAPI server: http://localhost:{port}")
                logger.info(f"  - WebSocket streaming: ws://localhost:{port}/ws/")
                logger.info(f"  - API docs: http://localhost:{port}/docs")
            
            # Set up periodic status logging
            asyncio.create_task(self._periodic_status_log())
            asyncio.create_task(self._periodic_resource_report())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"❌ Error starting application: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            await self.shutdown()

    # Rest of your existing methods (periodic_status_log, shutdown, etc.) remain the same...
    
    async def _periodic_status_log(self):
        """Log application status periodically"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                if self.processor:
                    discovery_status = self.processor.get_discovery_status()
                    active_streams = self.processor.get_active_streams()
                    stream_info = self.processor.get_stream_info()
                    mini_pc_status = await self.get_mini_pc_status()
                    
                    logger.info("📈 === Mini PC Application Status Report ===")
                    logger.info(f"🖥️  Mini PC: {mini_pc_status.get('device_name', 'Unknown')} ({mini_pc_status.get('mac_address', 'Unknown')})")
                    logger.info(f"📹 Assigned cameras: {mini_pc_status.get('assigned_cameras_count', 0)}")
                    logger.info(f"🎥 Active streams: {len(active_streams)}")
                    logger.info(f"📡 Last camera discovery: {discovery_status.get('last_discovery', 'N/A')}")
                    
                    # Log stream health
                    healthy_streams = sum(1 for info in stream_info.values() if info.status.value == 'active')
                    error_streams = sum(1 for info in stream_info.values() if info.status.value in ['error', 'reconnecting'])
                    
                    logger.info(f"💚 Healthy streams: {healthy_streams}")
                    if error_streams > 0:
                        logger.warning(f"🔴 Streams with issues: {error_streams}")
                    
                    # Memory usage
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    logger.info(f"🧠 Memory usage: {memory_mb:.1f} MB")
                    
                    # ADD THIS DEBUG CODE HERE:
                    if hasattr(self.processor, '_emotion_processor') and self.processor._emotion_processor:
                        emotion_processor = self.processor._emotion_processor
                        
                        # Debug upload service status
                        if hasattr(emotion_processor, 'upload_service'):
                            stats = emotion_processor.upload_service.get_stats()
                            logger.info(f"🔍 Upload Service Debug: {stats}")
                            logger.info(f"📋 Queue Size: {emotion_processor.upload_service.get_queue_size()}")
                            logger.info(f"🏃 Service Running: {emotion_processor.upload_service.running}")
                            logger.info(f"🔗 ucode Available: {emotion_processor.upload_service.ucode_available}")
                            
                            # Check if ucode-sdk is working
                            if emotion_processor.upload_service.ucode_available:
                                try:
                                    logger.info(f"📱 ucode SDK Config: {emotion_processor.upload_service.app_id}")
                                    logger.info(f"🌐 ucode Base URL: {emotion_processor.upload_service.base_url}")
                                except Exception as e:
                                    logger.error(f"❌ ucode SDK Error: {e}")
                        else:
                            logger.error("❌ Upload service not found!")
            except Exception as e:
                logger.error(f"Error in periodic status log: {e}")

    async def _periodic_resource_report(self):
        """Report resource usage periodically"""
        while not self.shutdown_event.is_set() if self.shutdown_event else True:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                # Get current resource usage
                display = self.resource_monitor.get_realtime_display()
                logging.info(f"📊 Resource Status: {display}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in resource reporting: {e}")

    async def shutdown(self):
        """Gracefully shutdown the application"""
        logger.info("🛑 Shutting down Mini PC application...")

        if self.processor and hasattr(self.processor, 'display_manager'):
            try:
                await self.processor.display_manager.stop_fastapi_server()
            except Exception as e:
                logger.error(f"Error stopping FastAPI server: {e}")
        
        if self.fastapi_task and not self.fastapi_task.done():
            self.fastapi_task.cancel()
            try:
                await self.fastapi_task
            except asyncio.CancelledError:
                pass
        # Log final Mini PC status
        try:
            mini_pc_status = await self.get_mini_pc_status()
            logger.info(f"📊 Final Mini PC Status: {mini_pc_status}")
        except Exception as e:
            logger.error(f"Error getting final status: {e}")
        
        # Cancel monitoring tasks
        for task, name in [(self.memory_monitor_task, "memory monitor"), 
                          (self.command_handler_task, "command handler")]:
            if task:
                logger.info(f"Stopping {name}...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        if self.processor:
            try:
                # Stop processor
                self.processor.stop_all_streams()
                logger.info("✅ Stream processor stopped")
                
            except Exception as e:
                logger.error(f"Error stopping stream processor: {e}")
        
        self.resource_monitor.stop_monitoring()
        
        logger.info("👋 Mini PC application shutdown complete")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self._set_shutdown_event())
    
    async def _set_shutdown_event(self):
        """Set the shutdown event"""
        self.shutdown_event.set()

# Main function remains the same
async def main():
    """Main application entry point"""
    try:
        # Parse command line arguments
        env_version = "production"
        if len(sys.argv) > 1:
            versions = ["development", "production", "minipc-01", "minipc-02"]
            if sys.argv[1] not in versions:
                print(f"❌ Environment version must be one of: {versions}")
                sys.exit(1)
            env_version = sys.argv[1]
        
        # Load configuration from environment
        logger.info(f"🔧 Loading configuration for environment: {env_version}")
        load_dotenv(f'.env.{env_version}')
        config = AppConfig.from_env()
        
        # Add camera discovery configuration if not present
        if not hasattr(config.camera, 'discovery_interval_hours'):
            config.camera.discovery_interval_hours = 1.0  # Default to 1 hour
        
        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"📝 Log file: {config.log_file}")
        logger.info(f"📡 Camera discovery interval: {str(config.camera.discovery_interval_hours) + 'hours' if config.camera.discovery_interval_hours >= 1 else str(config.camera.discovery_interval_hours) * 60 + 'minutes'} ")
        logger.info(f"🖥️  GUI enabled: {'Yes' if config.enable_gui else 'No'}")
        
        # Initialize dependencies
        logger.info("🔌 Initializing dependencies...")
        dependency_container = initialize_dependencies(config)
        logger.info("✅ Dependencies initialized")
        
        # Create and start application
        app = StreamApplication(config, dependency_container)
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in [signal.SIGTERM, signal.SIGINT]:
            loop.add_signal_handler(sig, app.signal_handler, sig, None)
        
        # Start the application
        await app.start()
        
    except KeyboardInterrupt:
        logger.info("⌨️  Application interrupted by user")
    except Exception as e:
        logger.error(f"💥 Application error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())