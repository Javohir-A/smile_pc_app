# src/processors/stream_factory.py - UPDATED WITH VIDEO STORAGE
import time
from typing import Dict
import logging
from src.di.dependencies import DependencyContainer
from src.usecases.face_usecase import FaceUseCase
from src.processors.multi_stream_processor import MultiStreamProcessor
from src.config.settings import AppConfig
from src.processors.face_detection_processor import FaceDetectionProcessor, DetectionResult
from src.models.stream import FrameData
from src.processors.emotion_processor import SimpleEmotionProcessor

# logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class StreamProcessorFactory:
    """Factory for creating and configuring stream processors with centralized display and video storage"""
        
    @staticmethod
    def create_processor(config: AppConfig, container: DependencyContainer) -> MultiStreamProcessor:
        """Create stream processor with face detection, emotion tracking, video storage, and centralized display"""
        logger.info("Creating stream processor with centralized display, face detection, and video storage...")
        
        # Create multi-stream processor with centralized display
        processor = MultiStreamProcessor(config, container)
        # Create face detector
        face_detector = StreamProcessorFactory._create_face_detector(config, container)
        # Create emotion processor with video storage
        emotion_processor = SimpleEmotionProcessor(config)
        
        if face_detector:
            # Add face detection processor
            processor.add_frame_processor(face_detector.process_frame)
            processor.set_face_detector(face_detector)
            
            # Add callback for logging and emotion processing with video storage
            def process_callback(result: DetectionResult, frame_data: FrameData):
                try:
                    # Log recognized faces
                    recognized = [f for f in result.faces if f.is_recognized]
                    if recognized:
                        for face in recognized:
                            logger.debug(f"🎯 {face.human_name}: {face.emotion} "
                                    f"(confidence: {face.recognition_confidence:.1%})")
                                
                    # Process emotions AND video storage using SimpleEmotionProcessor
                    # Pass both detection result and frame for video processing
                    emotion_processor.process_detections(result, frame_data.frame)
                    
                except Exception as e:
                    logger.error(f"Error in callback: {e}")
            
            face_detector.add_detection_callback(process_callback)
            logger.info("✅ Face detection, emotion tracking, video storage, and centralized display enabled")
        else:
            logger.warning("⚠️ Face detection not available - display will show raw camera feeds")
        
        # Store emotion processor for cleanup and video storage
        processor._emotion_processor = emotion_processor
        
        # FIXED: Don't replace stop method during initialization - do it later
        # Store original stop method for later override
        processor._original_stop = processor.stop_all_streams
        
        # Initialize streams
        processor.initialize_streams()
        
        # NOW attach the enhanced cleanup method AFTER initialization
        def enhanced_stop():
            logger.info("Stopping streams, display manager, and cleaning up emotion/video processors...")
            if hasattr(processor, '_emotion_processor') and processor._emotion_processor:
                processor._emotion_processor.cleanup()
            processor._original_stop()
        
        processor.stop_all_streams = enhanced_stop
        
        logger.info("✅ Stream processor created with video storage and centralized display management")
        return processor
    
    @staticmethod
    def _create_face_detector(config: AppConfig, container: DependencyContainer) -> FaceDetectionProcessor:
        """Create and validate face detection processor"""
        try:
            logger.info("Initializing face detection processor...")
            
            # Create the face detector
            face_detector = FaceDetectionProcessor(config, container)
            
            # Validate that all components are properly initialized
            stats = face_detector.get_detection_stats()
            logger.info(f"Face detection initialization status: {stats}")
            
            # Check critical components
            if not stats['models_loaded']['dnn_model']:
                logger.error("DNN face detection model not loaded")
                return None
            
            if not stats['face_recognition_available']:
                logger.warning("Face recognition library not available - face recognition disabled")
            
            if not stats['face_usecase_available']:
                logger.error("Face usecase not available - face recognition will not work")
                logger.error("This means the database connection or face embedding search is not properly configured")
            
            # Log component status
            logger.info("Face Detection Components Status:")
            logger.info(f"  - DNN Model: {'✓' if stats['models_loaded']['dnn_model'] else '✗'}")
            logger.info(f"  - Emotion Recognizer: {'✓' if stats['models_loaded']['emotion_recognizer'] else '✗'}")
            logger.info(f"  - Face Recognition Library: {'✓' if stats['face_recognition_available'] else '✗'}")
            logger.info(f"  - Face Usecase (Database): {'✓' if stats['face_usecase_available'] else '✗'}")
            
            if stats['face_usecase_available'] and stats['face_recognition_available']:
                logger.info("✓ Face recognition is fully functional")
            elif stats['models_loaded']['dnn_model']:
                logger.warning("⚠ Face detection available but recognition may be limited")
            else:
                logger.error("✗ Face detection system not functional")
                return None
            
            return face_detector
            
        except Exception as e:
            logger.error(f"Failed to create face detection processor: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    @staticmethod
    def validate_dependencies(container: DependencyContainer) -> Dict[str, bool]:
        """Validate that all required dependencies are available"""
        validation_results = {}
        
        try:
            mini_pc_usecase = container.get_mini_pc_usecase()
            validation_results['mini_pc_usecase'] = mini_pc_usecase is not None
        except Exception as e:
            logger.error(f"Mini PC usecase validation failed: {e}")
            validation_results['mini_pc_usecase'] = False
            
        try:
            # Test camera usecase
            camera_usecase = container.get_camera_usecase()
            validation_results['camera_usecase'] = camera_usecase is not None
        except Exception as e:
            logger.error(f"Camera usecase validation failed: {e}")
            validation_results['camera_usecase'] = False
        
        try:
            # Test face usecase
            face_usecase: FaceUseCase = container.get_face_usecase()
            validation_results['face_usecase'] = face_usecase is not None
            
            if face_usecase:
                # Test face usecase functionality
                test_embedding = [0.0] * 128
                test_results = face_usecase.search_similar_faces(test_embedding, limit=1, threshold=50)
                validation_results['face_search_functional'] = True
                validation_results["milvus_client"] = True
                validation_results["face_repository"] = True
                logger.info("Face usecase search test passed")
                
        except Exception as e:
            logger.error(f"Face usecase validation failed: {e}")
            validation_results['face_usecase'] = False
            validation_results['face_search_functional'] = False
            validation_results["milvus_client"] = False
            validation_results["face_repository"] = False
            validation_results["mini_pc_usecase"] = False
        
        # Validate video storage capabilities
        try:
            # Check if ucode-sdk is available for video uploads
            from ucode_sdk.config import Config
            from ucode_sdk.sdk import new
            validation_results['video_upload_available'] = True
            logger.info("Video storage and upload capabilities available")
        except ImportError:
            validation_results['video_upload_available'] = False
            logger.warning("Video upload not available - ucode-sdk not installed")
        except Exception as e:
            validation_results['video_upload_available'] = False
            logger.error(f"Video upload validation failed: {e}")
        
        # Log validation results
        logger.info("Dependency Validation Results:")
        for component, status in validation_results.items():
            status_symbol = "✓" if status else "✗"
            logger.info(f"  {status_symbol} {component}: {'Available' if status else 'Not Available'}")
        
        return validation_results
    
    @staticmethod
    def create_with_face_detection(config: AppConfig, container: DependencyContainer) -> MultiStreamProcessor:
        """Create stream processor with face detection and recognition (alias for backward compatibility)"""
        return StreamProcessorFactory.create_processor(config, container)