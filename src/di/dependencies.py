# src/di/dependencies.py
from src.usecases.mini_pc_usecase import MiniPCUseCase
from src.repositories.ucode_impl.mini_pc_repository_ucode_impl import MiniPCRepositoryUcodeImpl

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
import logging

from src.config import AppConfig
from src.usecases.camer_usecase import CameraUseCase
from src.repositories.ucode_impl.camera_repository_ucode_impl import CameraRepositoryUcodeImpl 
from src.repositories.face_repository import FaceRepository
from src.repositories.relational_db.milvus_repository_impl import MilvusFaceRepository
from src.usecases.face_usecase import FaceUseCase

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, config: AppConfig):
        self.config: AppConfig = config
        self._engine = None
        self._session_factory = None
        self._initialize_postgres_database()
    
    def _initialize_postgres_database(self):
        """Initialize database engine and session factory"""
        try:
            database_url = self.config.database.postgres.connection_string
            logger.info(f"Initializing database connection to: {self.config.database.postgres.db_host}")
            
            # Create engine with connection pooling
            self._engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.config.database.postgres.max_connections,
                max_overflow=10,
                pool_timeout=self.config.database.postgres.connection_timeout,
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=self.config.debug,  # Log SQL queries in debug mode
                connect_args={
                    "connect_timeout": self.config.database.postgres.connection_timeout,
                    "sslmode": self.config.database.postgres.db_sslmode
                }
            )
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False
            )
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    @property
    def engine(self):
        """Get database engine"""
        return self._engine
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self._session_factory:
            raise RuntimeError("Database not initialized")
        return self._session_factory()
    
    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


class DependencyContainer:
    """Dependency injection container"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self._mini_pc_usecase: MiniPCUseCase = None
        
        # Initialize face repository and usecase as instance variables
        self._face_repository: FaceRepository = None
        self._face_usecase: FaceUseCase = None
        
        logger.info("Dependency container initialized")
    
    def get_camera_usecase(self) -> CameraUseCase:
        """Get CameraUsecase with proper session management"""
        # session = self.db_manager.get_session()
        camera_impl = CameraRepositoryUcodeImpl(self.config)
        usecase = CameraUseCase(camera_impl)
        return usecase
    
    def get_face_repository(self) -> FaceRepository:
        """Get face repository instance (singleton)"""
        if self._face_repository is None:
            self._face_repository = MilvusFaceRepository(self.config.external.milvus)
            logger.info("Face repository initialized")
        return self._face_repository
    
    def get_face_usecase(self) -> FaceUseCase:
        """Get face use case instance (singleton)"""
        if self._face_usecase is None:
            face_repository = self.get_face_repository()
            self._face_usecase = FaceUseCase(face_repository)
            logger.info("Face usecase initialized")
        return self._face_usecase
    
    def get_mini_pc_usecase(self) -> MiniPCUseCase:
        """Get Mini PC usecase with proper session management"""
        if self._mini_pc_usecase is None:
            mini_pc_repo = MiniPCRepositoryUcodeImpl(self.config)
            camera_repo = get_camera_usecase()
            self._mini_pc_usecase = MiniPCUseCase(mini_pc_repo, camera_repo)
            logger.info("Mini PC usecase initialized")
        return self._mini_pc_usecase
    
    @contextmanager
    def get_camera_usecase_context(self) -> Generator[CameraUseCase, None, None]:
        """Get CameraUsecase with automatic session cleanup"""
        camera_impl = CameraRepositoryUcodeImpl(self.config)
        usecase = CameraUseCase(camera_impl)
        yield usecase
    
    def get_database_session(self) -> Session:
        """Get raw database session (use with caution)"""
        return self.db_manager.get_session()
    
    @contextmanager
    def get_database_session_context(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        with self.db_manager.get_session_context() as session:
            yield session
            
    @contextmanager
    def get_mini_pc_usecase_context(self) -> Generator[MiniPCUseCase, None, None]:
        """Get Mini PC usecase with automatic session cleanup"""
        with self.db_manager.get_session_context() as session:
            mini_pc_repo = MiniPCRepositoryUcodeImpl(self.config)
            camera_repo = CameraRepositoryUcodeImpl(self.config)
            usecase = MiniPCUseCase(mini_pc_repo, camera_repo)
            yield usecase

    def close(self):
        """Close all resources"""
        self.db_manager.close()
        # Reset singletons
        self._face_repository = None
        self._face_usecase = None


# Global dependency container (initialized later)
_container: DependencyContainer = None


def initialize_dependencies(config: AppConfig) -> DependencyContainer:
    """Initialize the global dependency container"""
    global _container
    _container = DependencyContainer(config)
    logger.info("Global dependencies initialized")
    return _container


def get_dependency_container() -> DependencyContainer:
    """Get the global dependency container"""
    if _container is None:
        raise RuntimeError("Dependencies not initialized. Call initialize_dependencies() first.")
    return _container


def get_camera_usecase() -> CameraUseCase:
    """Get CameraUsecase instance"""
    return get_dependency_container().get_camera_usecase()

def get_face_usecase() -> FaceUseCase:
    """Get FaceUseCase instance"""
    return get_dependency_container().get_face_usecase()

def get_mini_pc_usecase() -> MiniPCUseCase:
    """Get Mini PC usecase instance"""
    return get_dependency_container().get_mini_pc_usecase()

def get_database_session() -> Session:
    """Get database session"""
    return get_dependency_container().get_database_session()


# Context manager functions for proper resource management
def get_camera_usecase_context():
    """Get CameraUsecase with automatic session cleanup"""
    return get_dependency_container().get_camera_usecase_context()

def get_mini_pc_usecase_context():
    """Get Mini PC usecase with automatic session cleanup"""
    return get_dependency_container().get_mini_pc_usecase_context()

def get_database_session_context():
    """Get database session with automatic cleanup"""
    return get_dependency_container().get_database_session_context()


def shutdown_dependencies():
    """Shutdown all dependencies"""
    global _container
    if _container:
        _container.close()
        _container = None
    logger.info("Dependencies shutdown complete")


# Example usage in main application:
"""
# In main.py or app initialization:
from config import AppConfig
from di.dependencies import initialize_dependencies, shutdown_dependencies
import atexit

# Initialize configuration
config = AppConfig.from_env()

# Initialize dependencies
container = initialize_dependencies(config)

# Register cleanup function
atexit.register(shutdown_dependencies)

# Use dependencies in your application:
from di.dependencies import (
    get_user_usecase_context, 
    get_camera_usecase_context, 
    get_face_usecase
)

# Example 1: Using context managers (recommended for DB operations)
with get_user_usecase_context() as user_usecase:
    users = user_usecase.get_all_users()

# Example 2: Using face usecase (singleton, no context needed)
face_usecase = get_face_usecase()
embeddings = face_usecase.search_similar_faces(embedding_vector)

# Example 3: Using direct access (manual session management)
user_usecase = get_user_usecase()
try:
    users = user_usecase.get_all_users()
finally:
    # You need to manually close the session
    user_usecase.repository.session.close()
"""