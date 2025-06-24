# config/database.py
import os
from dataclasses import dataclass
from src.config.base import BaseConfig

@dataclass
class PostgresConfig(BaseConfig):
    """PostgreSQL database configuration"""
    db_name: str = "udevs_4a17598bc0d14a18869d1b40d22ba7f3_p_postgres_svcs"
    db_user: str = "udevs_4a17598bc0d14a18869d1b40d22ba7f3_p_postgres_svcs"
    db_password: str = "7FZNexacHb"
    db_host: str = "142.93.164.37"
    db_port: str = "30032"
    db_sslmode: str = "disable"
    connection_timeout: int = 30
    max_connections: int = 20
    
    
    @classmethod
    def from_env(cls) -> 'PostgresConfig':
        return cls(
            db_name=os.getenv('DB_NAME', cls.db_name),
            db_user=os.getenv('DB_USER', cls.db_user),
            db_password=os.getenv('DB_PASSWORD', cls.db_password),
            db_host=os.getenv('DB_HOST', cls.db_host),
            db_port=os.getenv('DB_PORT', cls.db_port),
            db_sslmode=os.getenv('DB_SSLMODE', cls.db_sslmode),
            connection_timeout=cls.get_env_int('DB_CONNECTION_TIMEOUT', 30),
            max_connections=cls.get_env_int('DB_MAX_CONNECTIONS', 20)
        )
    
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}?sslmode={self.db_sslmode}"

@dataclass
class DatabaseConfig(BaseConfig):
    """Main database configuration"""
    postgres: PostgresConfig = None
    
    def __post_init__(self):
        if self.postgres is None:
            self.postgres = PostgresConfig.from_env()
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        return cls(
            postgres=PostgresConfig.from_env()
        )
