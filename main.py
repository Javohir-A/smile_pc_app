import asyncio
import sys
import logging
from src.app.camera_streaming_app import main

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

import threading

# Thread creation limiter
original_thread_init = threading.Thread.__init__

def limited_thread_init(self, *args, **kwargs):
    if threading.active_count() > 100:
        raise Exception(f"🚨 Thread limit reached: {threading.active_count()}")
    return original_thread_init(self, *args, **kwargs)

threading.Thread.__init__ = limited_thread_init

if __name__ == "__main__":
    try:
        logger.info("===== Stream Processing Application Starting =====")
        logger.info("Features enabled:")
        logger.info("- Face Detection with Emotion Recognition")
        logger.info("- Face Recognition with Milvus Integration")
        logger.info("- Automatic Camera Discovery (hourly)")
        logger.info("- Dynamic Stream Management")
        logger.info("- Real-time Display with Name Labels")
        logger.info("=================================================")
        
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
        
import time

def some_func():
    time.sleep(3600)
        
for i in range(1, 11):
    thread = threading.Thread(target=some_func)