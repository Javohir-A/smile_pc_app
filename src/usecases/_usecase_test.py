import sys, os, logging
import face_recognition
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.di.dependencies import *
from src.config.settings import AppConfig

load_dotenv(".env.development")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_config = AppConfig.from_env()

_container = initialize_dependencies(_config)
face_usecase = get_face_usecase()

def _test_create_face():
    try:
        image = face_recognition.load_image_file("/home/javokhir/go/src/gitlab.com/udevs/smile/src/workers/photo_2025-06-03_16-54-22.jpg")
        
        face_encodings = face_recognition.face_encodings(image)
        if not face_encodings:
            logger.error("No face found in iamge")
            exit(1)
        
        if len(face_encodings) > 1:
            logger.warning("Multiple face found using the first one")
            
        print(face_encodings[0])
        
        face_embedding = face_usecase.create_face(
            human_guid="ceef8383-28fa-4407-bffd-ce1444ab9039", 
            name="Javokhir",
            human_type="employee" ,
            face_embedding=face_encodings[0], 
            metadata={"position":"golang dev"}
        )
            
        logger.info(f"Human Face added: {face_embedding}")
    
    except Exception as e:
        logger.error(f"Faield to embed human face: {e}")


def _test_get_face():
    print(face_usecase.get_faces_by_user("ceef8383-28fa-4407-bffd-ce1444ab9039"))
    
if __name__ == "__main__":
    _test_create_face()