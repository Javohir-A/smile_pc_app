# src/repositories/ucode_impl/camera_repository_ucode_impl.py - FIXED VERSION
from ucode_sdk.config import Config
from ucode_sdk.sdk import new
from uuid import UUID
from typing import Optional, List, Dict, Any
import logging
from src.models.camera import Camera
from src.models.filters import GetListFilter, Filter
from src.config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraRepositoryUcodeImpl:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ucode_api = new(Config(config.ucode.app_id, config.ucode.base_url, __name__))
        self.table_slug: str = "camera_"
        
    def _safe_get_attribute(self, obj, attr_name, default=None):
        """Safely get attribute from object, handling various data types"""
        try:
            if hasattr(obj, attr_name):
                return getattr(obj, attr_name)
            elif isinstance(obj, dict):
                return obj.get(attr_name, default)
            elif hasattr(obj, '__dict__'):
                return obj.__dict__.get(attr_name, default)
            else:
                logger.debug(f"Cannot get attribute '{attr_name}' from {type(obj)}")
                return default
        except Exception as e:
            logger.debug(f"Error getting attribute '{attr_name}': {e}")
            return default
    
    def _extract_data_from_response(self, result) -> Optional[Dict]:
        """Extract data from UCode response, handling various response formats"""
        try:
            if not result:
                return None
            
            # Method 1: Direct data attribute
            if hasattr(result, 'data') and result.data:
                data = result.data
                
                # If data is already a dict, return it
                if isinstance(data, dict):
                    return data
                
                # If data has __dict__, convert to dict
                if hasattr(data, '__dict__'):
                    return data.__dict__
                
                # If data is a string, it might be an error message
                if isinstance(data, str):
                    logger.warning(f"UCode returned string data: {data}")
                    return None
            
            # Method 2: Check for data_container
            if hasattr(result, 'data_container'):
                container = result.data_container
                if hasattr(container, 'data') and isinstance(container.data, dict):
                    response_data = container.data.get('response')
                    if response_data:
                        return response_data[0] if isinstance(response_data, list) and response_data else response_data
            
            # Method 3: Direct dict access
            if isinstance(result, dict):
                return result
                
            logger.warning(f"Could not extract data from result type: {type(result)}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting data from response: {e}")
            return None
    
    def _extract_list_from_response(self, result) -> List[Dict]:
        """Extract list data from UCode response"""
        try:
            if not result:
                return []
            
            # Method 1: Check data_container first (most common for list responses)
            if hasattr(result, 'data_container'):
                container = result.data_container
                if hasattr(container, 'data') and isinstance(container.data, dict):
                    response_data = container.data.get('response', [])
                    if isinstance(response_data, list):
                        return [self._normalize_item_data(item) for item in response_data]
            
            # Method 2: Direct data attribute
            if hasattr(result, 'data'):
                data = result.data
                
                # If data is a list, process each item
                if isinstance(data, list):
                    return [self._normalize_item_data(item) for item in data]
                
                # If data has nested data attribute
                if hasattr(data, 'data') and isinstance(data.data, list):
                    return [self._normalize_item_data(item) for item in data.data]
            
            logger.warning(f"Could not extract list from result type: {type(result)}")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting list from response: {e}")
            return []
    
    def _normalize_item_data(self, item) -> Dict:
        """Normalize item data to dictionary format"""
        try:
            if isinstance(item, dict):
                return item
            elif hasattr(item, '__dict__'):
                return item.__dict__
            elif isinstance(item, str):
                logger.warning(f"Item is string: {item}")
                return {}
            else:
                logger.warning(f"Unknown item type: {type(item)}")
                return {}
        except Exception as e:
            logger.error(f"Error normalizing item data: {e}")
            return {}
        
    def _map_to_camera(self, data: Dict) -> Camera:
        """Map ucode API response data to Camera model with safe attribute access"""
        try:
            # Helper function to safely get values
            def safe_get(key, default=None):
                return self._safe_get_attribute(data, key, default)
            
            # Handle UUID fields safely
            def safe_uuid(key):
                value = safe_get(key)
                if value and value != "null":
                    try:
                        return UUID(str(value))
                    except (ValueError, TypeError):
                        logger.debug(f"Invalid UUID for {key}: {value}")
                return None
            
            return Camera(
                guid=safe_uuid('guid'),
                mini_pc_id=safe_uuid('mini_pc_id'),
                rtsp_url=safe_get('rtsp_url'),
                detect_emotion=bool(safe_get('detect_emotion', False)),
                detect_hands=bool(safe_get('detect_hands', False)),
                detect_uniform=bool(safe_get('detect_uniform', False)),
                detect_mask=bool(safe_get('detect_mask', False)),
                voice_detect=bool(safe_get('voice_detect', False)),
                branch_id=safe_get('branch_id'),
                company_id=safe_get('company_id'),
                port=int(safe_get('port', 554)) if safe_get('port') else 554,
                ip_address=safe_get('ip_address'),
                password=safe_get('password'),
                username=safe_get('username')
            )
        except Exception as e:
            logger.error(f"Error mapping data to Camera model: {e}")
            logger.debug(f"Data received: {data}")
            raise

    def _map_from_camera(self, camera: Camera) -> Dict:
        """Map Camera model to ucode API data format"""
        data = {}
        
        # Helper to add non-None values
        def add_if_not_none(key, value):
            if value is not None:
                if isinstance(value, UUID):
                    data[key] = str(value)
                else:
                    data[key] = value
        
        add_if_not_none('guid', camera.guid)
        add_if_not_none('mini_pc_id', camera.mini_pc_id)
        add_if_not_none('rtsp_url', camera.rtsp_url)
        add_if_not_none('branch_id', camera.branch_id)
        add_if_not_none('company_id', camera.company_id)
        add_if_not_none('port', camera.port)
        add_if_not_none('ip_address', camera.ip_address)
        add_if_not_none('password', camera.password)
        add_if_not_none('username', camera.username)
        
        # Boolean fields - always include
        data['detect_emotion'] = bool(camera.detect_emotion)
        data['detect_hands'] = bool(camera.detect_hands)
        data['detect_uniform'] = bool(camera.detect_uniform)
        data['detect_mask'] = bool(camera.detect_mask)
        data['voice_detect'] = bool(camera.voice_detect)
        
        return data

    def get_camera(self, camera_id: UUID) -> Optional[Camera]:
        try:
            logger.debug(f"Getting camera with ID: {camera_id}")
            
            result, response, error = self.ucode_api.items(self.table_slug).get_single(str(camera_id)).exec()
            
            if error:
                logger.error(f"UCode API error: {error}")
                return None
            
            # Extract data using improved method
            data = self._extract_data_from_response(result)
            if data:
                logger.debug(f"Successfully extracted camera data")
                return self._map_to_camera(data)
            
            logger.warning(f"No data found for camera {camera_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting camera {camera_id}: {e}")
            return None

    def list_all(self) -> List[Camera]:
        try:
            logger.debug("Listing all cameras")
            
            result, response, error = self.ucode_api.items(self.table_slug).get_list().exec()
            
            if error:
                logger.error(f"UCode API error: {error}")
                return []
            
            # Extract list data using improved method
            items_data = self._extract_list_from_response(result)
            
            cameras = []
            for item_data in items_data:
                try:
                    camera = self._map_to_camera(item_data)
                    cameras.append(camera)
                except Exception as e:
                    logger.warning(f"Skipping invalid camera data: {e}")
                    logger.debug(f"Invalid data: {item_data}")
                    continue
            
            logger.info(f"Successfully retrieved {len(cameras)} cameras")
            return cameras
            
        except Exception as e:
            logger.error(f"Error listing all cameras: {e}")
            return []

    def list(self, filters: GetListFilter) -> List[Camera]:
        try:
            logger.debug(f"Listing cameras with filters: {filters}")
            
            # Build the query
            builder = self.ucode_api.items(self.table_slug).get_list()
            
            # Apply filters
            if hasattr(filters, 'filters') and filters.filters:
                filter_map = {}
                for filter_item in filters.filters:
                    if filter_item.type == 'eq':
                        filter_map[str(filter_item.column)] = str(filter_item.value)
                    elif filter_item.type == 'like':
                        # UCode might handle like differently
                        filter_map[str(filter_item.column)] = str(filter_item.value)
                
                if filter_map:
                    logger.debug(f"Applying filters: {filter_map}")
                    builder = builder.filter(filter_map)
            
            # Apply pagination
            if hasattr(filters, 'limit') and filters.limit:
                builder = builder.limit(filters.limit)
            
            if hasattr(filters, 'page') and filters.page:
                builder = builder.page(filters.page)
            
            # Execute the query
            result, response, error = builder.exec()

            if error:
                logger.error(f"UCode API error: {error}")
                return []
            
            # Extract list data
            items_data = self._extract_list_from_response(result)
            
            cameras = []
            for item_data in items_data:
                try:
                    camera = self._map_to_camera(item_data)
                    cameras.append(camera)
                except Exception as e:
                    logger.warning(f"Skipping invalid camera data: {e}")
                    continue
            
            logger.info(f"Filtered query returned {len(cameras)} cameras")
            return cameras
            
        except Exception as e:
            logger.error(f"Error listing cameras with filters: {e}")
            return []

    def create_camera(self, camera: Camera) -> Camera:
        try:
            logger.debug(f"Creating camera: {camera.ip_address}:{camera.port}")
            
            data = self._map_from_camera(camera)
            logger.debug(f"Camera data for creation: {data}")
            
            builder = self.ucode_api.items(self.table_slug).create(data)
            result, response, error = builder.exec()
            
            if error:
                logger.error(f"UCode API error: {error}")
                raise Exception(f"Failed to create camera: {error}")
            
            print(result)
            
            # Extract created data
            created_data = self._extract_data_from_response(result)
            if created_data:
                created_camera = self._map_to_camera(created_data)
                logger.info(f"Successfully created camera: {created_camera.guid}")
                return created_camera
            else:
                raise Exception("Failed to create camera - no data returned")
                
        except Exception as e:
            logger.error(f"Error creating camera: {e}")
            raise

    def update_camera(self, camera: Camera) -> Camera:
        try:
            if not camera.guid:
                raise ValueError("Camera GUID is required for update")
                
            logger.debug(f"Updating camera: {camera.guid}")
            
            data = self._map_from_camera(camera)
            
            result, response, error = self.ucode_api.items(self.table_slug).update(data).exec()
            
            if error:
                logger.error(f"UCode API error: {error}")
                raise Exception(f"Failed to update camera: {error}")
            
            # Extract updated data
            updated_data = self._extract_data_from_response(result)
            if updated_data:
                updated_camera = self._map_to_camera(updated_data)
                logger.info(f"Successfully updated camera: {updated_camera.guid}")
                return updated_camera
            else:
                raise Exception("Failed to update camera - no data returned")
                
        except Exception as e:
            logger.error(f"Error updating camera {camera.guid}: {e}")
            raise

    def delete_camera(self, camera_id: UUID) -> bool:
        try:
            logger.debug(f"Deleting camera: {camera_id}")
            
            builder = self.ucode_api.items(self.table_slug).delete()
            builder = builder.filter({'guid': str(camera_id)})
            result, response, error = builder.exec()
            
            if error:
                logger.error(f"UCode API error: {error}")
                return False
            
            # Check if deletion was successful
            success = response and (
                hasattr(response, 'status_code') and response.status_code == 200 or
                hasattr(response, 'status') and response.status == "done"
            )
            
            if success:
                logger.info(f"Successfully deleted camera: {camera_id}")
            else:
                logger.warning(f"Failed to delete camera: {camera_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error deleting camera {camera_id}: {e}")
            return False

    def assign_camera_to_mini_pc(self, camera_id: UUID, mini_pc_id: UUID) -> Camera:
        """Assign a camera to a mini PC by updating the mini_pc_id field"""
        try:
            logger.debug(f"Assigning camera {camera_id} to mini PC {mini_pc_id}")
            
            # Get the existing camera
            camera = self.get_camera(camera_id)
            if not camera:
                raise Exception(f"Camera with ID {camera_id} not found")
            
            # Update the mini_pc_id
            camera.mini_pc_id = mini_pc_id
            
            # Save the updated camera
            updated_camera = self.update_camera(camera)
            
            logger.info(f"Successfully assigned camera {camera_id} to mini PC {mini_pc_id}")
            return updated_camera
            
        except Exception as e:
            logger.error(f"Error assigning camera {camera_id} to mini PC {mini_pc_id}: {e}")
            raise

    def unassign_camera_from_mini_pc(self, camera_id: UUID) -> Camera:
        """Remove camera assignment from Mini PC"""
        try:
            logger.debug(f"Unassigning camera {camera_id} from mini PC")
            
            # Get the existing camera
            camera = self.get_camera(camera_id)
            if not camera:
                raise Exception(f"Camera with ID {camera_id} not found")
            
            # Remove the mini_pc_id
            camera.mini_pc_id = None
            
            # Save the updated camera
            updated_camera = self.update_camera(camera)
            
            logger.info(f"Successfully unassigned camera {camera_id} from mini PC")
            return updated_camera
            
        except Exception as e:
            logger.error(f"Error unassigning camera {camera_id}: {e}")
            raise

    def get_cameras_by_mini_pc(self, mini_pc_id: UUID) -> List[Camera]:
        """Get all cameras assigned to a specific Mini PC"""
        try:
            logger.debug(f"Getting cameras for mini PC: {mini_pc_id}")
            
            filters = GetListFilter(
                filters=[Filter(column="mini_pc_id", type="eq", value=str(mini_pc_id))]
            )
            
            cameras = self.list(filters)
            logger.info(f"Found {len(cameras)} cameras for mini PC {mini_pc_id}")
            return cameras
            
        except Exception as e:
            logger.error(f"Error getting cameras for mini PC {mini_pc_id}: {e}")
            return []