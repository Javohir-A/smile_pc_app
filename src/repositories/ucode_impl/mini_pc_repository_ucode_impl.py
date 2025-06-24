# src/repositories/ucode_impl/mini_pc_repository_ucode_impl.py
from ucode_sdk.config import Config
from ucode_sdk.sdk import new
from uuid import UUID
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from src.models.mini_pc import MiniPC
from src.models.filters import GetListFilter, Filter
from src.repositories.mini_pc_repository import MiniPCRepository
from src.config import AppConfig

logger = logging.getLogger(__name__)

class MiniPCRepositoryUcodeImpl(MiniPCRepository):
    def __init__(self, config: AppConfig):
        self.config = config
        self.ucode_api = new(Config(config.ucode.app_id, config.ucode.base_url, __name__))
        self.table_slug: str = "mini_pc"
        logger.info(f"Initialized UCode Mini PC repository: {config.ucode.app_id}")
    
    def _extract_data_from_response(self, api_response) -> Optional[Dict]:
        """Extract data from standardized ApiResponse"""
        try:
            if not api_response:
                return None
            
            # All responses now use ApiResponse.data_container
            if hasattr(api_response, 'data_container'):
                data_fields = api_response.data_container.get_all_data_fields()
                logger.debug(f"Data fields keys: {list(data_fields.keys()) if isinstance(data_fields, dict) else 'not dict'}")
                
                # Direct data object (create/update responses)
                if isinstance(data_fields, dict):
                    # Check if it's mini PC data directly
                    if any(key in data_fields for key in ['guid', 'device_name', 'mac_address']):
                        return data_fields
                    
                    # Check for nested response (get_single responses)
                    if 'response' in data_fields:
                        response_data = data_fields['response']
                        if isinstance(response_data, dict):
                            return response_data
                        elif isinstance(response_data, list) and len(response_data) > 0:
                            return response_data[0]
                    
                    # Check for nested data
                    if 'data' in data_fields:
                        return data_fields['data']
                
                return data_fields
            
            logger.warning(f"No data_container in response: {type(api_response)}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return None
    
    def _extract_list_from_response(self, api_response) -> List[Dict]:
        """Extract list data from standardized ApiResponse"""
        try:
            if not api_response:
                return []
            
            # All responses now use ApiResponse.data_container
            if hasattr(api_response, 'data_container'):
                data_fields = api_response.data_container.get_all_data_fields()
                logger.debug(f"List data fields: {type(data_fields)}")
                
                # Check for response array
                if isinstance(data_fields, dict) and 'response' in data_fields:
                    response_data = data_fields['response']
                    if isinstance(response_data, list):
                        return response_data
                
                # Check if data_fields is directly a list
                if isinstance(data_fields, list):
                    return data_fields
                
                # Check for nested data
                if isinstance(data_fields, dict) and 'data' in data_fields:
                    nested_data = data_fields['data']
                    if isinstance(nested_data, list):
                        return nested_data
            
            logger.warning(f"Could not extract list from response")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting list: {e}")
            return []
    
    def _map_to_mini_pc(self, data: Dict) -> MiniPC:
        """Map API data to MiniPC model"""
        try:
            def safe_get(key, default=None):
                value = data.get(key, default)
                return value if value not in [None, "", "null"] else default
            
            def safe_uuid(key):
                value = safe_get(key)
                if value:
                    try:
                        return UUID(str(value))
                    except (ValueError, TypeError):
                        logger.debug(f"Invalid UUID for {key}: {value}")
                return None
            
            def safe_int(key, default=None):
                value = safe_get(key)
                try:
                    return int(value) if value else default
                except (ValueError, TypeError):
                    return default
            
            def safe_bool(key, default=True):
                value = safe_get(key)
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value) if value is not None else default
            
            def safe_datetime(key):
                value = safe_get(key)
                if value:
                    try:
                        if isinstance(value, str):
                            # Try parsing common datetime formats
                            from datetime import datetime
                            # Try ISO format first
                            try:
                                return datetime.fromisoformat(value.replace('Z', '+00:00'))
                            except:
                                # Try other common formats
                                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
                                    try:
                                        return datetime.strptime(value, fmt)
                                    except:
                                        continue
                        elif isinstance(value, datetime):
                            return value
                    except Exception as e:
                        logger.debug(f"Could not parse datetime {key}: {value} - {e}")
                return None
            
            mini_pc = MiniPC(
                guid=safe_uuid('guid'),
                device_name=safe_get('device_name'),
                mac_address=safe_get('mac_address'),
                ip_address=safe_get('ip_address'),
                port=safe_int('port'),
                branch_id=safe_uuid('branch_id'),
                company_id=safe_uuid('company_id'),
                is_active=safe_bool('is_active', True),
                created_at=safe_datetime('created_at')
            )
            
            logger.debug(f"Mapped Mini PC: {mini_pc.guid} - {mini_pc.device_name} ({mini_pc.mac_address})")
            return mini_pc
            
        except Exception as e:
            logger.error(f"Error mapping Mini PC data: {e}")
            logger.debug(f"Data: {data}")
            raise
    
    def _map_from_mini_pc(self, mini_pc: MiniPC) -> Dict:
        """Map MiniPC model to API data"""
        data = {}
        
        def add_if_not_none(key, value):
            if value is not None:
                if isinstance(value, UUID):
                    data[key] = str(value)
                elif isinstance(value, datetime):
                    data[key] = value.isoformat()
                else:
                    data[key] = value
        
        # Add all fields
        add_if_not_none('guid', mini_pc.guid)
        add_if_not_none('device_name', mini_pc.device_name)
        add_if_not_none('mac_address', mini_pc.mac_address)
        add_if_not_none('ip_address', mini_pc.ip_address)
        add_if_not_none('port', mini_pc.port)
        add_if_not_none('branch_id', mini_pc.branch_id)
        add_if_not_none('company_id', mini_pc.company_id)
        add_if_not_none('created_at', mini_pc.created_at)
        
        # Boolean field
        data['is_active'] = bool(mini_pc.is_active)
        
        logger.debug(f"Mapped Mini PC to API data: {data}")
        return data

    # Implementation of abstract methods
    
    def list(self, filters: GetListFilter) -> List[MiniPC]:
        """List Mini PCs with filters"""
        try:
            logger.debug(f"Listing Mini PCs with filters")
            
            builder = self.ucode_api.items(self.table_slug).get_list()
            
            # Apply filters
            if hasattr(filters, 'filters') and filters.filters:
                filter_map = {}
                for filter_item in filters.filters:
                    if filter_item.type == 'eq':
                        filter_map[filter_item.column] = filter_item.value
                    elif filter_item.type == 'like':
                        filter_map[filter_item.column] = filter_item.value
                    elif filter_item.type == 'ne':
                        # For not equal, you might need to handle differently based on your API
                        filter_map[f"{filter_item.column}__ne"] = filter_item.value
                
                if filter_map:
                    logger.debug(f"Applying filters: {filter_map}")
                    builder = builder.filter(filter_map)
            
            # Apply pagination
            if hasattr(filters, 'limit') and filters.limit:
                builder = builder.limit(filters.limit)
            
            if hasattr(filters, 'page') and filters.page:
                builder = builder.page(filters.page)
            
            api_response, response, error = builder.exec()
            
            if error:
                logger.error(f"API error: {error}")
                return []
            
            items_data = self._extract_list_from_response(api_response)
            
            mini_pcs = []
            for item_data in items_data:
                try:
                    mini_pc = self._map_to_mini_pc(item_data)
                    mini_pcs.append(mini_pc)
                except Exception as e:
                    logger.warning(f"Skipping invalid Mini PC: {e}")
                    continue
            
            logger.info(f"Filtered query returned {len(mini_pcs)} Mini PCs")
            return mini_pcs
            
        except Exception as e:
            logger.error(f"Error listing Mini PCs with filters: {e}")
            return []
    
    def get_by_id(self, mini_pc_id: UUID) -> Optional[MiniPC]:
        """Get Mini PC by ID"""
        try:
            logger.debug(f"Getting Mini PC: {mini_pc_id}")
            
            api_response, response, error = self.ucode_api.items(self.table_slug).get_single(str(mini_pc_id)).exec()
            
            if error:
                logger.error(f"API error: {error}")
                return None
            
            data = self._extract_data_from_response(api_response)
            if data:
                return self._map_to_mini_pc(data)
            
            logger.warning(f"No data found for Mini PC: {mini_pc_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting Mini PC {mini_pc_id}: {e}")
            return None
    
    def get_by_mac_address(self, mac_address: str) -> Optional[MiniPC]:
        """Get Mini PC by MAC address"""
        try:
            logger.debug(f"Getting Mini PC by MAC: {mac_address}")
            
            # Use filter to find by MAC address
            filters = GetListFilter(
                filters=[Filter(column="mac_address", type="eq", value=mac_address)],
                limit=1,
                page=1
            )
            
            mini_pcs = self.list(filters)
            
            if mini_pcs:
                logger.info(f"Found Mini PC by MAC {mac_address}: {mini_pcs[0].guid}")
                return mini_pcs[0]
            else:
                logger.info(f"No Mini PC found with MAC: {mac_address}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting Mini PC by MAC {mac_address}: {e}")
            return None
    
    def create(self, mini_pc: MiniPC) -> MiniPC:
        """Create new Mini PC"""
        try:
            logger.debug(f"Creating Mini PC: {mini_pc.device_name} ({mini_pc.mac_address})")
            
            data = self._map_from_mini_pc(mini_pc)
            
            # Set created_at if not provided
            if not mini_pc.created_at:
                data['created_at'] = datetime.now().isoformat()
            
            api_response, response, error = self.ucode_api.items(self.table_slug).create(data).exec()
            
            if error:
                logger.error(f"API error: {error}")
                raise Exception(f"Failed to create Mini PC: {error}")
            
            created_data = self._extract_data_from_response(api_response)
            if created_data:
                created_mini_pc = self._map_to_mini_pc(created_data)
                logger.info(f"Successfully created Mini PC: {created_mini_pc.guid}")
                return created_mini_pc
            else:
                raise Exception("No data returned from create operation")
                
        except Exception as e:
            logger.error(f"Error creating Mini PC: {e}")
            raise
    
    def update(self, mini_pc: MiniPC) -> MiniPC:
        """Update existing Mini PC"""
        try:
            if not mini_pc.guid:
                raise ValueError("Mini PC GUID is required for update")
            
            logger.debug(f"Updating Mini PC: {mini_pc.guid}")
            
            data = self._map_from_mini_pc(mini_pc)
            
            api_response, response, error = (
                self.ucode_api.items(self.table_slug)
                .update(data)
                .filter({'guid': str(mini_pc.guid)})
                .exec()
            )
            
            if error:
                logger.error(f"API error: {error}")
                raise Exception(f"Failed to update Mini PC: {error}")
            
            updated_data = self._extract_data_from_response(api_response)
            if updated_data:
                updated_mini_pc = self._map_to_mini_pc(updated_data)
                logger.info(f"Successfully updated Mini PC: {updated_mini_pc.guid}")
                return updated_mini_pc
            else:
                raise Exception("No data returned from update operation")
                
        except Exception as e:
            logger.error(f"Error updating Mini PC: {e}")
            raise
    
    def delete(self, mini_pc_id: UUID) -> bool:
        """Delete Mini PC"""
        try:
            logger.debug(f"Deleting Mini PC: {mini_pc_id}")
            
            api_response, response, error = (
                self.ucode_api.items(self.table_slug)
                .delete()
                .single(str(mini_pc_id))
                .exec()
            )
            
            if error:
                logger.error(f"API error: {error}")
                return False
            
            # Check response status
            success = (
                response and response.status == "done" or
                (api_response and hasattr(api_response, 'status') and api_response.status in ['SUCCESS', 'OK'])
            )
            
            if success:
                logger.info(f"Successfully deleted Mini PC: {mini_pc_id}")
            else:
                logger.warning(f"Delete operation may have failed: {mini_pc_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error deleting Mini PC {mini_pc_id}: {e}")
            return False

    # Additional helper methods
    
    def get_active_mini_pcs(self) -> List[MiniPC]:
        """Get all active Mini PCs"""
        try:
            filters = GetListFilter(
                filters=[Filter(column="is_active", type="eq", value="true")]
            )
            return self.list(filters)
            
        except Exception as e:
            logger.error(f"Error getting active Mini PCs: {e}")
            return []
    
    def get_mini_pcs_by_branch(self, branch_id: UUID) -> List[MiniPC]:
        """Get Mini PCs by branch ID"""
        try:
            filters = GetListFilter(
                filters=[Filter(column="branch_id", type="eq", value=str(branch_id))]
            )
            return self.list(filters)
            
        except Exception as e:
            logger.error(f"Error getting Mini PCs for branch: {e}")
            return []
    
    def get_mini_pcs_by_company(self, company_id: UUID) -> List[MiniPC]:
        """Get Mini PCs by company ID"""
        try:
            filters = GetListFilter(
                filters=[Filter(column="company_id", type="eq", value=str(company_id))]
            )
            return self.list(filters)
            
        except Exception as e:
            logger.error(f"Error getting Mini PCs for company: {e}")
            return []
    
    def activate_mini_pc(self, mini_pc_id: UUID) -> bool:
        """Activate a Mini PC"""
        try:
            mini_pc = self.get_by_id(mini_pc_id)
            if not mini_pc:
                raise Exception(f"Mini PC not found: {mini_pc_id}")
            
            mini_pc.is_active = True
            updated_mini_pc = self.update(mini_pc)
            return updated_mini_pc.is_active
            
        except Exception as e:
            logger.error(f"Error activating Mini PC: {e}")
            return False
    
    def deactivate_mini_pc(self, mini_pc_id: UUID) -> bool:
        """Deactivate a Mini PC"""
        try:
            mini_pc = self.get_by_id(mini_pc_id)
            if not mini_pc:
                raise Exception(f"Mini PC not found: {mini_pc_id}")
            
            mini_pc.is_active = False
            updated_mini_pc = self.update(mini_pc)
            return not updated_mini_pc.is_active
            
        except Exception as e:
            logger.error(f"Error deactivating Mini PC: {e}")
            return False