# debug_ucode_responses.py - Debug UCode API response structures
import sys
import os
import logging
import json
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ucode_sdk.sdk import new, Config

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UCodeResponseDebugger:
    """Debug UCode API responses to understand data structures"""
    
    def __init__(self):
        self.app_id = "P-QlWnuJCdfy32dsQoIjXHQNScO7DR2TdL"
        self.base_url = "https://api.client.u-code.io"
        self.ucode_api = new(Config(self.app_id, self.base_url, __name__))
        self.table_slug = "camera_"
    
    def debug_response_structure(self, obj, name="object", depth=0, max_depth=3):
        """Recursively debug response structure"""
        indent = "  " * depth
        
        if depth > max_depth:
            print(f"{indent}{name}: <max depth reached>")
            return
        
        print(f"{indent}{name}: {type(obj).__name__}")
        
        if hasattr(obj, '__dict__'):
            for attr_name, attr_value in obj.__dict__.items():
                if not attr_name.startswith('_'):
                    self.debug_response_structure(attr_value, f"{name}.{attr_name}", depth + 1, max_depth)
        
        elif isinstance(obj, dict):
            for key, value in list(obj.items())[:5]:  # Limit to first 5 items
                self.debug_response_structure(value, f"{name}['{key}']", depth + 1, max_depth)
        
        elif isinstance(obj, list):
            print(f"{indent}  -> list with {len(obj)} items")
            if obj:
                self.debug_response_structure(obj[0], f"{name}[0]", depth + 1, max_depth)
        
        elif isinstance(obj, str):
            preview = obj[:100] + "..." if len(obj) > 100 else obj
            print(f"{indent}  -> str: '{preview}'")
        
        else:
            print(f"{indent}  -> value: {str(obj)[:100]}")
    
    def test_list_response(self):
        """Test and debug list response structure"""
        try:
            print("🔍 Testing LIST response structure")
            print("=" * 50)
            
            result, response, error = self.ucode_api.items(self.table_slug).get_list().limit(1).exec()
            
            print("📋 Result structure:")
            self.debug_response_structure(result, "result")
            
            print("\n📡 Response structure:")
            self.debug_response_structure(response, "response")
            
            if error:
                print(f"\n❌ Error: {error}")
            
            # Try to extract data using different methods
            print("\n🔧 Data extraction attempts:")
            
            # Method 1: Direct data
            if hasattr(result, 'data'):
                print(f"result.data type: {type(result.data)}")
                if hasattr(result.data, 'data'):
                    print(f"result.data.data type: {type(result.data.data)}")
            
            # Method 2: Data container
            if hasattr(result, 'data_container'):
                print(f"result.data_container type: {type(result.data_container)}")
                if hasattr(result.data_container, 'data'):
                    container_data = result.data_container.data
                    print(f"result.data_container.data type: {type(container_data)}")
                    if isinstance(container_data, dict):
                        print(f"Keys in data_container.data: {list(container_data.keys())}")
                        if 'response' in container_data:
                            print(f"response type: {type(container_data['response'])}")
            
            return result, response, error
            
        except Exception as e:
            print(f"❌ List test failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, e
    
    def test_single_response(self, camera_id):
        """Test and debug single item response structure"""
        try:
            print(f"\n🎯 Testing SINGLE response structure for ID: {camera_id}")
            print("=" * 50)
            
            result, response, error = self.ucode_api.items(self.table_slug).get_single(str(camera_id)).exec()
            
            print("📋 Result structure:")
            self.debug_response_structure(result, "result")
            
            print("\n📡 Response structure:")
            self.debug_response_structure(response, "response")
            
            if error:
                print(f"\n❌ Error: {error}")
            
            return result, response, error
            
        except Exception as e:
            print(f"❌ Single test failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, e
    
    def test_filtered_response(self):
        """Test and debug filtered response structure"""
        try:
            print(f"\n🔍 Testing FILTERED response structure")
            print("=" * 50)
            
            result, response, error = self.ucode_api.items(self.table_slug).get_list().filter({
                "mini_pc_mac": "f0:2f:74:9f:62:29"
            }).limit(1).exec()
            
            print("📋 Filtered result structure:")
            self.debug_response_structure(result, "result")
            
            if error:
                print(f"\n❌ Error: {error}")
            
            return result, response, error
            
        except Exception as e:
            print(f"❌ Filtered test failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, e
    
    def test_create_response(self):
        """Test and debug create response structure"""
        try:
            print(f"\n➕ Testing CREATE response structure")
            print("=" * 50)
            
            test_data = {
                "ip_address": "192.168.100.99",
                "port": 554,
                "username": "debug_user",
                "password": "debug_pass",
                "detect_emotion": True
            }
            
            result, response, error = self.ucode_api.items(self.table_slug).create(test_data).exec()
            
            print("📋 Create result structure:")
            self.debug_response_structure(result, "result")
            
            print("\n📡 Create response structure:")
            self.debug_response_structure(response, "response")
            
            if error:
                print(f"\n❌ Error: {error}")
            
            return result, response, error
            
        except Exception as e:
            print(f"❌ Create test failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, e
    
    def run_comprehensive_debug(self):
        """Run all debug tests"""
        print("🐛 UCode API Response Structure Debug")
        print("=" * 60)
        
        # Test 1: List response
        list_result, list_response, list_error = self.test_list_response()
        
        # Test 2: Single item response (if we got any items from list)
        camera_id = None
        if list_result and not list_error:
            try:
                # Try to extract a camera ID for single test
                if hasattr(list_result, 'data_container'):
                    container_data = list_result.data_container.data
                    if isinstance(container_data, dict) and 'response' in container_data:
                        response_list = container_data['response']
                        if response_list and len(response_list) > 0:
                            first_item = response_list[0]
                            if isinstance(first_item, dict) and 'guid' in first_item:
                                camera_id = first_item['guid']
                            elif hasattr(first_item, 'guid'):
                                camera_id = first_item.guid
            except:
                pass
        
        if camera_id:
            self.test_single_response(camera_id)
        else:
            print("\n⚠️  Could not extract camera ID for single test")
        
        # Test 3: Filtered response
        self.test_filtered_response()
        
        # Test 4: Create response (be careful with this)
        # Uncomment if you want to test create operations
        # self.test_create_response()
        
        print("\n✅ Debug session completed!")
        print("Use this information to fix the repository implementation.")

def main():
    """Main debug function"""
    load_dotenv('.env.development')
    
    debugger = UCodeResponseDebugger()
    debugger.run_comprehensive_debug()

if __name__ == "__main__":
    main()