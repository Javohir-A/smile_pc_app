# test_ucode_camera_functionality.py
import sys
import os
import logging
from datetime import datetime
from uuid import UUID, uuid4
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.di.dependencies import DependencyContainer, initialize_dependencies
from src.config import AppConfig
from src.models.filters import GetListFilter, Filter
from src.models.camera import Camera

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UCodeCameraRepositoryTester:
    """Comprehensive tester for UCode Camera Repository functionality"""
    
    def __init__(self):
        self.config = None
        self.container = None
        self.camera_usecase = None
        self.test_results = {
            'connection': False,
            'list_all': False,
            'list_filtered': False,
            'get_single': False,
            'create': False,
            'update': False,
            'delete': False,
            'mini_pc_operations': False
        }
        
    def setup(self):
        """Setup test environment"""
        try:
            logger.info("🔧 Setting up test environment...")
            
            # Load environment variables
            load_dotenv('.env.development')
            
            # Create config
            self.config = AppConfig.from_env()
            
            # Set UCode configuration
            self.config.ucode.app_id = "P-QlWnuJCdfy32dsQoIjXHQNScO7DR2TdL"
            self.config.ucode.base_url = "https://api.client.u-code.io"
            
            logger.info(f"✅ UCode API configured: {self.config.ucode.app_id}")
            
            # Initialize dependencies
            self.container = initialize_dependencies(self.config)
            self.camera_usecase = self.container.get_camera_usecase()
            
            logger.info("✅ Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False
    
    def test_connection(self):
        """Test basic UCode API connection"""
        try:
            logger.info("\n🔍 Testing UCode API connection...")
            
            # Try to get a list of cameras (small limit for connection test)
            filters = GetListFilter(limit=1, page=1)
            cameras = self.camera_usecase.list_cameras(filters)
            
            logger.info(f"✅ Connection successful - Retrieved {len(cameras)} cameras")
            self.test_results['connection'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def test_list_all_cameras(self):
        """Test listing all cameras"""
        try:
            logger.info("\n📋 Testing list all cameras...")
            
            cameras = self.camera_usecase.list_cameras(GetListFilter(limit=50))
            
            logger.info(f"✅ Found {len(cameras)} cameras total")
            
            # Show details of first few cameras
            for i, camera in enumerate(cameras[:3]):
                logger.info(f"  Camera {i+1}: {camera.ip_address}:{camera.port} "
                           f"(GUID: {camera.guid})")
                if hasattr(camera, 'mini_pc_id') and camera.mini_pc_id:
                    logger.info(f"    → Assigned to Mini PC: {camera.mini_pc_id}")
            
            self.test_results['list_all'] = True
            return cameras
            
        except Exception as e:
            logger.error(f"❌ List all cameras test failed: {e}")
            return []
    
    def test_filtered_list(self):
        """Test filtered camera listing"""
        try:
            logger.info("\n🔍 Testing filtered camera listing...")
            
            # Test 1: Filter by Mini PC MAC address
            mac_filter = GetListFilter(
                filters=[Filter(column="mini_pc_mac", type="eq", value="f0:2f:74:9f:62:29")]
            )
            cameras_by_mac = self.camera_usecase.list_cameras(mac_filter)
            logger.info(f"✅ Cameras for MAC f0:2f:74:9f:62:29: {len(cameras_by_mac)}")
            
            # Test 2: Filter by emotion detection capability
            emotion_filter = GetListFilter(
                filters=[Filter(column="detect_emotion", type="eq", value="true")]
            )
            emotion_cameras = self.camera_usecase.list_cameras(emotion_filter)
            logger.info(f"✅ Emotion detection cameras: {len(emotion_cameras)}")
            
            # Test 3: Filter by IP address pattern (if you have multiple cameras)
            ip_filter = GetListFilter(
                filters=[Filter(column="ip_address", type="like", value="192.168")]
            )
            ip_cameras = self.camera_usecase.list_cameras(ip_filter)
            logger.info(f"✅ Cameras with 192.168.x.x IP: {len(ip_cameras)}")
            
            self.test_results['list_filtered'] = True
            return cameras_by_mac
            
        except Exception as e:
            logger.error(f"❌ Filtered list test failed: {e}")
            return []
    
    def test_get_single_camera(self, camera_guid: UUID):
        """Test getting a single camera by GUID"""
        try:
            logger.info(f"\n🎯 Testing get single camera: {camera_guid}")
            
            camera = self.camera_usecase.get_camera(camera_guid)
            
            if camera:
                logger.info(f"✅ Retrieved camera: {camera.ip_address}:{camera.port}")
                logger.info(f"   Capabilities: Emotion={camera.detect_emotion}, "
                           f"Hands={camera.detect_hands}, Voice={camera.voice_detect}")
                
                if hasattr(camera, 'mini_pc_id') and camera.mini_pc_id:
                    logger.info(f"   Mini PC: {camera.mini_pc_id}")
                
                self.test_results['get_single'] = True
                return camera
            else:
                logger.warning("⚠️  Camera not found")
                return None
                
        except Exception as e:
            logger.error(f"❌ Get single camera test failed: {e}")
            return None
    
    def test_create_camera(self):
        """Test creating a new camera"""
        try:
            logger.info("\n➕ Testing camera creation...")
            
            # Create a test camera
            test_camera = Camera(
                ip_address="192.168.100.99",
                port=554,
                username="test_user",
                password="test_pass",
                detect_emotion=True,
                detect_hands=False,
                detect_uniform=True,
                detect_mask=False,
                voice_detect=False,
                # branch_id=,  # Use appropriate branch ID
                # company_id=1  # Use appropriate company ID
            )
            
            # print(tes)
            created_camera = self.camera_usecase.create_camera(test_camera)
            logger.info(f"✅ Created camera with GUID: {created_camera.guid}")
            logger.info(f"   IP: {created_camera.ip_address}:{created_camera.port}")
            logger.info(f"   RTSP URL: {created_camera.generate_rtsp_url()}")
            
            self.test_results['create'] = True
            return created_camera
            
        except Exception as e:
            logger.error(f"❌ Create camera test failed: {e}")
            return None
    
    def test_update_camera(self, camera: Camera):
        """Test updating an existing camera"""
        try:
            logger.info(f"\n✏️  Testing camera update: {camera.guid}")
            
            # Update some fields
            camera.detect_hands = True  # Toggle hands detection
            camera.voice_detect = True  # Enable voice detection
            camera.password = ""
            
            updated_camera = self.camera_usecase.update_camera(camera)
            
            logger.info(f"✅ Updated camera: {updated_camera.guid}")
            logger.info(f"   New capabilities: Hands={updated_camera.detect_hands}, "
                       f"Voice={updated_camera.voice_detect}")
            
            self.test_results['update'] = True
            return updated_camera
            
        except Exception as e:
            logger.error(f"❌ Update camera test failed: {e}")
            return None
    
    def test_mini_pc_operations(self, camera: Camera):
        """Test Mini PC assignment operations"""
        try:
            logger.info(f"\n🖥️  Testing Mini PC operations...")
            
            # Generate a test Mini PC UUID
            test_mini_pc_id = uuid4()
            
            # Test assignment
            assigned_camera = self.camera_usecase.assign_camera_to_mini_pc(
                camera.guid, test_mini_pc_id
            )
            
            logger.info(f"✅ Assigned camera to Mini PC: {test_mini_pc_id}")
            logger.info(f"   Camera Mini PC ID: {assigned_camera.mini_pc_id}")
            
            # Test getting cameras by Mini PC
            mini_pc_cameras = self.camera_usecase.get_cameras_by_mini_pc(test_mini_pc_id)
            logger.info(f"✅ Found {len(mini_pc_cameras)} cameras for Mini PC")
            
            # Test unassignment
            unassigned_camera = self.camera_usecase.unassign_camera_from_mini_pc(camera.guid)
            logger.info(f"✅ Unassigned camera from Mini PC")
            logger.info(f"   Camera Mini PC ID: {unassigned_camera.mini_pc_id}")
            
            self.test_results['mini_pc_operations'] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Mini PC operations test failed: {e}")
            return False
    
    def test_delete_camera(self, camera_guid: UUID):
        """Test deleting a camera"""
        try:
            logger.info(f"\n🗑️  Testing camera deletion: {camera_guid}")
            
            success = self.camera_usecase.delete_camera(camera_guid)
            
            if success:
                logger.info(f"✅ Successfully deleted camera: {camera_guid}")
                
                # Verify deletion by trying to get the camera
                deleted_camera = self.camera_usecase.get_camera(camera_guid)
                if deleted_camera is None:
                    logger.info("✅ Deletion verified - camera no longer exists")
                else:
                    logger.warning("⚠️  Camera still exists after deletion")
                
                self.test_results['delete'] = True
                return True
            else:
                logger.error("❌ Failed to delete camera")
                return False
                
        except Exception as e:
            logger.error(f"❌ Delete camera test failed: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        logger.info("🚀 Starting comprehensive UCode Camera Repository test")
        logger.info("=" * 60)
        
        # Setup
        if not self.setup():
            logger.error("❌ Setup failed - aborting tests")
            return False
        
        # Test 1: Connection
        if not self.test_connection():
            logger.error("❌ Connection failed - aborting remaining tests")
            return False
        
        # Test 2: List all cameras
        # all_cameras = self.test_list_all_cameras()
        
        # Test 3: Filtered listing
        # filtered_cameras = self.test_filtered_list()
        
        # Test 4: Get single camera (use first camera from list)
        # test_camera = None
        # if all_cameras:
        #     test_camera = self.test_get_single_camera(all_cameras[0].guid)
        
        # Test 5: Create new camera
        created_camera = self.test_create_camera()
        
        # Test 6: Update camera (use created camera)
        # updated_camera = None
        # if created_camera:
        #     updated_camera = self.test_update_camera(created_camera)
        
        # Test 7: Mini PC operations (use created camera)
        # if created_camera:
        #     self.test_mini_pc_operations(created_camera)
        
        # Test 8: Delete camera (use created camera)
        # if created_camera:
        #     self.test_delete_camera(created_camera.guid)
        
        # Print test results summary
        self.print_test_summary()
        
        return all(self.test_results.values())
    
    def print_test_summary(self):
        """Print a summary of all test results"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:20s}: {status}")
        
        logger.info("-" * 60)
        logger.info(f"TOTAL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! UCode integration is working perfectly.")
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Check logs above.")
        
        logger.info("=" * 60)


def main():
    """Main test execution function"""
    print("UCode Camera Repository Integration Test")
    print("=" * 50)
    
    tester = UCodeCameraRepositoryTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)