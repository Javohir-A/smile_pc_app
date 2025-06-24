import subprocess
import logging


def get_system_mac_address() -> str:
    """Get the MAC address of the primary network interface"""
    try:
        result = subprocess.check_output(
            "ip link show | grep 'link/ether' | awk '{print $2}' | head -1", 
            shell=True
        ).decode().strip()
        
        if result:
            return result
        else:
            # Fallback method
            import uuid
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                            for elements in range(0,2*6,2)][::-1])
            return mac
            
    except Exception as e:
        logging.error(f"Error getting MAC address: {e}")
        # Last resort fallback
        import uuid
        mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                        for elements in range(0,2*6,2)][::-1])
        return mac
