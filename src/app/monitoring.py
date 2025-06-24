import psutil
from datetime import datetime
import logging
import threading
import time

class ResourceMonitor:
    """Lightweight resource monitor for your multi-camera application"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.stats = []
        self.start_time = None
        
    def get_current_usage(self):
        """Get current resource usage"""
        try:
            # Get process stats
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            thread_count = self.process.num_threads()
            
            # Get system stats
            system_cpu = psutil.cpu_percent()
            system_memory = psutil.virtual_memory()
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'process_cpu_percent': cpu_percent,
                'process_memory_mb': memory_mb,
                'process_threads': thread_count,
                'system_cpu_percent': system_cpu,
                'system_memory_percent': system_memory.percent,
                'system_memory_available_gb': system_memory.available / 1024 / 1024 / 1024
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting resource stats: {e}")
            return None
    
    def start_monitoring(self, interval=10):
        """Start monitoring with specified interval (seconds)"""
        self.monitoring = True
        self.start_time = datetime.now()
        
        def monitor_loop():
            while self.monitoring:
                stats = self.get_current_usage()
                if stats:
                    self.stats.append(stats)
                    
                    # Log every few samples
                    if len(self.stats) % 6 == 0:  # Every minute if interval=10s
                        self.log_current_stats(stats)
                    
                    # Keep only last 100 samples to avoid memory growth
                    if len(self.stats) > 100:
                        self.stats = self.stats[-100:]
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring and print summary"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        
        # Print summary
        self.print_summary()
        logging.info("Resource monitoring stopped")
    
    def log_current_stats(self, stats):
        """Log current stats"""
        logging.info(
            f"📊 Resources: CPU {stats['process_cpu_percent']:.1f}% | "
            f"Memory {stats['process_memory_mb']:.1f}MB | "
            f"Threads {stats['process_threads']} | "
            f"System CPU {stats['system_cpu_percent']:.1f}%"
        )
    
    def print_summary(self):
        """Print resource usage summary"""
        if not self.stats:
            print("No monitoring data available")
            return
        
        # Calculate averages and peaks
        cpu_values = [s['process_cpu_percent'] for s in self.stats]
        memory_values = [s['process_memory_mb'] for s in self.stats]
        thread_values = [s['process_threads'] for s in self.stats]
        
        duration = datetime.now() - self.start_time if self.start_time else None
        
        print(f"\n{'='*60}")
        print(f"🖥️  RESOURCE USAGE SUMMARY")
        print(f"{'='*60}")
        print(f"Monitoring Duration: {duration}")
        print(f"Samples Collected: {len(self.stats)}")
        print(f"\n📈 CPU Usage:")
        print(f"  Average: {sum(cpu_values)/len(cpu_values):.1f}%")
        print(f"  Peak: {max(cpu_values):.1f}%")
        print(f"  Minimum: {min(cpu_values):.1f}%")
        
        print(f"\n💾 Memory Usage:")
        print(f"  Average: {sum(memory_values)/len(memory_values):.1f} MB")
        print(f"  Peak: {max(memory_values):.1f} MB")
        print(f"  Minimum: {min(memory_values):.1f} MB")
        
        print(f"\n🧵 Thread Count:")
        print(f"  Average: {sum(thread_values)/len(thread_values):.1f}")
        print(f"  Peak: {max(thread_values)}")
        print(f"  Minimum: {min(thread_values)}")
        
        # Current system info
        current = self.stats[-1]
        print(f"\n🖥️  Current System Status:")
        print(f"  System CPU: {current['system_cpu_percent']:.1f}%")
        print(f"  System Memory: {current['system_memory_percent']:.1f}%")
        print(f"  Available Memory: {current['system_memory_available_gb']:.1f} GB")
        print(f"{'='*60}")
    
    def get_realtime_display(self):
        """Get current stats for real-time display"""
        stats = self.get_current_usage()
        if not stats:
            return "Resource monitoring unavailable"
        
        return (f"CPU: {stats['process_cpu_percent']:.1f}% | "
                f"RAM: {stats['process_memory_mb']:.0f}MB | "
                f"Threads: {stats['process_threads']} | "
                f"Sys CPU: {stats['system_cpu_percent']:.1f}%")
