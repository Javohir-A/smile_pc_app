import psutil
import datetime
import logging
import threading
import time


def monitor_external_process(process_name="python", duration=300):
    """Monitor external process by name"""
    print(f"Looking for process containing: {process_name}")
    
    # Find your application process
    target_process = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if process_name.lower() in cmdline.lower() and 'main.py' in cmdline:
                target_process = proc
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not target_process:
        print(f"❌ Could not find process containing '{process_name}' with 'main.py'")
        print("Available Python processes:")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])[:100]
                    print(f"  PID {proc.info['pid']}: {cmdline}")
            except:
                pass
        return
    
    print(f"✅ Found target process PID: {target_process.info['pid']}")
    print(f"Command: {' '.join(target_process.info['cmdline'])[:100]}...")
    
    # Monitor the process
    class ExternalProcessMonitor:
        def __init__(self, process):
            self.process = psutil.Process(process.info['pid'])
            self.stats = []
            
        def monitor(self, duration):
            print(f"\n🔍 Monitoring for {duration} seconds...")
            start_time = time.time()
            
            try:
                while time.time() - start_time < duration:
                    try:
                        cpu = self.process.cpu_percent(interval=1)
                        memory = self.process.memory_info().rss / 1024 / 1024
                        threads = self.process.num_threads()
                        
                        stats = {
                            'cpu': cpu,
                            'memory_mb': memory,
                            'threads': threads,
                            'timestamp': datetime.now()
                        }
                        self.stats.append(stats)
                        
                        # Print live stats every 10 seconds
                        if len(self.stats) % 10 == 0:
                            print(f"⏱️  {datetime.now().strftime('%H:%M:%S')} - "
                                  f"CPU: {cpu:.1f}% | Memory: {memory:.1f}MB | Threads: {threads}")
                        
                    except psutil.NoSuchProcess:
                        print("❌ Process terminated")
                        break
                    except KeyboardInterrupt:
                        print("\n⏹️  Monitoring stopped by user")
                        break
                        
            finally:
                self.print_summary()
        
        def print_summary(self):
            if not self.stats:
                return
                
            cpu_values = [s['cpu'] for s in self.stats]
            memory_values = [s['memory_mb'] for s in self.stats]
            thread_values = [s['threads'] for s in self.stats]
            
            print(f"\n📊 MONITORING RESULTS")
            print(f"{'='*50}")
            print(f"Duration: {len(self.stats)} seconds")
            print(f"CPU Usage - Avg: {sum(cpu_values)/len(cpu_values):.1f}% | Peak: {max(cpu_values):.1f}%")
            print(f"Memory Usage - Avg: {sum(memory_values)/len(memory_values):.1f}MB | Peak: {max(memory_values):.1f}MB")
            print(f"Thread Count - Avg: {sum(thread_values)/len(thread_values):.1f} | Peak: {max(thread_values)}")
            print(f"{'='*50}")
    
    monitor = ExternalProcessMonitor(target_process)
    monitor.monitor(duration)


# QUICK MONITORING COMMANDS
# Use these for quick checks

def quick_system_check():
    """Quick system resource check"""
    print(f"\n🖥️  QUICK SYSTEM CHECK - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    
    # CPU info
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU: {cpu_percent:.1f}% usage ({cpu_count} cores)")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent:.1f}% used ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
    
    # Disk info
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.percent:.1f}% used ({disk.used/1024/1024/1024:.1f}GB / {disk.total/1024/1024/1024:.1f}GB)")
    
    # Find Python processes
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except:
            pass
    
    print(f"\n🐍 Python Processes ({len(python_processes)} found):")
    for proc in python_processes[:5]:  # Show top 5
        print(f"  PID {proc['pid']}: CPU {proc['cpu_percent']:.1f}% | Memory {proc['memory_percent']:.1f}%")
    
    print(f"{'='*50}")


def monitor_by_pid(pid, duration=60):
    """Monitor specific process by PID"""
    try:
        process = psutil.Process(pid)
        print(f"Monitoring PID {pid}: {process.name()}")
        
        for i in range(duration):
            try:
                cpu = process.cpu_percent()
                memory = process.memory_info().rss / 1024 / 1024
                threads = process.num_threads()
                
                print(f"\r{i+1:3d}s - CPU: {cpu:5.1f}% | Memory: {memory:6.1f}MB | Threads: {threads:2d}", end='', flush=True)
                time.sleep(1)
                
            except psutil.NoSuchProcess:
                print(f"\n❌ Process {pid} terminated")
                break
            except KeyboardInterrupt:
                print(f"\n⏹️  Monitoring stopped")
                break
        
        print()  # New line after monitoring
        
    except psutil.NoSuchProcess:
        print(f"❌ Process with PID {pid} not found")
    except Exception as e:
        print(f"❌ Error monitoring PID {pid}: {e}")


# USAGE EXAMPLES AND INSTRUCTIONS

def print_usage_instructions():
    """Print instructions for using the monitoring tools"""
    print(f"""
🔍 RESOURCE MONITORING USAGE GUIDE
{'='*60}

1. QUICK SYSTEM CHECK:
   python -c "from monitoring import quick_system_check; quick_system_check()"

2. MONITOR YOUR RUNNING APPLICATION:
   # First, find your process:
   python -c "from monitoring import quick_system_check; quick_system_check()"
   
   # Then monitor by name:
   python -c "from monitoring import monitor_external_process; monitor_external_process('main.py', 300)"

3. MONITOR BY PROCESS ID:
   python -c "from monitoring import monitor_by_pid; monitor_by_pid(12345, 120)"

4. INTEGRATE INTO YOUR APPLICATION:
   # Add this to your main.py:
   from monitoring import SimpleResourceMonitor
   
   # In your main function:
   monitor = SimpleResourceMonitor()
   monitor.start_monitoring()
   
   # Your application code here
   
   # Before exit:
   monitor.stop_monitoring()

5. DETAILED MONITORING (save to run_monitor.py):
   from monitoring import ResourceMonitor
   monitor = ResourceMonitor()
   monitor.start_monitoring()
   monitor.print_live_stats(300)  # 5 minutes
   monitor.stop_monitoring()

6. CONTINUOUS MONITORING:
   # Run this in a separate terminal while your app runs:
   python -c "
   from monitoring import monitor_external_process
   import sys
   
   print('Looking for your multi-camera application...')
   monitor_external_process('main.py', 3600)  # Monitor for 1 hour
   "

📊 WHAT TO LOOK FOR:
{'─'*30}
• CPU Usage: Should be < 80% average for stable operation
• Memory Usage: Monitor for memory leaks (constantly increasing)
• Thread Count: Should be stable at 8 threads for 2 cameras
• Peak values: Temporary spikes are normal, sustained high usage indicates issues

⚠️  WARNING SIGNS:
{'─'*20}
• CPU consistently > 90%
• Memory constantly increasing
• Thread count growing over time
• System becoming unresponsive

💡 TIPS:
{'─'*10}
• Run monitoring in a separate terminal
• Monitor for at least 10-15 minutes to see patterns
• Compare resource usage with/without cameras connected
• Save monitoring logs for performance optimization

Example command to start monitoring:
python -c "
import subprocess, sys, time
print('Starting your application monitoring...')
print('Press Ctrl+C to stop')

# This will find and monitor your running application
from monitoring import monitor_external_process
try:
    monitor_external_process('python', 600)  # 10 minutes
except KeyboardInterrupt:
    print('Monitoring stopped by user')
"
""")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            quick_system_check()
        elif command == "monitor" and len(sys.argv) > 2:
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 300
            monitor_external_process(sys.argv[2], duration)
        elif command == "pid" and len(sys.argv) > 2:
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            monitor_by_pid(int(sys.argv[2]), duration)
        elif command == "help":
            print_usage_instructions()
        else:
            print("❌ Unknown command")
            print_usage_instructions()
    else:
        print("🔍 Resource Monitor - Multi-Camera Application")
        print_usage_instructions()