# Use Python 3.11 slim image for better compatibility with your dependencies
FROM python:3.11-slim

# Set environment variables for headless operation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    QT_QPA_PLATFORM=offscreen \
    QT_X11_NO_MITSHM=1 \
    QT_DEBUG_PLUGINS=0 \
    OPENCV_VIDEOIO_PRIORITY_MSMF=0 \
    CUDA_VISIBLE_DEVICES="" \
    OPENCV_DNN_BACKEND=0 \
    MPLBACKEND=Agg

# Set work directory
WORKDIR /app

# Install system dependencies required for OpenCV, face-recognition, and other packages
# In your existing Dockerfile, update the Qt/GUI dependencies section:

# Install system dependencies required for OpenCV, face-recognition, and other packages
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # Network tools
    iproute2 \
    net-tools \
    # OpenCV dependencies
    libopencv-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Math libraries
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # dlib dependencies for face-recognition
    libboost-all-dev \
    # PostgreSQL dependencies
    libpq-dev \
    # Minimal Qt5 and X11 for GUI support
    qtbase5-dev \
    libqt5gui5 \
    libqt5widgets5 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    # Virtual display
    xvfb \
    # Additional utilities
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
# Upgrade pip and install wheel for better package compilation
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations for mini PC
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for application structure
RUN mkdir -p /app/src/app /app/logs /app/data

# Copy the entire application
COPY . .

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set Python path to include src directory
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Health check to ensure the application is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose port if your app serves any web interface (uncomment if needed)
# EXPOSE 8080

# Create wrapper script that sets up environment without modifying your code
# Update the entrypoint script creation in your Dockerfile:
RUN echo '#!/bin/bash\n\
\n\
echo "Setting up virtual display for GUI applications..."\n\
\n\
# Start Xvfb virtual display\n\
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &\n\
XVFB_PID=$!\n\
\n\
# Wait for Xvfb to start\n\
sleep 3\n\
\n\
# Set display environment\n\
export DISPLAY=:99\n\
\n\
# Qt configuration for virtual display\n\
export QT_X11_NO_MITSHM=1\n\
export QT_QPA_PLATFORM=xcb\n\
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins\n\
\n\
# OpenCV configuration\n\
export OPENCV_VIDEOIO_PRIORITY_MSMF=0\n\
export CUDA_VISIBLE_DEVICES=""\n\
export OPENCV_DNN_BACKEND=0\n\
\n\
echo "Display setup complete. Starting application..."\n\
\n\
# Function to cleanup on exit\n\
cleanup() {\n\
    echo "Cleaning up..."\n\
    kill $XVFB_PID 2>/dev/null\n\
    exit\n\
}\n\
\n\
# Set trap for cleanup\n\
trap cleanup SIGTERM SIGINT\n\
\n\
# Run your application\n\
cd /app\n\
python main.py &\n\
APP_PID=$!\n\
\n\
# Wait for the application to finish\n\
wait $APP_PID\n\
\n\
# Cleanup\n\
cleanup\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Use the wrapper script as entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]
EXPOSE 8765