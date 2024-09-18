NVIDIA DeepStream Object Detection System
====================================================

Overview
--------

This repository provides an example of using NVIDIA DeepStream to build a livestream object detection system. It processes multiple input streams, such as RTSP feeds or local video files, and performs object detection in real-time. The core of the system utilizes the `nvstreamdemux` plugin to split input batches and output separate streams, making it highly scalable for multi-stream processing.

The project demonstrates how to handle different display modes, encoding types, and codecs, enabling flexible deployment for various streaming and detection applications. It’s designed to support both hardware-accelerated and software-based encoding, ensuring optimal performance across diverse setups.

![example](video/example.gif)

### Prerequisites

*   Docker
*   Docker Compose
*   NVIDIA Docker runtime (for GPU support)

Usage
-----

The `main.py` script allows you to process multiple URI streams (RTSP or local video files) using NVIDIA DeepStream. The script leverages the `nvstreamdemux` plugin to split batches and output separate buffers/streams.

### Command-Line Arguments

You can control the behavior of the script through the following command-line arguments:

    
    python main.py [-d DISPLAY] [-t RTSP_TXT_PATH] [-c CODEC] [-e ENC_TYPE]
    

#### Arguments:

*   **\-d**, **\--display**: Controls the display output mode. Options include:
    
    *   `0`: No display (fakesink)
    *   `1`: Display on screen (EGLSink, nv3dsink)
    *   `2`: RTSP streaming (output via UDPSINK)
    
    Default: `0`.
*   **\-t**, **\--rtsps\_txt**: Path to a `.txt` file containing RTSP stream links or video file URIs. **Required**.
*   **\-c**, **\--codec**: Specifies the codec for RTSP streaming. Options include:
    *   `H264`: Default value.
    *   `H265`
*   **\-e**, **\--enc\_type**: Determines the encoding method. Options include:
    *   `0`: Hardware encoder (default).
    *   `1`: Software encoder.

### RTSP/Video Input Format

The `.txt` file provided via the `-t` argument must contain a list of RTSP streams and/or video file URIs, with one entry per line. The supported formats are:

*   **Video file format**:
    
        file:///home/user/videos/sample.mp4
    
*   **RTSP stream format**:
    
        rtsp://username:password@ip/stream1
    

Example:

    
    file:///home/user/videos/sample.mp4
    rtsp://user:pass@192.168.1.10:554/stream2
        

### Example Usage

**1\. No Display (fakesink):**

    python main.py -d 0 -t video_source.txt

**2\. Display on Screen (EGLSink):**

    python main.py -d 1 -t video_source.txt

**3\. RTSP Output via UDPSINK:**

    python main.py -d 2 -t video_source.txt

Ensure that the `.txt` file contains valid RTSP links or video file path in the correct format for processing.

**4\. Build and Run the Docker Container**

    docker build -t deepstream/object-detection .
    
    docker compose up
