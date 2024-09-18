FROM nvcr.io/nvidia/deepstream:7.0-samples-multiarch

COPY .  /root/apps/
WORKDIR /root/apps/

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

RUN /opt/nvidia/deepstream/deepstream-7.0/user_additional_install.sh

# Update and install required packages
RUN apt-get update && \
    apt-get install -y python3-gi python3-dev python3-gst-1.0 && \
    apt-get install -y python3-opencv python3-numpy && \
    apt-get install -y libgstrtspserver-1.0-0 gstreamer1.0-rtsp && \
    apt-get install -y libgirepository1.0-dev gobject-introspection gir1.2-gst-rtsp-server-1.0 && \
    apt-get install -y libcairo2-dev libpq-dev

RUN pip3 install -r requirements.txt
RUN pip3 install ./pyds-1.1.11-py3-none*.whl

ENTRYPOINT ["python3", "main.py", "-t", "example_video_source.txt"]