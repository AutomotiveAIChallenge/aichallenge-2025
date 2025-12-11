# FROM osrf/ros:humble-desktop AS common
FROM ghcr.io/automotiveaichallenge/autoware-universe:humble-latest AS common

COPY ./vehicle/zenoh-bridge-ros2dds_1.5.0_amd64.deb /tmp/
RUN apt install /tmp/zenoh-bridge-ros2dds_1.5.0_amd64.deb
COPY packages.txt /tmp/packages.txt
RUN apt-get update \
    && xargs -a /tmp/packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-4 && \
    rm -rf /var/lib/apt/lists/*
  

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV LD_LIBRARY_PATH="/usr/local/cuda-12-4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ENV PATH="/usr/local/cuda-12-4/bin${PATH:+:${PATH}}"

ENV XDG_RUNTIME_DIR=/tmp/xdg
ENV ROS_LOCALHOST_ONLY=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=file:///opt/autoware/cyclonedds.xml

COPY vehicle/cyclonedds.xml /opt/autoware/cyclonedds.xml

FROM common AS dev

RUN echo 'export PS1="\[\e]0;(AIC_DEV) ${debian_chroot:+($debian_chroot)}\u@\h: \w\a\](AIC_DEV) ${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /etc/skel/.bashrc
RUN echo 'cd /aichallenge' >> /etc/skel/.bashrc
RUN echo 'eval $(resize)' >> /etc/skel/.bashrc
ENV RCUTILS_COLORIZED_OUTPUT=1

FROM common AS eval

ENV RCUTILS_COLORIZED_OUTPUT=0

RUN mkdir /ws
RUN git clone --depth 1 https://github.com/AutomotiveAIChallenge/aichallenge-2025 /ws/repository
RUN mv /ws/repository/aichallenge /aichallenge
RUN rm -rf /aichallenge/simulator
RUN rm -rf /aichallenge/workspace/src/aichallenge_submit
RUN chmod 757 /aichallenge

COPY aichallenge/simulator/ /aichallenge/simulator/
COPY submit/aichallenge_submit.tar.gz /ws
RUN tar zxf /ws/aichallenge_submit.tar.gz -C /aichallenge/workspace/src
RUN rm -rf /ws

RUN bash -c ' \
  source /autoware/install/setup.bash; \
  cd /aichallenge/workspace; \
  rosdep update; \
  rosdep install -y -r -i --from-paths src --ignore-src --rosdistro $ROS_DISTRO; \
  colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'

ENTRYPOINT []
CMD ["bash", "/aichallenge/run_evaluation.bash"]