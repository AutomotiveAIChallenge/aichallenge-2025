#!/bin/bash
RUST_BACKTRACE=1 zenoh-bridge-ros2dds client -e tls/57.180.63.135:7447 -c zenoh-user.json5
