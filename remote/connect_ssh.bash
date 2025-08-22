#!/bin/bash
ssh -ACY -p 10022 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    tier4@57.180.63.135
