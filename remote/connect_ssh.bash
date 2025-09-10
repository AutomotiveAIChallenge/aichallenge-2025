#!/bin/bash
autossh -ACY -M 0 -p 10022 \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    tier4@57.180.63.135
