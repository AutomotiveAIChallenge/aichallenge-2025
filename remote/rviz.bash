#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
AIC_DIR="${REPO_ROOT}/aichallenge"
COMPOSE_FILE="${AIC_DIR}/docker-compose.yml"

usage() {
    cat <<'USAGE'
Usage:
  rviz.bash            # start RViz stack via make rviz2
  rviz.bash down       # stop and remove rviz2 service
  rviz.bash restart    # restart rviz2 service
USAGE
}

if [ ! -d "${AIC_DIR}" ]; then
    echo "Error: vehicle directory not found at '${AIC_DIR}'." >&2
    exit 1
fi

if [ ! -f "${AIC_DIR}/Makefile" ]; then
    echo "Error: Makefile not found in '${AIC_DIR}'." >&2
    exit 1
fi

if [ ! -f "${COMPOSE_FILE}" ]; then
    echo "Error: docker-compose.yml not found at '${COMPOSE_FILE}'." >&2
    exit 1
fi

mode="start"
if [ $# -gt 0 ]; then
    case "$1" in
    down)
        mode="down"
        shift
        ;;
    restart)
        mode="restart"
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown argument '$1'." >&2
        usage
        exit 1
        ;;
    esac
fi

if [ $# -gt 0 ]; then
    echo "Error: too many arguments." >&2
    usage
    exit 1
fi

case "${mode}" in
start)
    echo "Running 'make rviz2' inside '${AIC_DIR}'."
    cd "${AIC_DIR}"
    make rviz2 "$@"
    ;;
down)
    echo "Stopping and removing 'rviz2' service using compose file '${COMPOSE_FILE}'."
    docker compose -f "${COMPOSE_FILE}" rm -f -s rviz2
    ;;
restart)
    echo "Restarting 'rviz2' service."
    docker compose -f "${COMPOSE_FILE}" rm -f -s rviz2
    echo "Running 'make rviz2' inside '${AIC_DIR}' after restart."
    cd "${AIC_DIR}"
    make rviz2 "$@"
    ;;
*)
    echo "Error: unsupported mode '${mode}'." >&2
    exit 1
    ;;
esac
