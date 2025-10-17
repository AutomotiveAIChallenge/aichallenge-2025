SHELL := /bin/bash

.PHONY: download

# 既にコンテナが立ち上がっている場合はログの表示を行います。
# autowareのみ起動・ログ表示
autoware:
	docker compose up -d autoware
	docker compose logs -f autoware

# racing kartの起動・ログ表示
driver:
	docker compose up -d driver
	docker compose logs -f driver

# zenohの起動・ログ表示
zenoh:
	docker compose up -d zenoh
	docker compose logs -f zenoh

# aicコンテナにてros2 bag record -a
#rosbag:
#	docker compose up -d rosbag 
#	docker compose logs -f rosbag

run-full-system:
	docker compose up -d driver
	docker compose up -d autoware
# 	$(MAKE) rosbag
	sleep 15
	docker compose up -d zenoh
	docker compose logs -f autoware driver zenoh 2>&1 | tee log/full_log.log | stdbuf -oL grep -E "(ERROR|WARN)"

# Download submission data by asking for credentials interactively
# Usage:
#   make download [SUBMISSION_ID=<id>]
# Usage (Only Admins):
#   make download [USER_ID=<id>] [SUBMISSION_ID=<id>]
download:
	@if [ -n "$(USER_ID)" ]; then \
		if [ -n "$(SUBMISSION_ID)" ]; then \
			./download_submission.sh --output ../aichallenge/workspace/src/ --user-id $(USER_ID) --submission-id $(SUBMISSION_ID); \
		else \
			./download_submission.sh --output ../aichallenge/workspace/src/ --user-id $(USER_ID); \
		fi; \
	else \
		if [ -n "$(SUBMISSION_ID)" ]; then \
			./download_submission.sh --output ../aichallenge/workspace/src/ --submission-id $(SUBMISSION_ID); \
		else \
			./download_submission.sh --output ../aichallenge/workspace/src/; \
		fi; \
	fi

# rvizの起動・ログ表示
rviz2:
	docker compose stop rviz2
	docker compose up -d rviz2
	docker compose logs -f rviz2
