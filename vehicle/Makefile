SHELL := /bin/bash
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
download:
	@read -p "Enter Username: " USERNAME; \
	read -sp "Enter Password: " PASSWORD; \
	echo ""; \
	if [ -z "$$USERNAME" ] || [ -z "$$PASSWORD" ]; then \
		echo "\nError: USERNAME and PASSWORD must be provided."; \
		exit 1; \
	fi; \
	echo "Downloading submission data..."; \
	CMD="python3 ../download_latest_submission.py --username='$$USERNAME' --password='$$PASSWORD'"; \
	if [ -n "$(SUBMISSION_ID)" ]; then \
		CMD="$$CMD --submission-id='$(SUBMISSION_ID)'"; \
	fi; \
	OUTPUT=$$(eval $$CMD 2>&1); \
	echo "$$OUTPUT"; \
	GREPPED_LINE=$$(echo "$$OUTPUT" | grep "File saved to:"); \
	TARBALL_PATH=$$(echo "$$GREPPED_LINE" | awk '{print $$NF}' | tr -d '\r'); \
	if [ -n "$$TARBALL_PATH" ] && [ -f "$$TARBALL_PATH" ]; then \
		echo "----------------------------------------"; \
		echo "✅ Downloaded file found: $$TARBALL_PATH"; \
		SUBMIT_DIR_PATH="/home/$$USER/aichallenge-2025/aichallenge/workspace/src/aichallenge_submit"; \
		EXTRACT_DIR_PATH="/home/$$USER/aichallenge-2025/aichallenge/workspace/src"; \
		echo "🧹 Deleting existing directory: $$SUBMIT_DIR_PATH"; \
		rm -rf "$$SUBMIT_DIR_PATH"; \
		echo "📦 Extracting archive to: $$SUBMIT_DIR_PATH/"; \
		tar -xzf "$$TARBALL_PATH" -C "$$EXTRACT_DIR_PATH"; \
		rm  "$$TARBALL_PATH"; \
		echo "Extraction completed successfully."; \
		echo "----------------------------------------"; \
	else \
		echo "----------------------------------------"; \
		echo "⚠️ Extraction failed: Could not find a valid file to extract."; \
		echo "----------------------------------------"; \
	fi
	@echo "Process finished."

# rvizの起動・ログ表示
rviz2:
	docker compose stop rviz2
	docker compose up -d rviz2
	docker compose logs -f rviz2
