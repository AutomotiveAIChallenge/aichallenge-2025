SHELL := /bin/bash

# autowareのみ起動
run-autoware-only:
	docker compose up -d autoware
	docker compose logs -f autoware

# racing kartとzenohの起動 
run-driver-zenoh:
	docker compose up -d driver zenoh
	docker compose logs -f driver zenoh

# Autowareとracing kartとzenohの起動
run-full-system:
	docker compose up -d autoware driver zenoh
	docker compose logs -f autoware driver zenoh

# Download submission data by asking for credentials interactively
# Usage:
#   make download [SUBMISSION_ID=<id>]
download-only:
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
# autowareのbuildのみ
build-only:
	docker compose up -d aic-build
	docker compose logs -f aic-build

# 上記全部やっちゃう
all:download-only
	docker compose up -d aic-build driver zenoh autoware
	docker compose logs -f aic-build dricer zenoh autoware
# all:download-only build-only run-driver-zenoh run-autoware-only
# 	@echo "All services started, showing logs..."
# 	docker compose logs -f autoware driver zenoh

# Stop all services
stop-all:
	docker compose down
