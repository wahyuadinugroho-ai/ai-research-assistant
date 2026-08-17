.PHONY: install run clean clean-history clean-all

# Install dependencies using uv (if you use uv package manager)
install:
	uv sync

format:
	ruff check --fix src && ruff format src

lint:
	ruff check src

# Run the Streamlit application
run:
	streamlit run main.py

# Clean Python caches and local vector DB files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache

# Clean all saved chat history and archives
clean-history:
	rm -rf chat_history/*.json
	rm -rf chat_archive/*.json
	@echo "Chat history and archives cleared."

# Deep clean everything (caches, DBs, and history)
clean-all: clean clean-history
	@echo "Deep clean complete."
