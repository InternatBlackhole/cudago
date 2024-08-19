# Variables
APP_NAME := cudago
SRC_DIR := .
BUILD_DIR := ./build
SRC_FILES := $(wildcard $(SRC_DIR)/*.go)
BUILD_SRC_DIR := $(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%,$(SRC_FILES))
BIN_DIR := $(BUILD_DIR)/bin
BIN_FILE := $(BIN_DIR)/$(APP_NAME)

# Default target
all: build

# Build the Go project
build: $(BIN_FILE)

$(BIN_FILE): $(SRC_FILES) | $(BUILD_DIR) $(BIN_DIR)
	@mkdir -p $(BIN_DIR)
	@echo "Building $(APP_NAME)..."
	@go build -o $(BIN_FILE) $(SRC_FILES)
	@echo "Build completed: $(BIN_FILE)"

#$(BUILD_DIR)/%.go: $(SRC_DIR)/%.go | $(BUILD_DIR) $(BIN_DIR)
#	@echo "Filling in the template..."
#	@# Match the pattern that contains the path to the file, e.g., "{{path/to/file}}"
#	@# Extract the path from the matched pattern, removing the {{ and }}
#	@# Read the contents of the specified file
#	@# Replace the pattern with the contents of the file
#	@# Print the line (either modified or original)
#	@./exp.sh $(SRC_DIR)/$(notdir $@) > $@

# Create build and bin directories if they don't exist
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Clean up the binary and any other generated files
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned."

# Run the application
run: build
	@echo "Running $(APP_NAME)..."
	@$(BIN_FILE)

# Install the binary
install: build
	@echo "Installing $(APP_NAME)..."
	@go install $(SRC_FILES)

# Format the Go source code
fmt:
	@echo "Formatting code..."
	@go fmt $(SRC_FILES)

# Run Go tests
test:
	@echo "Running tests..."
	@go test ./...

# Lint the Go source code
lint:
	@echo "Linting code..."
	@golangci-lint run

# Remove cached files
clean-cache:
	@echo "Cleaning Go cache..."
	@go clean -cache

.PHONY: all build clean run install fmt test lint clean-cache
