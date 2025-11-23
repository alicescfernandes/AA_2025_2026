#!/bin/bash

# Script to install git hooks

echo "Installing git hooks..."

# Create hooks directory if it doesn't exist
mkdir -p .git/hooks

# Copy pre-commit hook
cp pre-commit .git/hooks/pre-commit
cp pre-commit .git/hooks/pre-pull

# Make them executable
chmod +x .git/hooks/pre-commit
chmod +x .git/hooks/pre-pull

echo "✓ Pre-commit hook installed successfully!"
echo "✓ Pre-pull hook installed successfully!"
