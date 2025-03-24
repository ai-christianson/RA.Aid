#!/bin/bash

echo "Building docs in CI..."
cd docs
corepack enable
npm install
npm run build
