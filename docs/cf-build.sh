#!/bin/bash

echo "Building docs in CI..."
corepack enable
npm install
npm run build
