#!/bin/bash

echo "Building docs in CI..."
cd ../
corepack enable
npm install
npm run build
mv -f build .cf/build
