#!/usr/bin/env node

/**
 * Script to read version from ra_aid/__version__.py and create version.json
 * in the Docusaurus static directory.
 */

const fs = require('fs');
const path = require('path');

// Paths
const versionFilePath = path.resolve(__dirname, '../../ra_aid/__version__.py');
const outputPath = path.resolve(__dirname, '../static/version.json');

/**
 * Extract version string from the __version__.py file
 * @param {string} content - The file content
 * @returns {string|null} - Extracted version or null if not found
 */
function extractVersion(content) {
  const regex = /__version__\s*=\s*["']([^"']+)["']/;
  const match = content.match(regex);
  return match ? match[1] : null;
}

// Main function to create version.json
function createVersionJson() {
  try {
    // Read version file
    console.log(`Reading version from ${versionFilePath}...`);
    const versionFileContent = fs.readFileSync(versionFilePath, 'utf8');
    
    // Extract version
    const version = extractVersion(versionFileContent);
    if (!version) {
      console.error('Failed to extract version from file');
      process.exit(1);
    }
    
    console.log(`Extracted version: ${version}`);
    
    // Create version JSON
    const versionJson = JSON.stringify({ version }, null, 2);
    
    // Ensure static directory exists
    const staticDir = path.dirname(outputPath);
    if (!fs.existsSync(staticDir)) {
      console.log(`Creating directory: ${staticDir}`);
      fs.mkdirSync(staticDir, { recursive: true });
    }
    
    // Write JSON file
    fs.writeFileSync(outputPath, versionJson);
    console.log(`Version JSON written to ${outputPath}`);
  } catch (error) {
    console.error(`Error creating version.json: ${error.message}`);
    process.exit(1);
  }
}

// Execute the main function
createVersionJson();