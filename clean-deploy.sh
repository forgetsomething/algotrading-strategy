#!/bin/bash

# Clean Deployment Script for MkDocs GitHub Pages

echo "🧹 Cleaning up deployment artifacts..."

# Remove any cached build files
rm -rf site/
rm -rf .cache/

echo "🔨 Building fresh MkDocs site..."

# Clean build with verbose output
mkdocs build --clean --verbose

echo "✅ Fresh build complete!"
echo "📁 Site contents:"
ls -la site/

echo "🎨 CSS files:"
ls -la site/stylesheets/

echo "🚀 Ready for deployment!"
