#!/bin/bash

# Build and serve MkDocs site

echo "🚀 MkDocs Algorithmic Trading Documentation"
echo "==========================================="

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "❌ MkDocs not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Parse command line arguments
case "$1" in
    "serve"|"")
        echo "📡 Starting development server..."
        echo "🌐 Site will be available at: http://localhost:8000"
        mkdocs serve
        ;;
    "build")
        echo "🔨 Building static site..."
        mkdocs build --clean
        echo "✅ Site built successfully in 'site/' directory"
        ;;
    "deploy")
        echo "🚀 Deploying to GitHub Pages..."
        mkdocs gh-deploy --clean
        echo "✅ Site deployed successfully"
        ;;
    "new-page")
        if [ -z "$2" ]; then
            echo "❌ Please specify a page name"
            echo "Usage: ./build.sh new-page <section/page-name>"
            exit 1
        fi
        echo "📝 Creating new page: $2"
        mkdir -p "docs/$(dirname "$2")"
        touch "docs/$2.md"
        echo "# $(basename "$2" .md)" > "docs/$2.md"
        echo "✅ Created docs/$2.md"
        ;;
    "clean")
        echo "🧹 Cleaning build artifacts..."
        rm -rf site/
        echo "✅ Cleaned successfully"
        ;;
    "help")
        echo "Available commands:"
        echo "  serve     - Start development server (default)"
        echo "  build     - Build static site"
        echo "  deploy    - Deploy to GitHub Pages"
        echo "  new-page  - Create new page"
        echo "  clean     - Clean build artifacts"
        echo "  help      - Show this help"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Run './build.sh help' for available commands"
        exit 1
        ;;
esac
