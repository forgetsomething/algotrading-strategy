# GitHub Pages Configuration Guide

## Current Issue
The GitHub Pages site is serving the repository README content instead of the MkDocs-generated site.

## Root Cause
GitHub Pages is likely configured to deploy from a branch (main) instead of GitHub Actions.

## Solution Steps

### 1. Configure GitHub Pages Source
1. Go to: https://github.com/forgetsomething/algotrading-strategy
2. Click the "Settings" tab
3. In the left sidebar, click "Pages"
4. Under "Source", select **"GitHub Actions"** (NOT "Deploy from a branch")
5. Save the changes

### 2. Verify Workflow
- The GitHub Actions workflow is already correctly configured
- It builds the MkDocs site and uploads it as an artifact
- It should deploy automatically once Pages source is set to GitHub Actions

### 3. Expected Results
After configuration:
- Site should show ThinkBayes2-style layout with left sidebar navigation
- Homepage should display book-style content from docs/index.md
- Custom CSS styling should be applied

### 4. Verification URLs
- Main site: https://forgetsomething.github.io/algotrading-strategy/
- Test page: https://forgetsomething.github.io/algotrading-strategy/test-deployment.html

### 5. Troubleshooting
If the issue persists:
- Check GitHub Actions tab for any failed workflows
- Ensure the deployment step shows "success"
- Clear browser cache or try incognito mode

## Technical Details
- Local site works correctly: ✅
- MkDocs build successful: ✅
- CSS files generated: ✅
- GitHub Actions workflow: ✅
- Missing: Proper Pages source configuration
