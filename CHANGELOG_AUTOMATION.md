# CHANGELOG Automation with Claude Code

This document describes how to set up automated CHANGELOG.md updates using the claude-code GitHub Action.

## Overview

The automation detects merges to the main branch and uses Claude to intelligently update the CHANGELOG.md file based on commit messages and PR descriptions. This reduces manual overhead while maintaining a high-quality changelog.

## Setup Instructions

### 1. Choose a Workflow

Two workflow options are provided:

- **Basic (`update-changelog.yml`)**: Simple automation with minimal configuration
- **Advanced (`update-changelog-advanced.yml`)**: Full-featured with better analysis and error handling

### 2. Add the Workflow

1. Copy your chosen workflow file to `.github/workflows/update-changelog.yml`
2. Ensure you have the required secrets configured (see below)

### 3. Configure Secrets

Required GitHub secrets:
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude
- `CLAUDE_CODE_PRIVATE_KEY`: Private key for the Claude Code GitHub App (if using advanced workflow)

### 4. Configure Variables (Advanced workflow only)

GitHub variables:
- `CLAUDE_CODE_APP_ID`: The Claude Code GitHub App ID (default: 1040539)

## How It Works

### Trigger Events

The workflow runs on:
- Push to main branch
- Merged pull requests to main
- Manual trigger (advanced workflow only)

### Process Flow

1. **Commit Analysis**: Examines recent commits for changelog-worthy changes
2. **Keyword Detection**: Looks for conventional commit patterns (feat, fix, etc.)
3. **Skip Detection**: Avoids infinite loops by skipping its own commits
4. **Claude Update**: Uses Claude to intelligently update the CHANGELOG
5. **Auto-commit**: Commits and pushes changes with `[skip ci]` flag

### What Gets Documented

The automation looks for:
- **Features**: `feat`, `feature`, `add`, `new`
- **Fixes**: `fix`, `bug`, `patch`, `repair`
- **Breaking Changes**: `breaking`, `!:`, `BREAKING CHANGE`
- **Performance**: `perf`, `performance`, `optimize`, `speed`
- **Documentation**: `docs`, `documentation`

### What Gets Skipped

- Minor refactoring without user impact
- Typo fixes in code comments
- Development dependency updates
- CI/CD changes (unless they affect users)
- Commits from the changelog automation itself

## Advanced Features

The advanced workflow includes:

- **Version Bump Detection**: Automatically creates new version sections
- **PR Label Analysis**: Uses PR labels for better categorization
- **Detailed Summaries**: Creates GitHub Action summaries with analysis results
- **Error Handling**: Graceful failure with helpful messages
- **Manual Triggers**: Force updates when needed
- **Better Git Integration**: Uses GitHub App tokens for proper attribution

## Customization

### Modify Keywords

Edit the keyword patterns in the workflow:
```yaml
FEAT_KEYWORDS="feat|feature|add|new|implement"
FIX_KEYWORDS="fix|bug|patch|repair|resolve"
```

### Change Claude's Behavior

Modify the `custom_instructions` in the workflow to:
- Adjust formatting preferences
- Add project-specific categories
- Change description style
- Add custom rules

### Skip Certain Files

Add path filters to the workflow trigger:
```yaml
on:
  push:
    branches:
      - main
    paths-ignore:
      - 'docs/**'
      - 'tests/**'
```

## Best Practices

1. **Commit Messages**: Use conventional commit format for better detection
2. **PR Descriptions**: Include clear descriptions of changes
3. **PR Labels**: Use labels like `enhancement`, `bug`, `breaking-change`
4. **Manual Review**: Periodically review automated updates
5. **Version Releases**: Create tags/releases to trigger version sections

## Troubleshooting

### Workflow Not Running

- Check if commits contain `[skip ci]` in the message
- Verify secrets are properly configured
- Check workflow permissions in repository settings

### Updates Not Appearing

- Ensure commits match keyword patterns
- Check if the last commit was from the automation
- Review workflow logs for Claude's analysis

### Incorrect Updates

- Review and adjust Claude's instructions
- Consider using more specific commit messages
- Use PR descriptions for complex changes

## Example Output

When the workflow runs successfully, it will update CHANGELOG.md like:

```markdown
## Unreleased

### New Features
- Add document reranking functionality using Qwen3-Reranker-4B model (#123)
- Implement structured generation with GBNF grammar support (#124)

### Bug Fixes
- Fix memory leak in daemon mode when processing large batches (#125)
- Resolve context window calculation for Gemma models (#126)

### Documentation
- Update PostgreSQL extension documentation with async examples (#127)
```

## Manual Intervention

Sometimes manual updates are still needed for:
- Major version releases
- Detailed migration guides
- Security advisories
- Complex feature explanations

The automation complements, not replaces, human judgment in maintaining a quality changelog.