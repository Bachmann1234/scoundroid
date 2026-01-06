# GitHub Actions Workflows

This directory contains CI/CD workflows for the Scoundroid project.

## Workflows

### CI (`ci.yml`)
**Triggers:** Push to main/master/develop/claude/** branches, Pull Requests to main/master/develop

**Jobs:**
1. **test-and-lint**
   - Runs ktlint check for Kotlin formatting
   - Runs unit tests (`./gradlew test`)
   - Runs Android Lint
   - Uploads test and lint results as artifacts
   - Publishes test results to PR

2. **build**
   - Builds debug APK
   - Uploads APK as artifact (7-day retention)

## Local Commands

```bash
# Check formatting
./gradlew ktlintCheck

# Auto-format code
./gradlew ktlintFormat

# Run tests
./gradlew test

# Run lint
./gradlew lint

# Run all checks
./gradlew check

# Build debug APK
./gradlew assembleDebug
```

## Configuration

- **ktlint**: Configured in `app/build.gradle.kts` and `.editorconfig`
- **Android Lint**: Configured in `app/build.gradle.kts` lint block
- **JDK Version**: Java 17 (required for Android SDK 35)
