# Current State & Next Steps

**Last Updated**: 2026-01-05

## What's Already Done

✅ Android project created with correct structure
✅ Package name: `dev.mattbachmann.scoundroid`
✅ Target SDK: 36 (latest Android)
✅ Min SDK: 36 (Pixel 10 Pro Fold specific)
✅ Project name: "Scoundroid"
✅ Basic project structure (app module, gradle config)
✅ Test files scaffolded

## What's NOT Done Yet

❌ Jetpack Compose not configured (currently uses traditional View system)
❌ No Activity/UI code
❌ No game logic
❌ No data models
❌ No dependencies for Compose, Room, etc.

## Current Project Analysis

### build.gradle.kts Status
The current build file is very minimal with only basic dependencies:
- androidx.core.ktx
- androidx.appcompat
- material (traditional Material, not Compose)

**Needs**: Migration to Jetpack Compose dependencies

### Manifest Status
The manifest has no activities declared - it's completely empty except for app metadata.

**Needs**: MainActivity and proper configuration

### Source Code Status
Only test files exist:
- `ExampleInstrumentedTest.kt`
- `ExampleUnitTest.kt`

**Needs**: All actual source code

## Recommended First Session Tasks

**IMPORTANT**: We're using Test-Driven Development (TDD). See [`05-testing-strategy.md`](05-testing-strategy.md) for details.

When you're ready to start coding, here's what we should do first:

### 1. Configure Jetpack Compose & Test Dependencies
- Add Compose BOM to dependencies
- Add Compose UI, Material3, ViewModel dependencies
- **Add testing dependencies (JUnit, Kotlin Test, Turbine, MockK)**
- Enable Compose in build config
- Update to Material3 theme

### 2. Set Up Test Infrastructure
- Create test directory structure
- Create test utility classes (test builders)
- Set up test configuration

### 3. Create MainActivity (Minimal)
- Create `MainActivity.kt` with basic Compose setup
- Add to manifest
- Create simple "Hello Scoundroid" screen to verify setup

### 4. Set Up Package Structure
- Create package structure:
  - `data/model/` (and `test/data/model/`)
  - `ui/screen/`
  - `ui/component/`
  - `ui/theme/`

### 5. Create Core Data Models (TDD!)
- **Write tests for Card class** (value calculation, type determination)
- Create `Card.kt`
- **Verify tests pass**
- **Write tests for enums** (Suit, Rank, CardType)
- Create enum classes
- **Verify tests pass**
- **Write tests for deck initialization**
- Implement deck initialization
- **Verify tests pass**

### 6. Verify Build & Tests
- Build project: `./gradlew build`
- **Run tests: `./gradlew test`**
- **All tests must pass**
- Optionally run on device to verify UI

## Quick Commands for Next Session

```bash
# Run unit tests (do this frequently!)
./gradlew test

# Run tests with coverage report
./gradlew testDebugUnitTestCoverage

# Run specific test class
./gradlew test --tests CardTest

# Build the project
./gradlew build

# Run all checks (tests + lint)
./gradlew check

# Install on device
./gradlew installDebug

# Clean build
./gradlew clean build
```

## TDD Workflow Reminder

**Every feature follows this cycle:**

1. ✍️ **Write failing test** (RED)
2. ✅ **Write code to pass test** (GREEN)
3. 🔄 **Refactor while keeping tests green**
4. 🔁 **Repeat**

**Never write implementation before tests (except UI components)**

## Files to Create in First Session

1. `app/src/main/java/dev/mattbachmann/scoundroid/MainActivity.kt`
2. `app/src/main/java/dev/mattbachmann/scoundroid/data/model/Card.kt`
3. `app/src/main/java/dev/mattbachmann/scoundroid/data/model/Suit.kt`
4. `app/src/main/java/dev/mattbachmann/scoundroid/data/model/Rank.kt`
5. `app/src/main/java/dev/mattbachmann/scoundroid/data/model/CardType.kt`
6. `app/src/main/java/dev/mattbachmann/scoundroid/ui/theme/Theme.kt`
7. `app/src/main/java/dev/mattbachmann/scoundroid/ui/theme/Color.kt`

## Dependencies to Add

Update `app/build.gradle.kts` with:

```kotlin
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)  // Add this
}

android {
    // ... existing config ...

    buildFeatures {
        compose = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.8"
    }
}

dependencies {
    // Compose BOM
    val composeBom = platform("androidx.compose:compose-bom:2024.10.01")
    implementation(composeBom)
    androidTestImplementation(composeBom)

    // Compose
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    debugImplementation("androidx.compose.ui:ui-tooling")

    // Activity Compose
    implementation("androidx.activity:activity-compose:1.9.0")

    // ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")

    // Existing dependencies
    implementation(libs.androidx.core.ktx)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
```

You may also need to update `gradle/libs.versions.toml` if it exists.

## Ready to Start?

When you're ready to begin implementation:
1. Say "Let's start Session 1"
2. Or specify which part you want to work on
3. Or ask any questions about the plan

The planning is complete and saved in this `plans/` directory. We can pick up anytime!
