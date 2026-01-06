# Scoundroid

An Android implementation of **Scoundrel**, a single-player rogue-like card game by Zach Gage and Kurt Bieg.

## About

Scoundroid brings the strategic card game Scoundrel to Android, optimized for the Pixel 10 Pro Fold. Navigate through a dungeon of cards, fighting monsters with weapons that degrade with use, and try to survive with your health intact.

### Game Rules

- **Deck**: 44 cards (standard deck minus Jokers, red face cards, and red Aces)
- **Cards**:
  - **Monsters** (26 Clubs & Spades): Deal damage equal to their value (2-14)
  - **Weapons** (9 Diamonds): Reduce monster damage but degrade with use
  - **Potions** (9 Hearts): Restore health (max 20)
- **Objective**: Survive the entire dungeon or score as high as possible

Key mechanic: Weapons can only be used on monsters of equal or lesser value than the last monster they defeated.

Original game rules: See [`docs/Scoundrel.pdf`](docs/Scoundrel.pdf) or [online](http://www.stfj.net/scoundrel/)

## Tech Stack

- **Language**: Kotlin
- **UI**: Jetpack Compose with Material3
- **Architecture**: MVI (Model-View-Intent)
- **Persistence**: Room Database
- **Min SDK**: 36 (Android latest)
- **Target Device**: Pixel 10 Pro Fold

## Project Status

🚧 **In Development**

See [`plans/`](plans/) directory for detailed planning documents and progress tracking.

## Building

```bash
# Build the project
./gradlew build

# Run tests
./gradlew test

# Check code formatting
./gradlew ktlintCheck

# Auto-format code
./gradlew ktlintFormat

# Run Android Lint
./gradlew lint

# Install on device
./gradlew installDebug
```

## CI/CD

[![CI](https://github.com/Bachmann1234/scoundroid/actions/workflows/ci.yml/badge.svg)](https://github.com/Bachmann1234/scoundroid/actions/workflows/ci.yml)

Automated workflows run on every push:
- **Tests**: Unit tests with JUnit and Kotlin Test
- **Formatting**: ktlint checks for Kotlin style
- **Linting**: Android Lint for code quality
- **Build**: Assembles debug APK

See [`.github/workflows/README.md`](.github/workflows/README.md) for details.

## Project Structure

```
scoundroid/
├── app/
│   └── src/main/java/dev/mattbachmann/scoundroid/
│       ├── data/          # Data models and repositories
│       ├── domain/        # Business logic and use cases
│       ├── ui/            # Compose UI (screens, components, theme)
│       └── util/          # Utilities
├── docs/                  # Game rules and documentation
├── plans/                 # Project planning documents
└── README.md              # This file
```

## Development

This is a personal project optimized specifically for the Pixel 10 Pro Fold. See the [Session Guide](plans/03-session-guide.md) for the development roadmap.

### Current Session

See [`plans/04-current-state.md`](plans/04-current-state.md) for the current development status and next steps.

## License

This is a personal project. Original Scoundrel game design © 2011 Zach Gage and Kurt Bieg.

## Credits

- **Original Game**: Zach Gage and Kurt Bieg
- **Android Implementation**: Matt Bachmann
