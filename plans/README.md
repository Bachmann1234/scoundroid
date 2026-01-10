# Scoundroid Plans Directory

Welcome! This directory contains all planning documents for the Scoundroid Android app project.

## Quick Start

**New session? Start here:**
1. Read `03-session-guide.md` to see where we are
2. Check what tasks are next
3. Begin working!

## Document Overview

### 00-project-overview.md
High-level project description, technology stack, and feature list. Read this first to understand what we're building.

### 01-task-breakdown.md
**Updated for TDD**: Detailed task breakdown organized into 6 phases with test-first tasks. Use this as a checklist to track progress.

### 02-technical-architecture.md
Technical design decisions, data models, and architecture patterns. Reference when implementing features.

### 03-session-guide.md
**Most Important**: Session-by-session roadmap with TDD workflow. Update this after each session.

### 04-current-state.md
Current project status and immediate next steps. Check this when resuming work.

### 05-testing-strategy.md
**Test-First Approach**: Comprehensive testing strategy, TDD workflow, and test case examples. Critical for implementation.

### 06-help-system-design.md
**Help & Tutorial**: Design for in-game help system, tutorial flow, and contextual tooltips. Important for usability since rules are complex.

## Project Quick Facts

- **Name**: Scoundroid
- **Type**: Single-player card game for Android
- **Target Device**: Pixel 10 Pro Fold
- **Tech**: Kotlin + Jetpack Compose
- **Game**: Scoundrel by Zach Gage and Kurt Bieg
- **Rules**: See `../docs/rules.md`

## How to Use These Plans

1. **Starting a new session**:
   - Check `04-current-state.md` for current status
   - Check `03-session-guide.md` for next tasks
   - **Run `./gradlew test` to verify current state**
2. **Writing code**: **Read `05-testing-strategy.md` for TDD workflow**
3. **Need technical details**: Check `02-technical-architecture.md`
4. **Want full task list**: Check `01-task-breakdown.md`
5. **Need project context**: Check `00-project-overview.md`

## Test-Driven Development

**This project uses TDD** - Write tests before implementation!

- See `05-testing-strategy.md` for comprehensive guide
- All game logic requires 100% test coverage
- Tests must pass before moving to next phase
- Run `./gradlew test` frequently

## Progress Tracking

Mark tasks complete in `01-task-breakdown.md` as you finish them.
Update current phase/session in `03-session-guide.md`.

## Notes & Discoveries

Add any important discoveries, decisions, or notes here as we work:

- [Date] Note 1
- [Date] Note 2

---

**Last Updated**: 2026-01-05
**Status**: Planning complete, ready to begin implementation
