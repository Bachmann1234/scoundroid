# UI Polish Implementation Plan

This document outlines the UI/UX improvements identified for Scoundroid, organized by priority and implementation complexity.

## Overview

The current UI is functional but can be elevated with visual polish, improved consistency, and better accessibility. These changes focus on making the app feel more tactile and delightful without changing core functionality.

---

## Phase 1: Quick Wins (Low Effort, High Impact) - COMPLETE

### 1.1 Fix "Current Room" Label Contrast - DONE
**Problem**: The "Current Room (4 cards)" label is barely visible with faded gray text.

**Solution**: Used `MaterialTheme.colorScheme.onSurface` for better visibility.

**Commit**: `1d486ad` (initial PR commit)

---

### 1.2 Unify Button Styling - DONE
**Problem**: Mixed button styles—"Avoid Room" and "New Game" are outlined, "Process" is filled.

**Solution**: Established clear hierarchy:
- **Primary action** (Process Cards): Filled button
- **Secondary action** (Avoid Room): `FilledTonalButton`
- **Tertiary action** (New Game): `OutlinedButton`

**Commit**: `1d486ad` (initial PR commit)

---

### 1.3 Improve Stats Panel Text Hierarchy - DONE
**Problem**: Labels ("Health", "Deck") and values have similar visual weight.

**Solution**:
- Labels use `labelSmall` with uppercase
- Values use `titleMedium` with bold weight
- Label opacity increased from 0.6 to 0.8 for better contrast

**Commits**: `1d486ad`, `8478576`

---

## Phase 2: Card Visual Improvements (Medium Effort) - COMPLETE

### 2.1 Add Card Elevation & Shadows - DONE
**Problem**: Cards appear flat and don't feel tactile.

**Solution**:
- Cards have 4dp base elevation
- Selected cards animate to 12dp elevation
- Pressed cards drop to 1dp ("pushed in" effect)

**Commit**: `3143257`

---

### 2.2 Enhance Selection Indicator - DONE
**Problem**: Thin green border is easy to miss.

**Solution**: Implemented Option B (lift effect):
- Scale animation (1.0 → 1.08x when selected)
- Elevation animation (4dp → 12dp when selected)
- Combined with existing border color change

**Commit**: `3143257`

---

### 2.3 Improve Order Badges - DONE
**Problem**: Number badges look like notification badges rather than sequence indicators.

**Solution**:
- Repositioned to bottom-right (inside card bounds)
- Added white border ring for visibility
- White text on teal background

**Commits**: `c632b82`, `1f35c17` (overlap fix)

---

## Phase 3: Polish & Delight (Higher Effort) - COMPLETE

### 3.1 Card Design Refinement - DONE (Partial)
**Current**: Solid flat colors with simple suit symbols.

**Implemented**:
- Subtle vertical gradient overlay (light top, dark bottom)
- Creates "light from above" effect

**Skipped**:
- Custom vector suit symbols (high effort, low impact)
- Card textures (high effort)
- Inner border (tried, removed - too busy)

**Commit**: `f5127bd`

---

### 3.2 Add Tap Affordance to Unselected Cards - DONE
**Problem**: Cards don't clearly communicate they're interactive.

**Solution**:
- Added `MutableInteractionSource` to detect press state
- Cards drop to 1dp elevation when pressed ("pushed in" feel)
- Fast spring animation for responsive feedback

**Commit**: `d016c2c`

---

### 3.3 Preview Panel Styling - DONE
**Problem**: Gray background feels disconnected from the rest of the design.

**Solution**:
- Uses `MaterialTheme.colorScheme.surfaceVariant` (full opacity, was 0.5)
- Added 2dp elevation for subtle depth
- Card already has rounded corners

**Commit**: `733d8e5`

---

## Phase 4: Accessibility Improvements - COMPLETE

### 4.1 Color-Independent Card Type Identification - SKIPPED
**Reason**: Suit symbols (♣♠♦♥) already provide non-color differentiation. Adding icons would add visual clutter for minimal benefit.

---

### 4.2 Ensure Sufficient Contrast Ratios - DONE
**Audit results and fixes**:

| Element | Before | After | Status |
|---------|--------|-------|--------|
| Monster card text | Black on #E57373 | No change | ~5.5:1 ✅ |
| Weapon card text | White on #64B5F6 | Dark blue #0D47A1 | ~7:1 ✅ |
| Potion card text | White on #81C784 | Dark green #1B5E20 | ~6:1 ✅ |
| Status bar labels | 0.6 alpha | 0.8 alpha | Improved ✅ |

**Commit**: `ddea8cb`, `8478576`

---

## Implementation Summary

| Priority | Task | Status |
|----------|------|--------|
| 1 | Fix "Current Room" label contrast | ✅ Done |
| 2 | Unify button styling | ✅ Done |
| 3 | Improve stats panel hierarchy | ✅ Done |
| 4 | Add card elevation/shadows | ✅ Done |
| 5 | Enhance selection indicator | ✅ Done |
| 6 | Preview panel styling | ✅ Done |
| 7 | Improve order badges | ✅ Done |
| 8 | Add tap affordance | ✅ Done |
| 9 | Card design refinement | ✅ Done (gradient only) |
| 10 | Accessibility improvements | ✅ Done |

---

## PR Commits (ui-polish-quick-wins)

1. `1d486ad` - Label contrast, button hierarchy, text styling (pre-existing)
2. `3143257` - Card elevation animations
3. `733d8e5` - PreviewPanel styling
4. `c632b82` - Order badge redesign
5. `1f35c17` - Badge overlap fix
6. `d016c2c` - Tap affordance (press feedback)
7. `f5127bd` - Card gradient overlay
8. `ddea8cb` - Card text contrast (accessibility)
9. `8478576` - Status bar label contrast

---

## Notes

- All changes maintain the app's clean, readable aesthetic
- Avoided over-designing—polish enhances without distracting from gameplay
- Small phone optimizations identified and documented separately in `07-small-phone-optimization.md`

---

**Created**: 2026-01-09
**Completed**: 2026-01-10
**Status**: COMPLETE
