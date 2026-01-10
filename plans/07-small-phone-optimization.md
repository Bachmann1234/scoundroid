# Small Phone Optimization Plan

This document outlines UI/UX improvements for small phone screens (compact mode).

## Problem Statement

On small phones, the current layout has several issues:
- Action buttons are below the fold, requiring scroll to access
- Preview panel is cut off
- Button text wraps awkwardly ("Process 0/3 Cards" becomes two lines)
- Cards take up excessive vertical space
- "Draw Room" label has poor contrast/visibility

## Goals

1. All essential UI (cards + action buttons) visible without scrolling
2. Text fits properly without awkward wrapping
3. Maintain usability and touch targets

---

## Phase 1: Quick Wins (Low Effort)

### 1.1 Shorten Button Text
**Problem**: "Process 0/3 Cards" wraps to two lines on narrow screens.

**Solution**: Shorten to "Process 0/3" - "Cards" is redundant.

**Files to modify**: `ui/screen/game/GameScreen.kt`

---

### 1.2 Fix "Draw Room" Label Visibility
**Problem**: The "Draw Room" text appears very faded/low contrast.

**Solution**: Check the color/alpha and increase visibility.

**Files to modify**: `ui/screen/game/GameScreen.kt` or `ui/component/RoomDisplay.kt`

---

### 1.3 Reduce Vertical Spacing
**Problem**: Excessive padding/margins eat into limited vertical space.

**Solution**: Audit and reduce vertical spacing in compact mode:
- Title bar padding
- Status bar margins
- Card grid spacing
- Preview panel margins

**Files to modify**: `ui/screen/game/GameScreen.kt`, `ui/component/GameStatusBar.kt`

---

## Phase 2: Card Size Optimization (Medium Effort)

### 2.1 Smaller Cards on Compact Screens
**Problem**: Cards use fixed 100x140dp size regardless of screen.

**Solution**: Detect screen height and use smaller cards (e.g., 80x112dp) on compact screens.

**Implementation options**:
- Pass screen dimensions to RoomDisplay
- Use `LocalConfiguration.current.screenHeightDp` to determine size
- Define card size tiers: compact (<700dp height), regular (>=700dp)

**Files to modify**: `ui/component/RoomDisplay.kt`, `ui/screen/game/GameScreen.kt`

---

### 2.2 Reduce Card Grid Spacing
**Problem**: Gap between cards in 2x2 grid adds to height.

**Solution**: Use smaller `Arrangement.spacedBy()` value on compact screens.

**Files to modify**: `ui/component/RoomDisplay.kt`

---

## Phase 3: Layout Restructuring (Higher Effort)

### 3.1 Collapsible/Minimal Preview Panel
**Problem**: Preview panel takes vertical space even when empty.

**Solution options**:
- Hide completely when empty (show only when cards selected)
- Reduce to single line when empty
- Make it expandable/collapsible

**Files to modify**: `ui/component/PreviewPanel.kt`, `ui/screen/game/GameScreen.kt`

---

### 3.2 More Compact Status Bar
**Problem**: Status bar with 2 rows + weapon info takes significant space.

**Solution options**:
- Single row with smaller text for very compact screens
- Collapsible weapon info (tap to expand)
- Move some stats to a modal/overlay

**Files to modify**: `ui/component/GameStatusBar.kt`

---

### 3.3 Sticky Action Buttons
**Problem**: Buttons scroll off screen.

**Solution**: Pin action buttons to bottom of screen, let content scroll above.

**Implementation**:
```kotlin
Column {
    // Scrollable content
    Column(Modifier.weight(1f).verticalScroll(...)) {
        StatusBar()
        Cards()
        Preview()
    }
    // Fixed at bottom
    ActionButtons()
}
```

**Files to modify**: `ui/screen/game/GameScreen.kt`

---

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Shorten button text | Low | Medium |
| 2 | Fix "Draw Room" visibility | Low | Medium |
| 3 | Smaller cards on compact | Medium | High |
| 4 | Reduce vertical spacing | Low | Medium |
| 5 | Collapsible preview panel | Medium | Medium |
| 6 | Sticky action buttons | Medium | High |
| 7 | Compact status bar | Medium | Medium |

**Recommended approach**: Start with 1-4 (quick wins), evaluate if 5-7 are still needed.

---

## Testing

- Test on small phone emulator (e.g., Pixel 4a, 5.8" screen)
- Test on fold cover screen (Pixel Fold outer display)
- Verify touch targets remain at least 48dp
- Ensure text remains readable at smaller sizes

---

## Device Reference

| Device | Screen Height | Category |
|--------|---------------|----------|
| Pixel 4a | 2340px / ~760dp | Compact |
| Pixel Fold (cover) | 2092px / ~680dp | Compact |
| Pixel 9 Pro Fold (cover) | Similar | Compact |
| Pixel 9 Pro Fold (inner) | Large | Expanded |

---

**Created**: 2026-01-10
**Status**: Ready for implementation
