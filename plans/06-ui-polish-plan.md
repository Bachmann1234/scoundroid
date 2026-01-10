# UI Polish Implementation Plan

This document outlines the UI/UX improvements identified for Scoundroid, organized by priority and implementation complexity.

## Overview

The current UI is functional but can be elevated with visual polish, improved consistency, and better accessibility. These changes focus on making the app feel more tactile and delightful without changing core functionality.

---

## Phase 1: Quick Wins (Low Effort, High Impact)

### 1.1 Fix "Current Room" Label Contrast
**Problem**: The "Current Room (4 cards)" label is barely visible with faded gray text.

**Solution**:
- Increase text opacity or use a more visible color from the theme
- Consider using `MaterialTheme.colorScheme.onSurface` instead of a dim variant

**Files to modify**: `ui/screen/GameScreen.kt` (or wherever room label is defined)

---

### 1.2 Unify Button Styling
**Problem**: Mixed button styles—"Avoid Room" and "New Game" are outlined, "Process" is filled.

**Solution**: Establish clear hierarchy:
- **Primary action** (Process Cards): Filled button with prominent color
- **Secondary action** (Avoid Room): Tonal button (filled but muted)
- **Tertiary action** (New Game): Outlined or text button

**Implementation**:
```kotlin
// Primary
Button(onClick = { ... }) { Text("Process 3/3 Cards") }

// Secondary
FilledTonalButton(onClick = { ... }) { Text("Avoid Room") }

// Tertiary
OutlinedButton(onClick = { ... }) { Text("New Game") }
```

**Files to modify**: `ui/screen/GameScreen.kt`, potentially `ui/component/` if buttons are componentized

---

### 1.3 Improve Stats Panel Text Hierarchy
**Problem**: Labels ("Health", "Deck") and values have similar visual weight.

**Solution**:
- Make labels smaller and/or use `colorScheme.onSurfaceVariant`
- Make values larger and/or bolder
- Consider using `labelSmall` for labels and `titleMedium` for values

**Example**:
```kotlin
Text("Health", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
Text("20 / 20", style = MaterialTheme.typography.titleMedium)
```

---

## Phase 2: Card Visual Improvements (Medium Effort)

### 2.1 Add Card Elevation & Shadows
**Problem**: Cards appear flat and don't feel tactile.

**Solution**:
- Wrap cards in a `Card` composable with elevation
- Use `Modifier.shadow()` for custom shadow effects
- Consider slight rotation or offset for a "hand of cards" feel

**Implementation**:
```kotlin
Card(
    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
    // ...
) {
    // Card content
}
```

---

### 2.2 Enhance Selection Indicator
**Problem**: Thin green border is easy to miss.

**Solution options** (choose one or combine):

**Option A - Thicker border with glow**:
```kotlin
Modifier.border(
    width = 3.dp,
    color = Color.Green,
    shape = RoundedCornerShape(8.dp)
)
```

**Option B - Lift effect (increased elevation + scale)**:
```kotlin
val scale by animateFloatAsState(if (selected) 1.05f else 1f)
val elevation by animateDpAsState(if (selected) 12.dp else 4.dp)

Card(
    modifier = Modifier.scale(scale),
    elevation = CardDefaults.cardElevation(defaultElevation = elevation)
)
```

**Option C - Background glow**:
- Add a blurred shadow layer behind selected cards using `Modifier.drawBehind`

**Recommendation**: Option B (lift effect) provides the best tactile feedback.

---

### 2.3 Improve Order Badges
**Problem**: Number badges look like notification badges rather than sequence indicators.

**Solution**:
- Reposition to bottom-right or use a ribbon/banner style
- Use a more intentional design (e.g., circled numbers with consistent styling)
- Consider connecting badges with a subtle line to show sequence

**Alternative**: Show order as overlay text on the card itself when selected

---

## Phase 3: Polish & Delight (Higher Effort)

### 3.1 Card Design Refinement
**Current**: Solid flat colors with simple suit symbols.

**Improvements**:
- Add subtle gradient to card backgrounds
- Improve suit symbol rendering (consider custom vector assets)
- Add subtle inner shadow or border for depth
- Consider a card "texture" or pattern

**Example gradient**:
```kotlin
Modifier.background(
    brush = Brush.verticalGradient(
        colors = listOf(
            cardColor.copy(alpha = 0.9f),
            cardColor
        )
    )
)
```

---

### 3.2 Add Tap Affordance to Unselected Cards
**Problem**: Cards don't clearly communicate they're interactive.

**Solution**:
- Add subtle hover/press states
- Unselected cards could have a slight "resting" shadow that increases on press
- Consider a subtle pulse animation on first room to guide new players

**Implementation**:
```kotlin
val interactionSource = remember { MutableInteractionSource() }
val isPressed by interactionSource.collectIsPressedAsState()

Card(
    interactionSource = interactionSource,
    elevation = CardDefaults.cardElevation(
        defaultElevation = 4.dp,
        pressedElevation = 2.dp
    )
)
```

---

### 3.3 Preview Panel Styling
**Problem**: Gray background feels disconnected from the rest of the design.

**Solution**:
- Use `MaterialTheme.colorScheme.surfaceVariant` for consistency
- Add subtle rounded corners and elevation
- Consider making it a collapsible/expandable section for smaller screens

---

## Phase 4: Accessibility Improvements

### 4.1 Color-Independent Card Type Identification
**Problem**: Relying on color alone for card types could challenge colorblind users.

**Solutions**:
- Suits already help, but consider adding:
  - Subtle pattern/texture per card type (crosshatch for monsters, dots for weapons, etc.)
  - Small icon in corner (sword for weapons, skull for monsters, heart for potions)
  - Different card shapes (slightly different corner radius per type)

**Recommendation**: Add small thematic icons to card corners as a secondary identifier.

---

### 4.2 Ensure Sufficient Contrast Ratios
- Audit all text against WCAG 2.1 AA standards (4.5:1 for normal text, 3:1 for large text)
- Pay special attention to:
  - Stats panel labels
  - Preview panel text
  - Button text on colored backgrounds

---

## Implementation Order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Fix "Current Room" label contrast | Low | High |
| 2 | Unify button styling | Low | Medium |
| 3 | Improve stats panel hierarchy | Low | Medium |
| 4 | Add card elevation/shadows | Medium | High |
| 5 | Enhance selection indicator | Medium | High |
| 6 | Preview panel styling | Low | Medium |
| 7 | Improve order badges | Medium | Medium |
| 8 | Add tap affordance | Medium | Medium |
| 9 | Card design refinement | High | Medium |
| 10 | Accessibility improvements | Medium | High |

---

## Testing Considerations

- Test on actual Pixel 10 Pro Fold in both folded and unfolded states
- Verify animations don't cause performance issues
- Test with Android accessibility settings (font scaling, color correction)
- Get user feedback on selection indicator visibility

---

## Notes

- All changes should maintain the current app's clean, readable aesthetic
- Avoid over-designing—polish should enhance, not distract from gameplay
- Consider making some polish features toggleable if they impact performance on older devices (though this is a personal app for a specific device)

---

**Created**: 2026-01-09
**Status**: Ready for implementation
