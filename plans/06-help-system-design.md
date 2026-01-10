# Help & Tutorial System Design

## Problem Statement

The user is not deeply familiar with Scoundrel rules, particularly:
- Weapon degradation mechanics (most complex rule)
- Card values (face cards = J:11, Q:12, K:13, A:14)
- Scoring system
- Room avoidance restrictions

**Solution**: Comprehensive in-game help system that's always accessible.

## Help System Components

### 1. First-Time Tutorial (Optional)

**Trigger**: First time app is launched, or manually from menu

**Flow**:
```
Welcome Screen
    ↓
"This is a dungeon deck of 44 cards..."
    ↓
Interactive First Room
    ↓
"Choose 3 of these 4 cards..."
    ↓
Card Type Explanation (with example)
    ↓
Combat Demonstration
    ↓
Weapon Degradation Demo (CRITICAL)
    ↓
"Ready to play!"
```

**Features**:
- Skip tutorial option
- Can replay anytime from menu
- Highlights UI elements with annotations
- Step-by-step with "Next" button
- Practice mode (doesn't count as game)

### 2. Rules Reference Screen

**Access**: Help button in menu / during game

**Content Structure**:
```
┌─────────────────────────────┐
│ Rules Reference        [X]  │
├─────────────────────────────┤
│ 📚 Search rules...          │
├─────────────────────────────┤
│ ▼ Game Setup               │
│   • 44 Card Deck            │
│   • Starting Health: 20     │
│                             │
│ ▼ Card Types               │
│   ♠♣ Monsters (26 cards)   │
│   ♦  Weapons (9 cards)     │
│   ♥  Potions (9 cards)     │
│                             │
│ ▼ Weapon Degradation ⚠️    │
│   [Detailed explanation]    │
│   [Visual examples]         │
│                             │
│ ▼ Room Mechanics           │
│ ▼ Scoring System           │
│ ▼ Tips & Strategy          │
└─────────────────────────────┘
```

**Sections**:

1. **Game Setup**
   - 44 card deck composition
   - Removed cards (red faces/aces)
   - Starting health: 20

2. **Card Types**
   - Monsters (♠♣): Damage = value
   - Weapons (♦): Reduce damage
   - Potions (♥): Restore health

3. **Card Values Table**
   ```
   2-10: Face value
   Jack (J): 11
   Queen (Q): 12
   King (K): 13
   Ace (A): 14
   ```

4. **Weapon Degradation** ⭐ MOST IMPORTANT
   - Clear explanation with visual examples
   - Interactive demo showing degradation
   - Common scenarios:
     ```
     Example 1:
     Weapon: 5♦
     Defeat: Q♣ (12) → max = 12 ✓ Can use on ≤12
     Defeat: 6♠ (6)  → max = 6  ✓ Can use on ≤6
     Face:   7♥ (7)  → ✗ Cannot use weapon!

     Example 2:
     Weapon: 10♦
     Never used → Can defeat ANY monster
     ```

5. **Room Mechanics**
   - Draw 4 cards
   - Choose 3 to face
   - Keep 1 for next room
   - Can avoid room (places 4 at bottom)
   - Cannot avoid twice in a row

6. **Combat**
   - Barehanded: Full monster damage
   - With weapon: max(0, monster - weapon)
   - Monster goes on weapon stack

7. **Health Potions**
   - Adds value to health
   - Max health: 20
   - Only 1 per turn
   - Excess discarded

8. **Scoring**
   - Win: Remaining health (or +last potion if health=20)
   - Lose: Current health - remaining monsters

9. **Tips & Strategy**
   - When to avoid rooms
   - Weapon management
   - Health conservation

### 3. Contextual In-Game Help

**Always Visible**:
- Help icon (?) in top corner
- Tapping shows quick help overlay

**Contextual Tooltips**:

**When selecting a monster with weapon equipped**:
```
┌───────────────────────────┐
│ 6♠ Monster                │
│                           │
│ Your weapon: 5♦           │
│ Weapon max: 12            │
│                           │
│ ✓ Can use weapon          │
│ Damage: 6-5 = 1 ❤️        │
│                           │
│ [Fight with Weapon]       │
│ [Fight Barehanded (6 ❤️)] │
└───────────────────────────┘
```

**When weapon can't be used**:
```
┌───────────────────────────┐
│ 8♣ Monster                │
│                           │
│ Your weapon: 5♦           │
│ Weapon max: 6             │
│                           │
│ ✗ Cannot use weapon       │
│ Monster (8) > max (6)     │
│                           │
│ ℹ️ Weapon degraded after  │
│   defeating 6♠            │
│                           │
│ [Fight Barehanded (8 ❤️)] │
└───────────────────────────┘
```

**Weapon degradation indicator**:
```
┌─────────────────────┐
│ Equipped: 5♦        │
│ Can defeat: ≤ 6     │
│ [View History]      │
└─────────────────────┘
```

### 4. Quick Reference Overlay

**Access**: Swipe up from bottom or help button

**Content**:
```
┌─────────────────────────────┐
│ 📖 Quick Reference          │
├─────────────────────────────┤
│ CARD VALUES                 │
│ 2-10: Face value            │
│ J=11  Q=12  K=13  A=14     │
│                             │
│ YOUR WEAPON                 │
│ 5♦ (Defeats ≤ 6)           │
│ History: Q♣(12) → 6♠(6)    │
│                             │
│ CURRENT STATE               │
│ Health: 15/20               │
│ Dungeon: 22 cards left      │
│ Last action: Can't avoid    │
└─────────────────────────────┘
```

### 5. Settings/Help Menu

**Menu Items**:
- View Rules Reference
- Replay Tutorial
- View Original Rules (PDF)
- About Scoundrel (credits, link to creators)
- Tips & Strategy Guide

## Implementation Priority

### Phase 3 (with UI): Basic Help
- [ ] Help button in game screen
- [ ] Basic rules reference screen
- [ ] Card values table

### Phase 5: Complete Help System
- [ ] Interactive tutorial
- [ ] Contextual tooltips
- [ ] Weapon degradation explanations
- [ ] Quick reference overlay
- [ ] In-app PDF viewer (optional)

## UI Design Considerations

### Help Button Placement
- **Always visible** during gameplay
- Top-right corner (?) icon
- Bottom sheet on tap with quick tips

### Tutorial Style
- Use actual game UI (not separate screens)
- Highlight elements with spotlight effect
- Dimmed background with highlighted interactive area
- Clear, concise text (1-2 sentences per step)
- "Skip" always available

### Rules Reference Style
- Material3 design
- Collapsible sections
- Search functionality
- Bookmark favorite sections
- Examples with card graphics

### Contextual Help Style
- Non-intrusive
- Appears when needed (first time actions)
- Can be dismissed
- "Don't show again" option for repeated tips
- Visual indicators (colors, icons)

## Content Writing Guidelines

### Tone
- Clear and concise
- Friendly but not condescending
- Assume player is intelligent but unfamiliar
- Use examples liberally

### Structure
- Short paragraphs
- Bullet points
- Visual examples
- "Try it yourself" interactive demos

### Special Focus: Weapon Degradation

This is the hardest rule to understand. Multiple approaches:

1. **Visual Timeline**
   ```
   5♦ ─── defeats ───> Q♣(12)  max=12
       ─── defeats ───> 6♠(6)   max=6 ⬇ Degraded!
       ─── cannot defeat ─X─> 7♥(7)
   ```

2. **Interactive Demo**
   Let user try weapon degradation in practice mode

3. **Real-time Feedback**
   Show why weapon can/can't be used each time

4. **History Tracker**
   Show what monsters weapon has defeated

## Testing the Help System

### Usability Tests
- [ ] User can find help within 5 seconds
- [ ] Weapon degradation explanation is clear
- [ ] Tutorial completable in < 2 minutes
- [ ] Rules searchable and scannable
- [ ] Help doesn't obstruct gameplay

### Content Tests
- [ ] All rules accurately explained
- [ ] Examples match actual game behavior
- [ ] No jargon or unclear terms
- [ ] Covers all edge cases

## Alternative: Minimalist Approach

If full tutorial is too much work initially:

**Essential Help** (MVP):
1. Card values table (always accessible)
2. Weapon state display (current max value)
3. Damage preview before combat
4. Link to PDF rules in menu

This can be implemented in Phase 3, then enhanced in Phase 5.

## Resources Needed

- **Text Content**: Rule explanations written clearly
- **Graphics**: Card illustrations for examples
- **Interactive Elements**: Tutorial overlay components
- **Rules**: Game rules accessible in-app

## Success Criteria

User playing for first time should:
- ✅ Understand basic gameplay within 2 minutes
- ✅ Understand weapon degradation after first occurrence
- ✅ Be able to find help when confused
- ✅ Never feel lost or frustrated by rules
- ✅ Not need to reference external rules document

---

**Note**: Since this is a personal app, help system can evolve based on your own experience playing. Start simple, add features as needed.
