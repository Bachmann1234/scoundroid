package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme
import kotlinx.coroutines.launch

private const val TOTAL_PAGES = 9

/**
 * Tutorial slideshow content with card visuals and minimal text.
 * Displays 9 slides explaining the game mechanics.
 */
@Composable
fun TutorialContent(
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val pagerState = rememberPagerState(pageCount = { TOTAL_PAGES })
    val coroutineScope = rememberCoroutineScope()
    val isLastPage = pagerState.currentPage == TOTAL_PAGES - 1

    Column(
        modifier =
            modifier
                .fillMaxWidth()
                .padding(16.dp),
    ) {
        // Progress dots
        Row(
            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
            horizontalArrangement = Arrangement.Center,
        ) {
            repeat(TOTAL_PAGES) { index ->
                Box(
                    modifier =
                        Modifier
                            .padding(horizontal = 3.dp)
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(
                                if (index == pagerState.currentPage) {
                                    MaterialTheme.colorScheme.primary
                                } else {
                                    MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                                },
                            ),
                )
            }
        }

        // Pager content
        HorizontalPager(
            state = pagerState,
            modifier =
                Modifier
                    .weight(1f)
                    .fillMaxWidth(),
        ) { page ->
            when (page) {
                0 -> GoalSlide()
                1 -> CardTypesSlide()
                2 -> RoomsSlide()
                3 -> AvoidRoomSlide()
                4 -> PotionsSlide()
                5 -> WeaponsSlide()
                6 -> CombatSlide()
                7 -> WeaponDegradationSlide()
                8 -> ScoringSlide()
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Navigation buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Back button
            if (pagerState.currentPage > 0) {
                IconButton(
                    onClick = {
                        coroutineScope.launch {
                            pagerState.animateScrollToPage(pagerState.currentPage - 1)
                        }
                    },
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Previous",
                        tint = MaterialTheme.colorScheme.primary,
                    )
                }
            } else {
                Spacer(modifier = Modifier.size(48.dp))
            }

            // Skip / Done button
            if (isLastPage) {
                Button(onClick = onDismiss) {
                    Text("Done")
                }
            } else {
                TextButton(onClick = onDismiss) {
                    Text("Skip")
                }
            }

            // Next button
            if (!isLastPage) {
                IconButton(
                    onClick = {
                        coroutineScope.launch {
                            pagerState.animateScrollToPage(pagerState.currentPage + 1)
                        }
                    },
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                        contentDescription = "Next",
                        tint = MaterialTheme.colorScheme.primary,
                    )
                }
            } else {
                Spacer(modifier = Modifier.size(48.dp))
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

/**
 * Slide 1: Goal - Survive the dungeon
 */
@Composable
private fun GoalSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Goal",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Health indicator
        Text(
            text = "20",
            style = MaterialTheme.typography.displayLarge,
            fontWeight = FontWeight.Bold,
            color = Color(0xFFE57373),
        )
        Text(
            text = "Health",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Deck visual
        Row(horizontalArrangement = Arrangement.spacedBy((-30).dp)) {
            repeat(3) {
                PlaceholderCardView(cardWidth = 60.dp, cardHeight = 84.dp)
            }
        }
        Text(
            text = "44 Cards",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 8.dp),
        )

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Survive the dungeon",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Attribution
        Text(
            text = "Based on Scoundrel",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = "by Zach Gage & Kurt Bieg",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/**
 * Slide 2: Card Types - Monster, Weapon, Potion
 */
@Composable
private fun CardTypesSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Card Types",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(24.dp))

        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.Top,
        ) {
            // Monster
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.SPADES, Rank.KING),
                    cardWidth = 70.dp,
                    cardHeight = 98.dp,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Monster",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = Color(0xFFE57373),
                )
                Text(
                    text = "Deals damage",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // Weapon
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.DIAMONDS, Rank.SEVEN),
                    cardWidth = 70.dp,
                    cardHeight = 98.dp,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Weapon",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = Color(0xFF64B5F6),
                )
                Text(
                    text = "Blocks damage",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // Potion
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.HEARTS, Rank.FIVE),
                    cardWidth = 70.dp,
                    cardHeight = 98.dp,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Potion",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = Color(0xFF81C784),
                )
                Text(
                    text = "Heals you",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

/**
 * Slide 3: Rooms - Draw 4, choose 3
 */
@Composable
private fun RoomsSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Rooms",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // 4 example cards
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            CardView(
                card = Card(Suit.CLUBS, Rank.EIGHT),
                cardWidth = 60.dp,
                cardHeight = 84.dp,
                isSelected = true,
                selectionOrder = 1,
            )
            CardView(
                card = Card(Suit.DIAMONDS, Rank.FOUR),
                cardWidth = 60.dp,
                cardHeight = 84.dp,
                isSelected = true,
                selectionOrder = 2,
            )
            CardView(
                card = Card(Suit.HEARTS, Rank.SIX),
                cardWidth = 60.dp,
                cardHeight = 84.dp,
                isSelected = true,
                selectionOrder = 3,
            )
            CardView(
                card = Card(Suit.SPADES, Rank.JACK),
                cardWidth = 60.dp,
                cardHeight = 84.dp,
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Draw 4 cards",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
        )
        Text(
            text = "Choose 3 to process",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "The 4th stays for next room",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/**
 * Slide 4: Avoid Room - Skip a room, but not twice in a row
 */
@Composable
private fun AvoidRoomSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Avoid Room",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // 4 cards going to bottom of deck
        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            CardView(
                card = Card(Suit.CLUBS, Rank.ACE),
                cardWidth = 50.dp,
                cardHeight = 70.dp,
            )
            CardView(
                card = Card(Suit.SPADES, Rank.KING),
                cardWidth = 50.dp,
                cardHeight = 70.dp,
            )
            CardView(
                card = Card(Suit.CLUBS, Rank.QUEEN),
                cardWidth = 50.dp,
                cardHeight = 70.dp,
            )
            CardView(
                card = Card(Suit.SPADES, Rank.JACK),
                cardWidth = 50.dp,
                cardHeight = 70.dp,
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "\u2193",
            style = MaterialTheme.typography.headlineLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Deck visual
        Row(horizontalArrangement = Arrangement.spacedBy((-30).dp)) {
            repeat(3) {
                PlaceholderCardView(cardWidth = 50.dp, cardHeight = 70.dp)
            }
        }
        Text(
            text = "Bottom of deck",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp),
        )

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Skip a bad room",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "All 4 cards go to bottom of deck",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(12.dp))
        Text(
            text = "Cannot skip twice in a row!",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.error,
            textAlign = TextAlign.Center,
        )
    }
}

/**
 * Slide 5: Potions - Only 1 per room, max health 20
 */
@Composable
private fun PotionsSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Potions",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Two potions - one works, one crossed out
        Row(
            horizontalArrangement = Arrangement.spacedBy(24.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // First potion - works
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.HEARTS, Rank.SEVEN),
                    cardWidth = 70.dp,
                    cardHeight = 98.dp,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "+7 health",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF81C784),
                )
            }

            // Second potion - doesn't work
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Box {
                    CardView(
                        card = Card(Suit.HEARTS, Rank.FOUR),
                        cardWidth = 70.dp,
                        cardHeight = 98.dp,
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "No effect",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textDecoration = TextDecoration.LineThrough,
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Only 1 potion works per room",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Max health is 20",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

/**
 * Slide 6: Weapons - Must equip, replaces old weapon
 */
@Composable
private fun WeaponsSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Weapons",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Old weapon being replaced by new weapon
        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Old weapon
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Box {
                    CardView(
                        card = Card(Suit.DIAMONDS, Rank.THREE),
                        cardWidth = 60.dp,
                        cardHeight = 84.dp,
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Old",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textDecoration = TextDecoration.LineThrough,
                )
            }

            Text(
                text = "\u2192",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )

            // New weapon
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.DIAMONDS, Rank.EIGHT),
                    cardWidth = 70.dp,
                    cardHeight = 98.dp,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Equipped!",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF64B5F6),
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Must equip new weapons",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "Replaces your current weapon",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

/**
 * Slide 7: Combat - Weapon reduces damage
 */
@Composable
private fun CombatSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Combat",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Visual comparison: Monster vs Weapon
        Row(
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Monster
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.CLUBS, Rank.QUEEN),
                    cardWidth = 70.dp,
                    cardHeight = 98.dp,
                )
                Text(
                    text = "12 damage",
                    style = MaterialTheme.typography.labelLarge,
                    color = Color(0xFFE57373),
                    modifier = Modifier.padding(top = 4.dp),
                )
            }

            Text(
                text = "-",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )

            // Weapon
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.DIAMONDS, Rank.SEVEN),
                    cardWidth = 70.dp,
                    cardHeight = 98.dp,
                )
                Text(
                    text = "7 blocked",
                    style = MaterialTheme.typography.labelLarge,
                    color = Color(0xFF64B5F6),
                    modifier = Modifier.padding(top = 4.dp),
                )
            }

            Text(
                text = "=",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )

            // Result
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "5",
                    style = MaterialTheme.typography.displayMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFE57373),
                )
                Text(
                    text = "damage",
                    style = MaterialTheme.typography.labelLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Weapon blocks monster damage",
            style = MaterialTheme.typography.titleMedium,
            textAlign = TextAlign.Center,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = "Or fight barehanded (full damage)",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}

/**
 * Slide 8: Weapon Degradation - Can only fight weaker monsters
 */
@Composable
private fun WeaponDegradationSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Weapon Wear",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Example: Weapon used on Queen, now can only fight Queen or lower
        Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Weapon
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.DIAMONDS, Rank.SEVEN),
                    cardWidth = 55.dp,
                    cardHeight = 77.dp,
                )
                Text(
                    text = "7",
                    style = MaterialTheme.typography.labelMedium,
                    color = Color(0xFF64B5F6),
                )
            }

            Text(
                text = "vs",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            // Defeated monster (Queen)
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.CLUBS, Rank.QUEEN),
                    cardWidth = 55.dp,
                    cardHeight = 77.dp,
                )
                Text(
                    text = "12",
                    style = MaterialTheme.typography.labelMedium,
                    color = Color(0xFFE57373),
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        Text(
            text = "After defeating Q, weapon limit = 12",
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface,
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Can fight / Can't fight examples
        Row(
            horizontalArrangement = Arrangement.spacedBy(24.dp),
            verticalAlignment = Alignment.Top,
        ) {
            // Can fight - 6 or lower
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.SPADES, Rank.SIX),
                    cardWidth = 50.dp,
                    cardHeight = 70.dp,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Can use",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF81C784),
                )
            }

            // Can't fight - King is higher than 12
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CardView(
                    card = Card(Suit.CLUBS, Rank.KING),
                    cardWidth = 50.dp,
                    cardHeight = 70.dp,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Barehanded",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFE57373),
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Weapons can only fight monsters",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
        Text(
            text = "equal or weaker than the last defeated",
            style = MaterialTheme.typography.titleSmall,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center,
        )
    }
}

/**
 * Slide 9: Scoring - Win/Lose conditions
 */
@Composable
private fun ScoringSlide() {
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "Scoring",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Win condition
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier =
                Modifier
                    .background(
                        color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f),
                        shape = MaterialTheme.shapes.medium,
                    ).padding(12.dp),
        ) {
            Text(
                text = "Victory",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Score = Health remaining",
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                text = "Example: 15 health = 15 points",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Bonus condition
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier =
                Modifier
                    .background(
                        color = Color(0xFF81C784).copy(alpha = 0.2f),
                        shape = MaterialTheme.shapes.medium,
                    ).padding(12.dp),
        ) {
            Text(
                text = "Bonus",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF388E3C),
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "20 health + last card potion?",
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                text = "Score = 20 + potion value",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Lose condition
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier =
                Modifier
                    .background(
                        color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f),
                        shape = MaterialTheme.shapes.medium,
                    ).padding(12.dp),
        ) {
            Text(
                text = "Defeat",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.error,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = "Score = Negative",
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                text = "(Sum of remaining monsters)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun TutorialContentPreview() {
    ScoundroidTheme {
        TutorialContent(onDismiss = {})
    }
}
