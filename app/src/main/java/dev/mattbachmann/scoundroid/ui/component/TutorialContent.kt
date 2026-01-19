package dev.mattbachmann.scoundroid.ui.component

import android.content.res.Configuration
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
import androidx.compose.ui.platform.LocalConfiguration
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
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
    val isTablet = configuration.smallestScreenWidthDp >= 600

    Column(
        modifier =
            modifier
                .fillMaxWidth()
                .padding(if (isLandscape) 8.dp else 16.dp),
    ) {
        // Progress dots
        Row(
            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(bottom = if (isLandscape) 4.dp else 16.dp),
            horizontalArrangement = Arrangement.Center,
        ) {
            repeat(TOTAL_PAGES) { index ->
                Box(
                    modifier =
                        Modifier
                            .padding(horizontal = 3.dp)
                            .size(if (isLandscape) 6.dp else 8.dp)
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
                0 -> GoalSlide(isLandscape, isTablet)
                1 -> CardTypesSlide(isLandscape, isTablet)
                2 -> RoomsSlide(isLandscape, isTablet)
                3 -> AvoidRoomSlide(isLandscape, isTablet)
                4 -> PotionsSlide(isLandscape, isTablet)
                5 -> WeaponsSlide(isLandscape, isTablet)
                6 -> CombatSlide(isLandscape, isTablet)
                7 -> WeaponDegradationSlide(isLandscape, isTablet)
                8 -> ScoringSlide(isLandscape, isTablet)
            }
        }

        Spacer(modifier = Modifier.height(if (isLandscape) 4.dp else 16.dp))

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

        Spacer(modifier = Modifier.height(if (isLandscape) 4.dp else 16.dp))
    }
}

/**
 * Slide 1: Goal - Survive the dungeon
 */
@Composable
private fun GoalSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    val cardWidth = if (isTablet) 90.dp else 60.dp
    val cardHeight = if (isTablet) 126.dp else 84.dp
    val cardOverlap = if (isTablet) (-45).dp else (-30).dp

    if (isLandscape) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Left: Health
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "20",
                    style = MaterialTheme.typography.displayMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFFE57373),
                )
                Text(
                    text = "Health",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // Center: Goal text
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "Goal",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Survive the dungeon",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    textAlign = TextAlign.Center,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Based on Scoundrel by Zach Gage & Kurt Bieg",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // Right: Deck
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Row(horizontalArrangement = Arrangement.spacedBy((-25).dp)) {
                    repeat(3) {
                        PlaceholderCardView(cardWidth = 45.dp, cardHeight = 63.dp)
                    }
                }
                Text(
                    text = "44 Cards",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
    } else {
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
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineMedium
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )

            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))

            Text(
                text = "20",
                style = MaterialTheme.typography.displayLarge,
                fontWeight = FontWeight.Bold,
                color = Color(0xFFE57373),
            )
            Text(
                text = "Health",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleLarge
                    } else {
                        MaterialTheme.typography.titleMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(cardOverlap)) {
                repeat(3) {
                    PlaceholderCardView(cardWidth = cardWidth, cardHeight = cardHeight)
                }
            }
            Text(
                text = "44 Cards",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleLarge
                    } else {
                        MaterialTheme.typography.titleMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 8.dp),
            )

            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))

            Text(
                text = "Survive the dungeon",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.headlineMedium
                    } else {
                        MaterialTheme.typography.titleLarge
                    },
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
            )

            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))

            Text(
                text = "Based on Scoundrel",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.bodyLarge
                    } else {
                        MaterialTheme.typography.bodySmall
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = "by Zach Gage & Kurt Bieg",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.bodyLarge
                    } else {
                        MaterialTheme.typography.bodySmall
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * Slide 2: Card Types - Monster, Weapon, Potion
 */
@Composable
private fun CardTypesSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    val cardSize =
        when {
            isLandscape -> 50.dp to 70.dp
            isTablet -> 100.dp to 140.dp
            else -> 70.dp to 98.dp
        }

    Row(
        modifier = Modifier.fillMaxSize(),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (!isLandscape) {
            Column(
                modifier = Modifier.fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
            ) {
                Text(
                    text = "Card Types",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.displaySmall
                        } else {
                            MaterialTheme.typography.headlineMedium
                        },
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
                CardTypesRow(cardSize.first, cardSize.second, isTablet)
            }
        } else {
            Text(
                text = "Card Types",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.padding(end = 24.dp),
            )
            CardTypesRow(cardSize.first, cardSize.second, isTablet = false)
        }
    }
}

@Composable
private fun CardTypesRow(
    cardWidth: androidx.compose.ui.unit.Dp,
    cardHeight: androidx.compose.ui.unit.Dp,
    isTablet: Boolean,
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(if (isTablet) 24.dp else 16.dp),
        verticalAlignment = Alignment.Top,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CardView(card = Card(Suit.SPADES, Rank.KING), cardWidth = cardWidth, cardHeight = cardHeight)
            Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
            Text(
                "Monster",
                style = if (isTablet) MaterialTheme.typography.titleMedium else MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color = Color(0xFFE57373),
            )
            Text(
                "Deals damage",
                style = if (isTablet) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CardView(card = Card(Suit.DIAMONDS, Rank.SEVEN), cardWidth = cardWidth, cardHeight = cardHeight)
            Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
            Text(
                "Weapon",
                style = if (isTablet) MaterialTheme.typography.titleMedium else MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color = Color(0xFF64B5F6),
            )
            Text(
                "Blocks damage",
                style = if (isTablet) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CardView(card = Card(Suit.HEARTS, Rank.FIVE), cardWidth = cardWidth, cardHeight = cardHeight)
            Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
            Text(
                "Potion",
                style = if (isTablet) MaterialTheme.typography.titleMedium else MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold,
                color = Color(0xFF81C784),
            )
            Text(
                "Heals you",
                style = if (isTablet) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * Slide 3: Rooms - Draw 4, choose 3
 */
@Composable
private fun RoomsSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    val cardSize =
        when {
            isLandscape -> 45.dp to 63.dp
            isTablet -> 90.dp to 126.dp
            else -> 60.dp to 84.dp
        }

    if (isLandscape) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = "Rooms",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("Draw 4 cards", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(
                    "Choose 3 to process",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Text(
                    "4th stays for next room",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                CardView(
                    card = Card(Suit.CLUBS, Rank.EIGHT),
                    cardWidth = cardSize.first,
                    cardHeight = cardSize.second,
                    isSelected = true,
                    selectionOrder = 1,
                )
                CardView(
                    card = Card(Suit.DIAMONDS, Rank.FOUR),
                    cardWidth = cardSize.first,
                    cardHeight = cardSize.second,
                    isSelected = true,
                    selectionOrder = 2,
                )
                CardView(
                    card = Card(Suit.HEARTS, Rank.SIX),
                    cardWidth = cardSize.first,
                    cardHeight = cardSize.second,
                    isSelected = true,
                    selectionOrder = 3,
                )
                CardView(card = Card(Suit.SPADES, Rank.JACK), cardWidth = cardSize.first, cardHeight = cardSize.second)
            }
        }
    } else {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                text = "Rooms",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineMedium
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(if (isTablet) 12.dp else 8.dp)) {
                CardView(
                    card = Card(Suit.CLUBS, Rank.EIGHT),
                    cardWidth = cardSize.first,
                    cardHeight = cardSize.second,
                    isSelected = true,
                    selectionOrder = 1,
                )
                CardView(
                    card = Card(Suit.DIAMONDS, Rank.FOUR),
                    cardWidth = cardSize.first,
                    cardHeight = cardSize.second,
                    isSelected = true,
                    selectionOrder = 2,
                )
                CardView(
                    card = Card(Suit.HEARTS, Rank.SIX),
                    cardWidth = cardSize.first,
                    cardHeight = cardSize.second,
                    isSelected = true,
                    selectionOrder = 3,
                )
                CardView(card = Card(Suit.SPADES, Rank.JACK), cardWidth = cardSize.first, cardHeight = cardSize.second)
            }
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Text(
                "Draw 4 cards",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.headlineSmall
                    } else {
                        MaterialTheme.typography.titleLarge
                    },
                fontWeight = FontWeight.SemiBold,
            )
            Text(
                "Choose 3 to process",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleLarge
                    } else {
                        MaterialTheme.typography.titleMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "The 4th stays for next room",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.bodyLarge
                    } else {
                        MaterialTheme.typography.bodyMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * Slide 4: Avoid Room - Skip a room, but not twice in a row
 */
@Composable
private fun AvoidRoomSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    val cardSize =
        when {
            isLandscape -> 40.dp to 56.dp
            isTablet -> 75.dp to 105.dp
            else -> 50.dp to 70.dp
        }

    if (isLandscape) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    "Avoid Room",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text("Skip a bad room", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(
                    "Cards go to bottom of deck",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "Cannot skip twice in a row!",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(2.dp)) {
                    CardView(
                        card = Card(Suit.CLUBS, Rank.ACE),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    CardView(
                        card = Card(Suit.SPADES, Rank.KING),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    CardView(
                        card = Card(Suit.CLUBS, Rank.QUEEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    CardView(
                        card = Card(Suit.SPADES, Rank.JACK),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                }
                Text(
                    "\u2192",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Row(horizontalArrangement = Arrangement.spacedBy((-20).dp)) {
                        repeat(3) { PlaceholderCardView(cardWidth = cardSize.first, cardHeight = cardSize.second) }
                    }
                    Text(
                        "Deck",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    } else {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                "Avoid Room",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineMedium
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(if (isTablet) 8.dp else 4.dp)) {
                CardView(card = Card(Suit.CLUBS, Rank.ACE), cardWidth = cardSize.first, cardHeight = cardSize.second)
                CardView(card = Card(Suit.SPADES, Rank.KING), cardWidth = cardSize.first, cardHeight = cardSize.second)
                CardView(card = Card(Suit.CLUBS, Rank.QUEEN), cardWidth = cardSize.first, cardHeight = cardSize.second)
                CardView(card = Card(Suit.SPADES, Rank.JACK), cardWidth = cardSize.first, cardHeight = cardSize.second)
            }
            Spacer(modifier = Modifier.height(if (isTablet) 12.dp else 8.dp))
            Text(
                "\u2193",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineLarge
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 12.dp else 8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(if (isTablet) (-45).dp else (-30).dp)) {
                repeat(3) { PlaceholderCardView(cardWidth = cardSize.first, cardHeight = cardSize.second) }
            }
            Text(
                "Bottom of deck",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleMedium
                    } else {
                        MaterialTheme.typography.labelMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = if (isTablet) 8.dp else 4.dp),
            )
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Text(
                "Skip a bad room",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleLarge
                    } else {
                        MaterialTheme.typography.titleMedium
                    },
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "All 4 cards go to bottom of deck",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.bodyLarge
                    } else {
                        MaterialTheme.typography.bodyMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                "Cannot skip twice in a row!",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleMedium
                    } else {
                        MaterialTheme.typography.titleSmall
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.error,
                textAlign = TextAlign.Center,
            )
        }
    }
}

/**
 * Slide 5: Potions - Only 1 per room, max health 20
 */
@Composable
private fun PotionsSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    val cardSize =
        when {
            isLandscape -> 50.dp to 70.dp
            isTablet -> 100.dp to 140.dp
            else -> 70.dp to 98.dp
        }

    if (isLandscape) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    "Potions",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "Only 1 potion works per room",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "Max health is 20",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.HEARTS, Rank.SEVEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text(
                        "+7 health",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF81C784),
                    )
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.HEARTS, Rank.FOUR),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text(
                        "No effect",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textDecoration = TextDecoration.LineThrough,
                    )
                }
            }
        }
    } else {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                "Potions",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineMedium
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(if (isTablet) 40.dp else 24.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.HEARTS, Rank.SEVEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                    Text(
                        "+7 health",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleMedium
                            } else {
                                MaterialTheme.typography.labelLarge
                            },
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF81C784),
                    )
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.HEARTS, Rank.FOUR),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                    Text(
                        "No effect",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleMedium
                            } else {
                                MaterialTheme.typography.labelLarge
                            },
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textDecoration = TextDecoration.LineThrough,
                    )
                }
            }
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Text(
                "Only 1 potion works per room",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleLarge
                    } else {
                        MaterialTheme.typography.titleMedium
                    },
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                "Max health is 20",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.bodyLarge
                    } else {
                        MaterialTheme.typography.bodyMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
    }
}

/**
 * Slide 6: Weapons - Must equip, replaces old weapon
 */
@Composable
private fun WeaponsSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    val cardSize =
        when {
            isLandscape -> 45.dp to 63.dp
            isTablet -> 90.dp to 126.dp
            else -> 60.dp to 84.dp
        }
    val newCardSize =
        when {
            isLandscape -> 50.dp to 70.dp
            isTablet -> 100.dp to 140.dp
            else -> 70.dp to 98.dp
        }

    if (isLandscape) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    "Weapons",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "Must equip new weapons",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "Replaces your current weapon",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.DIAMONDS, Rank.THREE),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text(
                        "Old",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textDecoration = TextDecoration.LineThrough,
                    )
                }
                Text(
                    "\u2192",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.DIAMONDS, Rank.EIGHT),
                        cardWidth = newCardSize.first,
                        cardHeight = newCardSize.second,
                    )
                    Text(
                        "Equipped!",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF64B5F6),
                    )
                }
            }
        }
    } else {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                "Weapons",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineMedium
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(if (isTablet) 24.dp else 16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.DIAMONDS, Rank.THREE),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                    Text(
                        "Old",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleSmall
                            } else {
                                MaterialTheme.typography.labelMedium
                            },
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textDecoration = TextDecoration.LineThrough,
                    )
                }
                Text(
                    "\u2192",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.displaySmall
                        } else {
                            MaterialTheme.typography.headlineMedium
                        },
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.DIAMONDS, Rank.EIGHT),
                        cardWidth = newCardSize.first,
                        cardHeight = newCardSize.second,
                    )
                    Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                    Text(
                        "Equipped!",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleMedium
                            } else {
                                MaterialTheme.typography.labelLarge
                            },
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF64B5F6),
                    )
                }
            }
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Text(
                "Must equip new weapons",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleLarge
                    } else {
                        MaterialTheme.typography.titleMedium
                    },
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "Replaces your current weapon",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.bodyLarge
                    } else {
                        MaterialTheme.typography.bodyMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
    }
}

/**
 * Slide 7: Combat - Weapon reduces damage
 */
@Composable
private fun CombatSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    val cardSize =
        when {
            isLandscape -> 45.dp to 63.dp
            isTablet -> 100.dp to 140.dp
            else -> 70.dp to 98.dp
        }

    if (isLandscape) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    "Combat",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "Weapon blocks damage",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "Or fight barehanded",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.CLUBS, Rank.QUEEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text("12", style = MaterialTheme.typography.labelMedium, color = Color(0xFFE57373))
                }
                Text("-", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.DIAMONDS, Rank.SEVEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text("7", style = MaterialTheme.typography.labelMedium, color = Color(0xFF64B5F6))
                }
                Text("=", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        "5",
                        style = MaterialTheme.typography.displaySmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFE57373),
                    )
                    Text(
                        "damage",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    } else {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                "Combat",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineMedium
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(if (isTablet) 24.dp else 16.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.CLUBS, Rank.QUEEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text(
                        "12 damage",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleMedium
                            } else {
                                MaterialTheme.typography.labelLarge
                            },
                        color = Color(0xFFE57373),
                        modifier = Modifier.padding(top = if (isTablet) 8.dp else 4.dp),
                    )
                }
                Text(
                    "-",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.displaySmall
                        } else {
                            MaterialTheme.typography.headlineMedium
                        },
                    fontWeight = FontWeight.Bold,
                )
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.DIAMONDS, Rank.SEVEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text(
                        "7 blocked",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleMedium
                            } else {
                                MaterialTheme.typography.labelLarge
                            },
                        color = Color(0xFF64B5F6),
                        modifier = Modifier.padding(top = if (isTablet) 8.dp else 4.dp),
                    )
                }
                Text(
                    "=",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.displaySmall
                        } else {
                            MaterialTheme.typography.headlineMedium
                        },
                    fontWeight = FontWeight.Bold,
                )
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        "5",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.displayLarge
                            } else {
                                MaterialTheme.typography.displayMedium
                            },
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFE57373),
                    )
                    Text(
                        "damage",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleMedium
                            } else {
                                MaterialTheme.typography.labelLarge
                            },
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
            Spacer(modifier = Modifier.height(if (isTablet) 32.dp else 24.dp))
            Text(
                "Weapon blocks monster damage",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleLarge
                    } else {
                        MaterialTheme.typography.titleMedium
                    },
                textAlign = TextAlign.Center,
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                "Or fight barehanded (full damage)",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.bodyLarge
                    } else {
                        MaterialTheme.typography.bodyMedium
                    },
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center,
            )
        }
    }
}

/**
 * Slide 8: Weapon Degradation - Can only fight weaker monsters
 */
@Composable
private fun WeaponDegradationSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    val cardSize =
        when {
            isLandscape -> 40.dp to 56.dp
            isTablet -> 80.dp to 112.dp
            else -> 55.dp to 77.dp
        }
    val smallCardSize =
        when {
            isLandscape -> 35.dp to 49.dp
            isTablet -> 75.dp to 105.dp
            else -> 50.dp to 70.dp
        }

    if (isLandscape) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    "Weapon Wear",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    "After defeating a monster,",
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "weapon limit = monster value",
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.SemiBold,
                )
            }
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CardView(
                            card = Card(Suit.DIAMONDS, Rank.SEVEN),
                            cardWidth = cardSize.first,
                            cardHeight = cardSize.second,
                        )
                        Text("7", style = MaterialTheme.typography.labelSmall, color = Color(0xFF64B5F6))
                    }
                    Text(
                        "vs",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CardView(
                            card = Card(Suit.CLUBS, Rank.QUEEN),
                            cardWidth = cardSize.first,
                            cardHeight = cardSize.second,
                        )
                        Text("12", style = MaterialTheme.typography.labelSmall, color = Color(0xFFE57373))
                    }
                }
                Text(
                    "Limit now 12",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.Top) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.SPADES, Rank.SIX),
                        cardWidth = smallCardSize.first,
                        cardHeight = smallCardSize.second,
                    )
                    Text(
                        "Can use",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF81C784),
                    )
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.CLUBS, Rank.KING),
                        cardWidth = smallCardSize.first,
                        cardHeight = smallCardSize.second,
                    )
                    Text(
                        "Barehanded",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFE57373),
                    )
                }
            }
        }
    } else {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                "Weapon Wear",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineMedium
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 24.dp else 16.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(if (isTablet) 16.dp else 8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.DIAMONDS, Rank.SEVEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text(
                        "7",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleMedium
                            } else {
                                MaterialTheme.typography.labelMedium
                            },
                        color = Color(0xFF64B5F6),
                    )
                }
                Text(
                    "vs",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.titleLarge
                        } else {
                            MaterialTheme.typography.titleMedium
                        },
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.CLUBS, Rank.QUEEN),
                        cardWidth = cardSize.first,
                        cardHeight = cardSize.second,
                    )
                    Text(
                        "12",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.titleMedium
                            } else {
                                MaterialTheme.typography.labelMedium
                            },
                        color = Color(0xFFE57373),
                    )
                }
            }
            Spacer(modifier = Modifier.height(if (isTablet) 16.dp else 12.dp))
            Text(
                "After defeating Q, weapon limit = 12",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.bodyLarge
                    } else {
                        MaterialTheme.typography.bodyMedium
                    },
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 24.dp else 16.dp))
            Row(
                horizontalArrangement = Arrangement.spacedBy(if (isTablet) 40.dp else 24.dp),
                verticalAlignment = Alignment.Top,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.SPADES, Rank.SIX),
                        cardWidth = smallCardSize.first,
                        cardHeight = smallCardSize.second,
                    )
                    Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                    Text(
                        "Can use",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.labelLarge
                            } else {
                                MaterialTheme.typography.labelSmall
                            },
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF81C784),
                    )
                }
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CardView(
                        card = Card(Suit.CLUBS, Rank.KING),
                        cardWidth = smallCardSize.first,
                        cardHeight = smallCardSize.second,
                    )
                    Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                    Text(
                        "Barehanded",
                        style =
                            if (isTablet) {
                                MaterialTheme.typography.labelLarge
                            } else {
                                MaterialTheme.typography.labelSmall
                            },
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFFE57373),
                    )
                }
            }
            Spacer(modifier = Modifier.height(if (isTablet) 24.dp else 16.dp))
            Text(
                "Weapons can only fight monsters",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleMedium
                    } else {
                        MaterialTheme.typography.titleSmall
                    },
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
            )
            Text(
                "equal or weaker than the last defeated",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.titleMedium
                    } else {
                        MaterialTheme.typography.titleSmall
                    },
                fontWeight = FontWeight.SemiBold,
                textAlign = TextAlign.Center,
            )
        }
    }
}

/**
 * Slide 9: Scoring - Win/Lose conditions
 */
@Composable
private fun ScoringSlide(
    isLandscape: Boolean,
    isTablet: Boolean,
) {
    if (isLandscape) {
        Row(
            modifier = Modifier.fillMaxSize(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "Scoring",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )

            // Victory
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier =
                    Modifier
                        .background(
                            color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f),
                            shape = MaterialTheme.shapes.medium,
                        ).padding(8.dp),
            ) {
                Text(
                    "Victory",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Text("Score = Health", style = MaterialTheme.typography.bodySmall)
            }

            // Bonus
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier =
                    Modifier
                        .background(
                            color = Color(0xFF81C784).copy(alpha = 0.2f),
                            shape = MaterialTheme.shapes.medium,
                        ).padding(8.dp),
            ) {
                Text(
                    "Bonus",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF388E3C),
                )
                Text("20 HP + potion?", style = MaterialTheme.typography.bodySmall)
                Text(
                    "20 + potion value",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // Defeat
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier =
                    Modifier
                        .background(
                            color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f),
                            shape = MaterialTheme.shapes.medium,
                        ).padding(8.dp),
            ) {
                Text(
                    "Defeat",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.error,
                )
                Text("Negative score", style = MaterialTheme.typography.bodySmall)
            }
        }
    } else {
        Column(
            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                "Scoring",
                style =
                    if (isTablet) {
                        MaterialTheme.typography.displaySmall
                    } else {
                        MaterialTheme.typography.headlineMedium
                    },
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
            Spacer(modifier = Modifier.height(if (isTablet) 24.dp else 16.dp))

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier =
                    Modifier
                        .background(
                            color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f),
                            shape = MaterialTheme.shapes.medium,
                        ).padding(if (isTablet) 16.dp else 12.dp),
            ) {
                Text(
                    "Victory",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.titleLarge
                        } else {
                            MaterialTheme.typography.titleMedium
                        },
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
                Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                Text(
                    "Score = Health remaining",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.bodyLarge
                        } else {
                            MaterialTheme.typography.bodyMedium
                        },
                )
                Text(
                    "Example: 15 health = 15 points",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.bodyMedium
                        } else {
                            MaterialTheme.typography.bodySmall
                        },
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Spacer(modifier = Modifier.height(if (isTablet) 16.dp else 12.dp))

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier =
                    Modifier
                        .background(
                            color = Color(0xFF81C784).copy(alpha = 0.2f),
                            shape = MaterialTheme.shapes.medium,
                        ).padding(if (isTablet) 16.dp else 12.dp),
            ) {
                Text(
                    "Bonus",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.titleLarge
                        } else {
                            MaterialTheme.typography.titleMedium
                        },
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF388E3C),
                )
                Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                Text(
                    "20 health + last card potion?",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.bodyLarge
                        } else {
                            MaterialTheme.typography.bodyMedium
                        },
                )
                Text(
                    "Score = 20 + potion value",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.bodyMedium
                        } else {
                            MaterialTheme.typography.bodySmall
                        },
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Spacer(modifier = Modifier.height(if (isTablet) 16.dp else 12.dp))

            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier =
                    Modifier
                        .background(
                            color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f),
                            shape = MaterialTheme.shapes.medium,
                        ).padding(if (isTablet) 16.dp else 12.dp),
            ) {
                Text(
                    "Defeat",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.titleLarge
                        } else {
                            MaterialTheme.typography.titleMedium
                        },
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.error,
                )
                Spacer(modifier = Modifier.height(if (isTablet) 8.dp else 4.dp))
                Text(
                    "Score = Negative",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.bodyLarge
                        } else {
                            MaterialTheme.typography.bodyMedium
                        },
                )
                Text(
                    "(Sum of remaining monsters)",
                    style =
                        if (isTablet) {
                            MaterialTheme.typography.bodyMedium
                        } else {
                            MaterialTheme.typography.bodySmall
                        },
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
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
