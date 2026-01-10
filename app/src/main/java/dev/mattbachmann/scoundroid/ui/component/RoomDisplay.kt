package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.ui.screen.game.ScreenSizeClass
import dev.mattbachmann.scoundroid.ui.theme.Purple80
import dev.mattbachmann.scoundroid.ui.theme.PurpleGrey80
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

/**
 * Configuration for card layout based on screen size.
 */
private data class CardLayoutConfig(
    val width: Dp,
    val height: Dp,
    val spacing: Dp,
    val useGrid: Boolean,
)

/**
 * Displays the current room of cards.
 * Shows 4 cards when a room is first drawn, or 1 card remaining after selection.
 * Shows placeholder cards when no cards are present.
 */
@Composable
fun RoomDisplay(
    cards: List<Card>,
    selectedCards: List<Card>,
    onCardClick: ((Card) -> Unit)?,
    modifier: Modifier = Modifier,
    screenSizeClass: ScreenSizeClass = ScreenSizeClass.MEDIUM,
    showPlaceholders: Boolean = false,
) {
    // Card sizes and layout based on screen size class
    // COMPACT: 76x106dp in 1x4 row (small phones)
    // MEDIUM: 85x119dp in 2x2 grid (fold cover, regular phones)
    // EXPANDED: 160x224dp in 1x4 row (tablets, unfolded)
    val (cardWidth, cardHeight, cardSpacing, useGridLayout) =
        when (screenSizeClass) {
            ScreenSizeClass.COMPACT -> CardLayoutConfig(76.dp, 106.dp, 4.dp, false)
            ScreenSizeClass.MEDIUM -> CardLayoutConfig(85.dp, 119.dp, 8.dp, true)
            ScreenSizeClass.EXPANDED -> CardLayoutConfig(160.dp, 224.dp, 16.dp, false)
        }
    val isExpanded = screenSizeClass == ScreenSizeClass.EXPANDED

    Column(
        modifier =
            modifier
                .fillMaxWidth()
                .padding(bottom = if (isExpanded) 12.dp else 0.dp),
        verticalArrangement = Arrangement.spacedBy(if (isExpanded) 16.dp else 8.dp),
    ) {
        Text(
            text =
                when {
                    cards.isEmpty() && showPlaceholders -> "Draw Room"
                    cards.size == 1 -> "Leftover Card"
                    else -> "Current Room"
                },
            style = if (isExpanded) MaterialTheme.typography.titleLarge else MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = Purple80,
        )

        // Only show instruction text when 4 cards present, use fixed height to prevent layout shift
        Box(modifier = Modifier.height(if (isExpanded) 20.dp else 16.dp)) {
            if (cards.size == 4) {
                Text(
                    text = "Select 3 cards to process (leave 1 for next room)",
                    style = if (isExpanded) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.bodySmall,
                    color = PurpleGrey80,
                )
            }
        }

        // Display cards - use grid for MEDIUM, row for COMPACT and EXPANDED
        // Use fixed height to prevent layout jumping between 1 card and 4 card states
        val cardAreaHeight = if (useGridLayout) cardHeight * 2 + cardSpacing else cardHeight
        Box(
            modifier =
                Modifier
                    .fillMaxWidth()
                    .height(cardAreaHeight),
            contentAlignment = Alignment.Center,
        ) {
            if (cards.isEmpty() && showPlaceholders) {
                // Show placeholder cards when no room drawn yet
                if (useGridLayout) {
                    // 2x2 grid for MEDIUM
                    Column(verticalArrangement = Arrangement.spacedBy(cardSpacing)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                        ) {
                            repeat(2) { PlaceholderCardView(cardWidth = cardWidth, cardHeight = cardHeight) }
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                        ) {
                            repeat(2) { PlaceholderCardView(cardWidth = cardWidth, cardHeight = cardHeight) }
                        }
                    }
                } else {
                    // 1x4 row for COMPACT and EXPANDED
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                    ) {
                        repeat(4) { PlaceholderCardView(cardWidth = cardWidth, cardHeight = cardHeight) }
                    }
                }
            } else if (cards.size == 4) {
                if (useGridLayout) {
                    // 2x2 grid for MEDIUM
                    Column(verticalArrangement = Arrangement.spacedBy(cardSpacing)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                        ) {
                            cards.take(2).forEach { card ->
                                val orderIndex = selectedCards.indexOf(card)
                                CardView(
                                    card = card,
                                    isSelected = card in selectedCards,
                                    selectionOrder = if (orderIndex >= 0) orderIndex + 1 else null,
                                    onClick = onCardClick?.let { { it(card) } },
                                    cardWidth = cardWidth,
                                    cardHeight = cardHeight,
                                )
                            }
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                        ) {
                            cards.drop(2).forEach { card ->
                                val orderIndex = selectedCards.indexOf(card)
                                CardView(
                                    card = card,
                                    isSelected = card in selectedCards,
                                    selectionOrder = if (orderIndex >= 0) orderIndex + 1 else null,
                                    onClick = onCardClick?.let { { it(card) } },
                                    cardWidth = cardWidth,
                                    cardHeight = cardHeight,
                                )
                            }
                        }
                    }
                } else {
                    // 1x4 row for COMPACT and EXPANDED
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                    ) {
                        cards.forEach { card ->
                            val orderIndex = selectedCards.indexOf(card)
                            CardView(
                                card = card,
                                isSelected = card in selectedCards,
                                selectionOrder = if (orderIndex >= 0) orderIndex + 1 else null,
                                onClick = onCardClick?.let { { it(card) } },
                                cardWidth = cardWidth,
                                cardHeight = cardHeight,
                            )
                        }
                    }
                }
            } else {
                // Single card or other layouts - always use row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                ) {
                    cards.forEach { card ->
                        val orderIndex = selectedCards.indexOf(card)
                        CardView(
                            card = card,
                            isSelected = card in selectedCards,
                            selectionOrder = if (orderIndex >= 0) orderIndex + 1 else null,
                            onClick = onCardClick?.let { { it(card) } },
                            cardWidth = cardWidth,
                            cardHeight = cardHeight,
                        )
                    }
                }
            }
        }

        // Only show selection count when cards are selected, use fixed height to prevent layout shift
        Box(modifier = Modifier.height(if (isExpanded) 24.dp else 16.dp)) {
            if (selectedCards.isNotEmpty()) {
                Text(
                    text = "Selected: ${selectedCards.size} / 3",
                    style = if (isExpanded) MaterialTheme.typography.bodyLarge else MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
        }
    }
}

@Preview(showBackground = true, widthDp = 400, heightDp = 800)
@Composable
fun RoomDisplayPreview() {
    val roomCards =
        listOf(
            Card(Suit.CLUBS, Rank.QUEEN),
            Card(Suit.DIAMONDS, Rank.FIVE),
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.SPADES, Rank.TEN),
        )
    ScoundroidTheme {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            // 4-card room with 2 selected
            RoomDisplay(
                cards = roomCards,
                selectedCards = listOf(roomCards[0], roomCards[2]),
                onCardClick = {},
            )

            // 1-card remaining
            RoomDisplay(
                cards =
                    listOf(
                        Card(Suit.CLUBS, Rank.ACE),
                    ),
                selectedCards = emptyList(),
                onCardClick = {},
            )

            // Placeholder state
            RoomDisplay(
                cards = emptyList(),
                selectedCards = emptyList(),
                onCardClick = null,
                showPlaceholders = true,
            )
        }
    }
}
