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
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

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
    isExpanded: Boolean = false,
    showPlaceholders: Boolean = false,
) {
    // Card sizes based on mode
    val cardWidth = if (isExpanded) 160.dp else 85.dp
    val cardHeight = if (isExpanded) 224.dp else 119.dp
    val cardSpacing = if (isExpanded) 16.dp else 8.dp

    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(if (isExpanded) 16.dp else 8.dp),
    ) {
        Text(
            text =
                when {
                    cards.isEmpty() && showPlaceholders -> "Draw Room"
                    cards.size == 1 -> "Leftover Card"
                    else -> "Current Room (${cards.size} cards)"
                },
            style = if (isExpanded) MaterialTheme.typography.titleLarge else MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
        )

        // Always reserve space for instruction text to prevent layout shift
        Text(
            text = "Select 3 cards to process (leave 1 for next room)",
            style = if (isExpanded) MaterialTheme.typography.bodyMedium else MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            modifier = Modifier.alpha(if (cards.size == 4) 1f else 0f),
        )

        // Display cards - 1x4 row for expanded mode, 2x2 grid for compact mode
        // Use fixed height to prevent layout jumping between 1 card and 4 card states
        val cardAreaHeight = if (isExpanded) cardHeight else (cardHeight * 2 + cardSpacing)
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(cardAreaHeight),
            contentAlignment = Alignment.Center,
        ) {
        if (cards.isEmpty() && showPlaceholders) {
            // Show placeholder cards when no room drawn yet
            if (isExpanded) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                ) {
                    repeat(4) {
                        PlaceholderCardView(
                            cardWidth = cardWidth,
                            cardHeight = cardHeight,
                        )
                    }
                }
            } else {
                Column(
                    verticalArrangement = Arrangement.spacedBy(cardSpacing),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                    ) {
                        PlaceholderCardView(cardWidth = cardWidth, cardHeight = cardHeight)
                        PlaceholderCardView(cardWidth = cardWidth, cardHeight = cardHeight)
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                    ) {
                        PlaceholderCardView(cardWidth = cardWidth, cardHeight = cardHeight)
                        PlaceholderCardView(cardWidth = cardWidth, cardHeight = cardHeight)
                    }
                }
            }
        } else if (cards.size == 4) {
            if (isExpanded) {
                // Expanded mode: all 4 cards in a single horizontal row (larger cards)
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
            } else {
                // Compact mode: 2x2 grid
                Column(
                    verticalArrangement = Arrangement.spacedBy(cardSpacing),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                    ) {
                        val orderIndex0 = selectedCards.indexOf(cards[0])
                        CardView(
                            card = cards[0],
                            isSelected = cards[0] in selectedCards,
                            selectionOrder = if (orderIndex0 >= 0) orderIndex0 + 1 else null,
                            onClick = onCardClick?.let { { it(cards[0]) } },
                            cardWidth = cardWidth,
                            cardHeight = cardHeight,
                        )
                        val orderIndex1 = selectedCards.indexOf(cards[1])
                        CardView(
                            card = cards[1],
                            isSelected = cards[1] in selectedCards,
                            selectionOrder = if (orderIndex1 >= 0) orderIndex1 + 1 else null,
                            onClick = onCardClick?.let { { it(cards[1]) } },
                            cardWidth = cardWidth,
                            cardHeight = cardHeight,
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(cardSpacing, Alignment.CenterHorizontally),
                    ) {
                        val orderIndex2 = selectedCards.indexOf(cards[2])
                        CardView(
                            card = cards[2],
                            isSelected = cards[2] in selectedCards,
                            selectionOrder = if (orderIndex2 >= 0) orderIndex2 + 1 else null,
                            onClick = onCardClick?.let { { it(cards[2]) } },
                            cardWidth = cardWidth,
                            cardHeight = cardHeight,
                        )
                        val orderIndex3 = selectedCards.indexOf(cards[3])
                        CardView(
                            card = cards[3],
                            isSelected = cards[3] in selectedCards,
                            selectionOrder = if (orderIndex3 >= 0) orderIndex3 + 1 else null,
                            onClick = onCardClick?.let { { it(cards[3]) } },
                            cardWidth = cardWidth,
                            cardHeight = cardHeight,
                        )
                    }
                }
            }
        } else {
            // Single card or other layouts
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

        // Always reserve space for selection text to prevent layout shift
        Text(
            text = "Selected: ${selectedCards.size} / 3",
            style = if (isExpanded) MaterialTheme.typography.bodyLarge else MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.alpha(if (selectedCards.isNotEmpty()) 1f else 0f),
        )
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
