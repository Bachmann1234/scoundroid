package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
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
    selectedCards: Set<Card>,
    onCardClick: ((Card) -> Unit)?,
    modifier: Modifier = Modifier,
    isExpanded: Boolean = false,
    showPlaceholders: Boolean = false,
) {
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text =
                when {
                    cards.isEmpty() && showPlaceholders -> "Draw a Room"
                    cards.size == 1 -> "Leftover Card"
                    else -> "Current Room (${cards.size} cards)"
                },
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
        )

        if (cards.size == 4) {
            Text(
                text = "Select 3 cards to process (leave 1 for next room)",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
            )
        }

        // Display cards - 1x4 row for expanded mode, 2x2 grid for compact mode
        if (cards.isEmpty() && showPlaceholders) {
            // Show placeholder cards when no room drawn yet
            if (isExpanded) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterHorizontally),
                ) {
                    repeat(4) {
                        PlaceholderCardView(
                            cardWidth = 120.dp,
                            cardHeight = 168.dp,
                        )
                    }
                }
            } else {
                Column(
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally),
                    ) {
                        PlaceholderCardView()
                        PlaceholderCardView()
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally),
                    ) {
                        PlaceholderCardView()
                        PlaceholderCardView()
                    }
                }
            }
        } else if (cards.size == 4) {
            if (isExpanded) {
                // Expanded mode: all 4 cards in a single horizontal row
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterHorizontally),
                ) {
                    cards.forEach { card ->
                        CardView(
                            card = card,
                            isSelected = card in selectedCards,
                            onClick = onCardClick?.let { { it(card) } },
                            cardWidth = 120.dp,
                            cardHeight = 168.dp,
                        )
                    }
                }
            } else {
                // Compact mode: 2x2 grid
                Column(
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally),
                    ) {
                        CardView(
                            card = cards[0],
                            isSelected = cards[0] in selectedCards,
                            onClick = onCardClick?.let { { it(cards[0]) } },
                        )
                        CardView(
                            card = cards[1],
                            isSelected = cards[1] in selectedCards,
                            onClick = onCardClick?.let { { it(cards[1]) } },
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally),
                    ) {
                        CardView(
                            card = cards[2],
                            isSelected = cards[2] in selectedCards,
                            onClick = onCardClick?.let { { it(cards[2]) } },
                        )
                        CardView(
                            card = cards[3],
                            isSelected = cards[3] in selectedCards,
                            onClick = onCardClick?.let { { it(cards[3]) } },
                        )
                    }
                }
            }
        } else {
            // Single card or other layouts
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp, Alignment.CenterHorizontally),
            ) {
                cards.forEach { card ->
                    CardView(
                        card = card,
                        isSelected = card in selectedCards,
                        onClick = onCardClick?.let { { it(card) } },
                        cardWidth = if (isExpanded) 120.dp else 100.dp,
                        cardHeight = if (isExpanded) 168.dp else 140.dp,
                    )
                }
            }
        }

        if (selectedCards.isNotEmpty()) {
            Text(
                text = "Selected: ${selectedCards.size} / 3",
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )
        }
    }
}

@Preview(showBackground = true)
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
                selectedCards = setOf(roomCards[0], roomCards[2]),
                onCardClick = {},
            )

            // 1-card remaining
            RoomDisplay(
                cards =
                    listOf(
                        Card(Suit.CLUBS, Rank.ACE),
                    ),
                selectedCards = emptySet(),
                onCardClick = {},
            )

            // Placeholder state
            RoomDisplay(
                cards = emptyList(),
                selectedCards = emptySet(),
                onCardClick = null,
                showPlaceholders = true,
            )
        }
    }
}
