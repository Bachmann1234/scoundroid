package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

/**
 * Displays a playing card with suit symbol, rank, and type-specific styling.
 *
 * Card colors:
 * - Monsters (♣ ♠): Red background with black text
 * - Weapons (♦): Blue background with white text
 * - Potions (♥): Green background with white text
 */
@Composable
fun CardView(
    card: Card,
    modifier: Modifier = Modifier,
    isSelected: Boolean = false,
    selectionOrder: Int? = null,
    onClick: (() -> Unit)? = null,
    cardWidth: Dp = 100.dp,
    cardHeight: Dp = 140.dp,
) {
    val (backgroundColor, textColor, borderColor) =
        when (card.type) {
            // Red
            CardType.MONSTER ->
                Triple(
                    Color(0xFFE57373),
                    Color.Black,
                    Color(0xFFD32F2F),
                )
            // Blue
            CardType.WEAPON ->
                Triple(
                    Color(0xFF64B5F6),
                    Color.White,
                    Color(0xFF1976D2),
                )
            // Green
            CardType.POTION ->
                Triple(
                    Color(0xFF81C784),
                    Color.White,
                    Color(0xFF388E3C),
                )
        }

    val typeName =
        when (card.type) {
            CardType.MONSTER -> "Monster"
            CardType.WEAPON -> "Weapon"
            CardType.POTION -> "Potion"
        }
    val suitName =
        when (card.suit) {
            Suit.CLUBS -> "Clubs"
            Suit.SPADES -> "Spades"
            Suit.DIAMONDS -> "Diamonds"
            Suit.HEARTS -> "Hearts"
        }
    val selectedText =
        if (isSelected) {
            if (selectionOrder != null) ", selected $selectionOrder of 3" else ", selected"
        } else {
            ""
        }
    val accessibilityDescription =
        "$typeName card, ${card.rank.displayName} of $suitName, value ${card.value}$selectedText"

    val actualBorderWidth = if (isSelected) 4.dp else 2.dp
    val actualBorderColor = if (isSelected) Color(0xFFFFD700) else borderColor

    Box(modifier = modifier) {
        Card(
            modifier =
                Modifier
                    .size(width = cardWidth, height = cardHeight)
                    .semantics { contentDescription = accessibilityDescription },
            colors =
                CardDefaults.cardColors(
                    containerColor = backgroundColor,
                ),
            border = BorderStroke(actualBorderWidth, actualBorderColor),
            onClick = onClick ?: {},
        ) {
            Column(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(12.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.SpaceBetween,
            ) {
                // Suit symbol
                Text(
                    text = card.suit.symbol,
                    style = MaterialTheme.typography.displayMedium,
                    color = textColor,
                )

                // Rank
                Text(
                    text = card.rank.displayName,
                    style = MaterialTheme.typography.headlineLarge,
                    fontWeight = FontWeight.Bold,
                    color = textColor,
                )

                // Value
                Text(
                    text = "${card.value}",
                    style = MaterialTheme.typography.bodyLarge,
                    color = textColor,
                )
            }
        }

        // Selection order badge
        if (selectionOrder != null) {
            Box(
                modifier =
                    Modifier
                        .align(Alignment.TopEnd)
                        .offset(x = 4.dp, y = (-4).dp)
                        .size(24.dp)
                        .background(
                            color = Color(0xFFFFD700),
                            shape = CircleShape,
                        ),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "$selectionOrder",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black,
                )
            }
        }
    }
}

/**
 * Displays an empty placeholder card to reserve space in the layout.
 * Used when no cards have been drawn yet to prevent UI shifting.
 */
@Composable
fun PlaceholderCardView(
    modifier: Modifier = Modifier,
    cardWidth: Dp = 100.dp,
    cardHeight: Dp = 140.dp,
) {
    Card(
        modifier =
            modifier
                .size(width = cardWidth, height = cardHeight)
                .semantics { contentDescription = "Empty card slot" },
        colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f),
            ),
        border = BorderStroke(2.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)),
    ) {
        Box(
            modifier = Modifier.fillMaxWidth().padding(12.dp),
            contentAlignment = Alignment.Center,
        ) {
            Text(
                text = "?",
                style = MaterialTheme.typography.displayMedium,
                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.5f),
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun CardViewPreview() {
    ScoundroidTheme {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            // Monster (Clubs)
            CardView(card = Card(Suit.CLUBS, Rank.QUEEN))

            // Weapon (Diamonds) - selected
            CardView(card = Card(Suit.DIAMONDS, Rank.FIVE), isSelected = true)

            // Potion (Hearts)
            CardView(card = Card(Suit.HEARTS, Rank.TEN))

            // Placeholder
            PlaceholderCardView()
        }
    }
}
