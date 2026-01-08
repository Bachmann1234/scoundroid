package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
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
    onClick: (() -> Unit)? = null,
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

    Card(
        modifier =
            modifier
                .size(width = 100.dp, height = 140.dp),
        colors =
            CardDefaults.cardColors(
                containerColor = backgroundColor,
            ),
        border = BorderStroke(2.dp, borderColor),
        onClick = onClick ?: {},
    ) {
        Column(
            modifier =
                Modifier
                    .fillMaxWidth()
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

            // Weapon (Diamonds)
            CardView(card = Card(Suit.DIAMONDS, Rank.FIVE))

            // Potion (Hearts)
            CardView(card = Card(Suit.HEARTS, Rank.TEN))
        }
    }
}
