package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
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
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
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
    val actualBorderColor = if (isSelected) Color(0xFF009688) else borderColor

    // Scale animation for selection feedback
    val scale by animateFloatAsState(
        targetValue = if (isSelected) 1.08f else 1f,
        animationSpec =
            spring(
                dampingRatio = Spring.DampingRatioMediumBouncy,
                stiffness = Spring.StiffnessMedium,
            ),
        label = "cardScale",
    )

    // Elevation animation for lift effect when selected
    val elevation by animateDpAsState(
        targetValue = if (isSelected) 12.dp else 4.dp,
        animationSpec =
            spring(
                dampingRatio = Spring.DampingRatioMediumBouncy,
                stiffness = Spring.StiffnessMedium,
            ),
        label = "cardElevation",
    )

    // Scale font sizes and padding proportionally to card dimensions
    val suitFontSize = (cardWidth.value * 0.45f).sp
    val rankFontSize = (cardWidth.value * 0.32f).sp
    val cardPadding = (cardWidth.value * 0.12f).dp

    Box(modifier = modifier.scale(scale)) {
        Card(
            modifier =
                Modifier
                    .size(width = cardWidth, height = cardHeight)
                    .semantics { contentDescription = accessibilityDescription },
            colors =
                CardDefaults.cardColors(
                    containerColor = backgroundColor,
                ),
            elevation =
                CardDefaults.cardElevation(
                    defaultElevation = elevation,
                ),
            border = BorderStroke(actualBorderWidth, actualBorderColor),
            onClick = onClick ?: {},
        ) {
            Column(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(cardPadding),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.SpaceBetween,
            ) {
                // Suit symbol
                Text(
                    text = card.suit.symbol,
                    fontSize = suitFontSize,
                    color = textColor,
                )

                // Rank
                Text(
                    text = card.rank.displayName,
                    fontSize = rankFontSize,
                    fontWeight = FontWeight.Bold,
                    color = textColor,
                )
            }
        }

        // Selection order badge - positioned at bottom-right with border for visibility
        if (selectionOrder != null) {
            Box(
                modifier =
                    Modifier
                        .align(Alignment.BottomEnd)
                        .offset(x = 6.dp, y = 6.dp)
                        .size(26.dp)
                        .background(
                            color = Color.White,
                            shape = CircleShape,
                        )
                        .padding(2.dp)
                        .background(
                            color = Color(0xFF009688),
                            shape = CircleShape,
                        ),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "$selectionOrder",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                )
            }
        }
    }
}

/**
 * Displays a card back design to represent face-down cards in the deck.
 * Features a classic playing card back with a crosshatch pattern.
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
                .semantics { contentDescription = "Card back" },
        colors =
            CardDefaults.cardColors(
                // Dark red
                containerColor = Color(0xFF8B0000),
            ),
        elevation =
            CardDefaults.cardElevation(
                defaultElevation = 4.dp,
            ),
        // Cream border
        border = BorderStroke(2.dp, Color(0xFFF5F5DC)),
    ) {
        Box(
            modifier =
                Modifier
                    .fillMaxSize()
                    .padding(6.dp),
        ) {
            // Inner decorative border
            Box(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .background(Color(0xFF8B0000))
                        .padding(2.dp),
            ) {
                // Cream inner border
                Box(
                    modifier =
                        Modifier
                            .fillMaxSize()
                            .background(Color(0xFFF5F5DC))
                            .padding(2.dp),
                ) {
                    // Dark red center with cached crosshatch pattern
                    Box(
                        modifier =
                            Modifier
                                .fillMaxSize()
                                .background(Color(0xFF8B0000))
                                .drawWithCache {
                                    val patternColor = Color(0xFFCD5C5C) // Indian red for pattern
                                    val spacing = 16f
                                    val strokeWidth = 1.5f
                                    val centerX = size.width / 2
                                    val centerY = size.height / 2
                                    val ovalWidth = size.width * 0.6f
                                    val ovalHeight = size.height * 0.5f

                                    onDrawBehind {
                                        // Draw diagonal lines from top-left to bottom-right
                                        var x = -size.height
                                        while (x < size.width + size.height) {
                                            drawLine(
                                                color = patternColor,
                                                start = Offset(x, 0f),
                                                end = Offset(x + size.height, size.height),
                                                strokeWidth = strokeWidth,
                                            )
                                            x += spacing
                                        }

                                        // Draw diagonal lines from top-right to bottom-left
                                        x = 0f
                                        while (x < size.width + size.height) {
                                            drawLine(
                                                color = patternColor,
                                                start = Offset(x, 0f),
                                                end = Offset(x - size.height, size.height),
                                                strokeWidth = strokeWidth,
                                            )
                                            x += spacing
                                        }

                                        // Draw center oval border
                                        drawOval(
                                            color = Color(0xFFF5F5DC),
                                            topLeft =
                                                Offset(
                                                    centerX - ovalWidth / 2,
                                                    centerY - ovalHeight / 2,
                                                ),
                                            size = Size(ovalWidth, ovalHeight),
                                            style = Stroke(width = 3f),
                                        )

                                        // Fill center oval with solid color
                                        drawOval(
                                            color = Color(0xFF8B0000),
                                            topLeft =
                                                Offset(
                                                    centerX - ovalWidth / 2 + 2,
                                                    centerY - ovalHeight / 2 + 2,
                                                ),
                                            size = Size(ovalWidth - 4, ovalHeight - 4),
                                        )
                                    }
                                },
                    )
                }
            }
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
