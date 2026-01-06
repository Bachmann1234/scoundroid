package dev.mattbachmann.scoundroid.data.model

/**
 * Represents the rank of a card.
 * Values: 2-10 = face value, J=11, Q=12, K=13, A=14
 */
enum class Rank(val value: Int, val displayName: String) {
    TWO(2, "2"),
    THREE(3, "3"),
    FOUR(4, "4"),
    FIVE(5, "5"),
    SIX(6, "6"),
    SEVEN(7, "7"),
    EIGHT(8, "8"),
    NINE(9, "9"),
    TEN(10, "10"),
    JACK(11, "J"),
    QUEEN(12, "Q"),
    KING(13, "K"),
    ACE(14, "A")
}
