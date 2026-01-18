package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for scoring mechanics.
 *
 * Scoring rules:
 * - **During play** (health > 0): score = health - remaining monster damage
 *   - This shows the projected loss score, giving real-time feedback during gameplay
 *   - Can be negative when remaining monsters outweigh health
 * - **Win** (deck empty, health > 0): score = health - 0 = health
 * - **Win with full health and last potion** (deck empty, health = 20, last card was potion):
 *   score = 20 + potion value (special case)
 * - **Lose** (health = 0): score = 0 - sum of remaining monsters (negative)
 *
 * Important: The potion bonus ONLY applies at game end (deck empty), with health = 20,
 * and the very last card processed was a health potion.
 */
class ScoringTest {
    @Test
    fun `winning score is remaining health`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 15,
            )

        val score = game.calculateScore()

        assertEquals(15, score, "Win score should equal remaining health")
    }

    @Test
    fun `winning with full health scores 20`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 20,
            )

        val score = game.calculateScore()

        assertEquals(20, score, "Win with full health scores 20")
    }

    @Test
    fun `winning with 1 health scores 1`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 1,
            )

        val score = game.calculateScore()

        assertEquals(1, score, "Win with 1 health scores 1")
    }

    @Test
    fun `losing score is negative sum of remaining monsters`() {
        val remainingCards =
            listOf(
                // 7
                Card(Suit.SPADES, Rank.SEVEN),
                // 5
                Card(Suit.CLUBS, Rank.FIVE),
                // 10
                Card(Suit.SPADES, Rank.TEN),
            )
        val deck = Deck(remainingCards)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        // Score = 0 - (7 + 5 + 10) = -22
        assertEquals(-22, score, "Lose score should be negative sum of remaining monsters")
    }

    @Test
    fun `losing with many monsters is very negative`() {
        val monsters =
            listOf(
                // 13
                Card(Suit.SPADES, Rank.KING),
                // 12
                Card(Suit.CLUBS, Rank.QUEEN),
                // 11
                Card(Suit.SPADES, Rank.JACK),
                // 10
                Card(Suit.CLUBS, Rank.TEN),
            )
        val deck = Deck(monsters)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        // Score = 0 - (13 + 12 + 11 + 10) = -46
        assertEquals(-46, score, "Many remaining monsters = very negative score")
    }

    @Test
    fun `losing score ignores non-monster cards`() {
        val remainingCards =
            listOf(
                // 7 (monster)
                Card(Suit.SPADES, Rank.SEVEN),
                // 5 (weapon - ignore)
                Card(Suit.DIAMONDS, Rank.FIVE),
                // 3 (potion - ignore)
                Card(Suit.HEARTS, Rank.THREE),
                // 10 (monster)
                Card(Suit.CLUBS, Rank.TEN),
            )
        val deck = Deck(remainingCards)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        // Score = 0 - (7 + 10) = -17 (weapons and potions don't count)
        assertEquals(-17, score, "Only monsters count toward losing score")
    }

    @Test
    fun `losing with no remaining cards scores 0`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 0,
            )

        val score = game.calculateScore()

        assertEquals(0, score, "Losing with no cards left scores 0")
    }

    @Test
    fun `mid-game score is health minus remaining monsters`() {
        // Mid-game score
        val partialDeck =
            Deck(
                listOf(
                    Card(Suit.SPADES, Rank.FIVE), // 5 damage
                    Card(Suit.CLUBS, Rank.SEVEN), // 7 damage
                ),
            )
        val game =
            GameState.newGame().copy(
                deck = partialDeck,
                health = 10,
            )

        // Can calculate score even if game isn't over
        val score = game.calculateScore()

        // Score = health - remaining monsters = 10 - (5 + 7) = -2
        assertEquals(-2, score, "Mid-game score = health - remaining monster damage")
    }

    @Test
    fun `mid-game score can be negative`() {
        // Many monsters remaining
        val monsterHeavyDeck =
            Deck(
                listOf(
                    Card(Suit.SPADES, Rank.KING), // 13
                    Card(Suit.CLUBS, Rank.QUEEN), // 12
                    Card(Suit.SPADES, Rank.JACK), // 11
                ),
            )
        val game =
            GameState.newGame().copy(
                deck = monsterHeavyDeck,
                health = 15,
            )

        val score = game.calculateScore()

        // Score = 15 - (13 + 12 + 11) = 15 - 36 = -21
        assertEquals(-21, score, "Mid-game score can be negative when monsters outweigh health")
    }

    @Test
    fun `mid-game score ignores non-monster cards`() {
        val mixedDeck =
            Deck(
                listOf(
                    Card(Suit.SPADES, Rank.TEN), // 10 (monster)
                    Card(Suit.DIAMONDS, Rank.FIVE), // 5 (weapon - ignored)
                    Card(Suit.HEARTS, Rank.THREE), // 3 (potion - ignored)
                ),
            )
        val game =
            GameState.newGame().copy(
                deck = mixedDeck,
                health = 15,
            )

        val score = game.calculateScore()

        // Score = 15 - 10 = 5 (only monster counts)
        assertEquals(5, score, "Mid-game score only considers monsters")
    }

    @Test
    fun `losing score with one Ace monster`() {
        val aceMonster = listOf(Card(Suit.SPADES, Rank.ACE)) // 14
        val deck = Deck(aceMonster)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        assertEquals(-14, score, "Single Ace monster = -14")
    }

    @Test
    fun `winning with various health values`() {
        val emptyDeck = Deck(emptyList())

        for (health in 1..20) {
            val game =
                GameState.newGame().copy(
                    deck = emptyDeck,
                    health = health,
                )

            assertEquals(health, game.calculateScore(), "Health $health should score $health")
        }
    }

    @Test
    fun `losing score includes all remaining monsters`() {
        // Create a deck with only monsters
        val monsters =
            listOf(
                // 2
                Card(Suit.SPADES, Rank.TWO),
                // 3
                Card(Suit.CLUBS, Rank.THREE),
                // 4
                Card(Suit.SPADES, Rank.FOUR),
                // 5
                Card(Suit.CLUBS, Rank.FIVE),
            )
        val deck = Deck(monsters)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        // 2 + 3 + 4 + 5 = 14, so score = -14
        assertEquals(-14, score)
    }

    @Test
    fun `score is negative when losing`() {
        val monsters = listOf(Card(Suit.SPADES, Rank.TWO))
        val deck = Deck(monsters)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        assertTrue(score <= 0, "Losing score should be ≤ 0")
    }

    @Test
    fun `score is positive when winning`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 5,
            )

        val score = game.calculateScore()

        assertTrue(score > 0, "Winning score should be > 0")
    }

    @Test
    fun `winning with full health and last potion adds potion value to score`() {
        val emptyDeck = Deck(emptyList())
        var game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 15,
            )

        // Use a potion that brings health to exactly 20
        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥
        game = game.usePotion(potion)

        // Health should be 20
        assertEquals(20, game.health)

        // Score should be 20 + 5 = 25 (special case)
        val score = game.calculateScore()
        assertEquals(25, score, "Win with health=20 after potion should score health + potion value")
    }

    @Test
    fun `winning with full health but no recent potion scores normal`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 20,
            )

        // No potion used, just at full health
        val score = game.calculateScore()
        assertEquals(20, score, "Win with health=20 without potion should score 20")
    }

    @Test
    fun `winning with full health after multiple potions uses last card if potion`() {
        val emptyDeck = Deck(emptyList())
        var game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 10,
            )

        // Use first potion
        val potion1 = Card(Suit.HEARTS, Rank.FIVE) // 5♥
        game = game.usePotion(potion1)
        assertEquals(15, game.health)

        // Start new turn and use second potion to reach 20 (this is the LAST card)
        game = game.drawRoom()
        val potion2 = Card(Suit.HEARTS, Rank.SEVEN) // 7♥
        game = game.usePotion(potion2)
        assertEquals(20, game.health)

        // Should use last card's potion value (7) because last card was a potion
        val score = game.calculateScore()
        assertEquals(27, score, "Should use last card potion value: 20 + 7 = 27")
    }

    @Test
    fun `winning with health less than 20 after potion scores normally`() {
        val emptyDeck = Deck(emptyList())
        var game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 10,
            )

        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥
        game = game.usePotion(potion)

        // Health is 15, not 20
        assertEquals(15, game.health)

        // Score should be just 15 (no special case)
        val score = game.calculateScore()
        assertEquals(15, score, "Win with health < 20 scores normally even after potion")
    }

    @Test
    fun `EXPLOIT FIX - potion then monster does not apply potion bonus`() {
        val emptyDeck = Deck(emptyList())
        var game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 20,
            )

        // Use a potion at full health (no health gain, but sets flag)
        val potion = Card(Suit.HEARTS, Rank.TEN) // 10♥
        game = game.usePotion(potion)
        assertEquals(20, game.health, "Health stays at 20")

        // Then fight a monster (last card is now a monster, not potion)
        val monster = Card(Suit.SPADES, Rank.TWO) // 2♠
        game = game.fightMonster(monster)
        assertEquals(18, game.health, "Health drops to 18 after barehanded combat")

        // Then heal back to 20 with another potion
        game = game.drawRoom() // New turn
        val potion2 = Card(Suit.HEARTS, Rank.TWO) // 2♥
        game = game.usePotion(potion2)
        assertEquals(20, game.health, "Back to full health")

        // Now fight another monster (last card = monster)
        game = game.drawRoom()
        val weapon = Card(Suit.DIAMONDS, Rank.TEN) // 10♦ weapon
        game = game.equipWeapon(weapon)
        val monster2 = Card(Suit.CLUBS, Rank.TWO) // 2♣
        game = game.fightMonster(monster2)

        // Score should be just 20, NOT 20 + 10 or 20 + 2
        // because last card was a monster, not a potion
        val score = game.calculateScore()
        assertEquals(20, score, "Score should be 20, not include potion bonus because last card was monster")
    }

    @Test
    fun `EXPLOIT FIX - potion then weapon does not apply potion bonus`() {
        val emptyDeck = Deck(emptyList())
        var game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 10,
            )

        // Use a potion to reach 20
        val potion = Card(Suit.HEARTS, Rank.TEN) // 10♥
        game = game.usePotion(potion)
        assertEquals(20, game.health)

        // Then equip a weapon (last card is now a weapon)
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        game = game.equipWeapon(weapon)

        // Score should be just 20, not 30, because last card was weapon
        val score = game.calculateScore()
        assertEquals(20, score, "Score should be 20 because last card was weapon, not potion")
    }

    @Test
    fun `BUG FIX - second potion in same turn should not affect scoring`() {
        val emptyDeck = Deck(emptyList())
        var game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 15,
            )

        // Use first potion to reach 20
        val potion1 = Card(Suit.HEARTS, Rank.FIVE) // 5♥
        game = game.usePotion(potion1)
        assertEquals(20, game.health)

        // Use second potion in SAME turn (should be discarded, no effect)
        val potion2 = Card(Suit.HEARTS, Rank.SEVEN) // 7♥
        game = game.usePotion(potion2)
        assertEquals(20, game.health)

        // Score should be 20 + 5 = 25 (first potion), NOT 20 + 7 = 27 (second potion)
        val score = game.calculateScore()
        assertEquals(
            25,
            score,
            "Score should use first potion value (5), not discarded second potion (7)",
        )
    }
}
