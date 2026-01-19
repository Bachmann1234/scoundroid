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
 * - **Win with full health and leftover potion** (deck empty, health = 20, leftover card is potion):
 *   score = 20 + potion value (special case)
 * - **Lose** (health = 0): score = 0 - sum of remaining monsters (negative)
 *
 * Important: The potion bonus ONLY applies at game end (deck empty), with health = 20,
 * and the leftover card (the one not selected from the final room) is a potion.
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
    fun `losing score includes monsters in current room`() {
        // When you die mid-game, monsters in the room count against you
        val deckMonsters = listOf(Card(Suit.SPADES, Rank.FIVE)) // 5 in deck
        val roomMonsters = listOf(
            Card(Suit.CLUBS, Rank.SEVEN), // 7 in room
            Card(Suit.SPADES, Rank.TEN), // 10 in room
        )
        val game =
            GameState.newGame().copy(
                deck = Deck(deckMonsters),
                currentRoom = roomMonsters,
                health = 0,
            )

        val score = game.calculateScore()

        // Score = 0 - (5 + 7 + 10) = -22 (all remaining monsters count)
        assertEquals(-22, score, "Losing score should include monsters from both deck and room")
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
    fun `potion bonus applies when leftover card is a potion`() {
        // The leftover card (the one not selected from final room) is a potion
        val emptyDeck = Deck(emptyList())
        val leftoverPotion = Card(Suit.HEARTS, Rank.SEVEN) // 7♥ potion left over
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                currentRoom = listOf(leftoverPotion),
                health = 20,
            )

        val score = game.calculateScore()
        // Score = 20 + 7 = 27 (health + leftover potion value)
        assertEquals(27, score, "Potion bonus should add leftover potion value to score")
    }

    @Test
    fun `win score with leftover monster is just remaining health`() {
        // Per docs/rules.md: "If you win: Score = remaining health"
        // The leftover card (unselected from final room) does NOT affect win score
        // unless it's a potion with full health (bonus case)
        val emptyDeck = Deck(emptyList())
        val leftoverMonster = Card(Suit.SPADES, Rank.FIVE) // 5♠ monster left over
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                currentRoom = listOf(leftoverMonster),
                health = 20,
            )

        val score = game.calculateScore()
        // Win score = remaining health = 20 (leftover monster does NOT reduce score)
        assertEquals(20, score, "Win score should be remaining health, not reduced by leftover monster")
    }

    @Test
    fun `potion bonus does not apply when leftover card is a weapon`() {
        val emptyDeck = Deck(emptyList())
        val leftoverWeapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦ weapon left over
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                currentRoom = listOf(leftoverWeapon),
                health = 20,
            )

        val score = game.calculateScore()
        // Score = 20 - 0 = 20 (no monster damage, no potion bonus)
        assertEquals(20, score, "No potion bonus when leftover card is a weapon")
    }

    @Test
    fun `potion bonus does not apply when health is less than 20`() {
        val emptyDeck = Deck(emptyList())
        val leftoverPotion = Card(Suit.HEARTS, Rank.SEVEN) // 7♥ potion left over
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                currentRoom = listOf(leftoverPotion),
                health = 15,
            )

        val score = game.calculateScore()
        // Score = 15 (no bonus because health < 20)
        assertEquals(15, score, "No potion bonus when health is less than 20")
    }

    @Test
    fun `potion bonus does not apply when no leftover card`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                currentRoom = null,
                health = 20,
            )

        val score = game.calculateScore()
        // Score = 20 (no leftover card to provide bonus)
        assertEquals(20, score, "No potion bonus when no leftover card")
    }

    @Test
    fun `potion bonus does not apply when deck is not empty`() {
        val remainingDeck = Deck(listOf(Card(Suit.SPADES, Rank.TWO))) // 2♠ still in deck
        val leftoverPotion = Card(Suit.HEARTS, Rank.SEVEN) // 7♥ potion left over
        val game =
            GameState.newGame().copy(
                deck = remainingDeck,
                currentRoom = listOf(leftoverPotion),
                health = 20,
            )

        val score = game.calculateScore()
        // Score = 20 - 2 = 18 (health minus remaining monster, no potion bonus yet)
        assertEquals(18, score, "No potion bonus when deck is not empty")
    }

    @Test
    fun `potion bonus uses leftover potion value`() {
        // Test different potion values
        val emptyDeck = Deck(emptyList())

        for (rank in listOf(Rank.TWO, Rank.FIVE, Rank.TEN)) {
            val leftoverPotion = Card(Suit.HEARTS, rank)
            val game =
                GameState.newGame().copy(
                    deck = emptyDeck,
                    currentRoom = listOf(leftoverPotion),
                    health = 20,
                )

            val expectedScore = 20 + rank.value
            assertEquals(
                expectedScore,
                game.calculateScore(),
                "Potion bonus should be 20 + ${rank.value} = $expectedScore",
            )
        }
    }
}
