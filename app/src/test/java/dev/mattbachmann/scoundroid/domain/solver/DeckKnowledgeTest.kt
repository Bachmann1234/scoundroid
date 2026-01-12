package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.Deck
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DeckKnowledgeTest {
    // Helper to create cards easily
    private fun monster(
        value: Int,
        suit: Suit = Suit.CLUBS,
    ): Card = Card(suit, Rank.fromValue(value))

    private fun weapon(value: Int): Card = Card(Suit.DIAMONDS, Rank.fromValue(value))

    private fun potion(value: Int): Card = Card(Suit.HEARTS, Rank.fromValue(value))

    @Test
    fun `initial state knows full deck composition`() {
        val knowledge = DeckKnowledge.initial()

        // Full Scoundrel deck: 26 monsters, 9 weapons, 9 potions
        assertEquals(26, knowledge.remainingMonsters.size)
        assertEquals(9, knowledge.remainingWeapons.size)
        assertEquals(9, knowledge.remainingPotions.size)
        assertEquals(44, knowledge.remainingCards.size)
    }

    @Test
    fun `initial state has correct max monster value`() {
        val knowledge = DeckKnowledge.initial()

        // Aces are the biggest monsters (value 14)
        assertEquals(14, knowledge.maxMonsterRemaining)
    }

    @Test
    fun `initial state has correct total damage remaining`() {
        val knowledge = DeckKnowledge.initial()

        // Monsters: 2-10, J(11), Q(12), K(13), A(14) for both Clubs and Spades
        // Sum per suit: 2+3+4+5+6+7+8+9+10+11+12+13+14 = 104
        // Total: 104 * 2 = 208
        assertEquals(208, knowledge.totalDamageRemaining)
    }

    @Test
    fun `initial state has correct max weapon value`() {
        val knowledge = DeckKnowledge.initial()

        // 10♦ is the biggest weapon
        assertEquals(10, knowledge.maxWeaponRemaining)
    }

    @Test
    fun `initial state has correct total healing remaining`() {
        val knowledge = DeckKnowledge.initial()

        // Potions: 2-10 of Hearts = 2+3+4+5+6+7+8+9+10 = 54
        assertEquals(54, knowledge.totalHealingRemaining)
    }

    @Test
    fun `processing a monster removes it from remaining`() {
        val knowledge = DeckKnowledge.initial()
        val aceOfClubs = monster(14, Suit.CLUBS)

        val updated = knowledge.cardProcessed(aceOfClubs)

        assertEquals(25, updated.remainingMonsters.size)
        assertTrue(aceOfClubs !in updated.remainingCards)
        assertTrue(aceOfClubs in updated.processedCards)
    }

    @Test
    fun `processing both aces lowers max monster to 13`() {
        var knowledge = DeckKnowledge.initial()
        knowledge = knowledge.cardProcessed(monster(14, Suit.CLUBS))
        knowledge = knowledge.cardProcessed(monster(14, Suit.SPADES))

        assertEquals(13, knowledge.maxMonsterRemaining)
    }

    @Test
    fun `processing a weapon removes it from remaining`() {
        val knowledge = DeckKnowledge.initial()
        val tenOfDiamonds = weapon(10)

        val updated = knowledge.cardProcessed(tenOfDiamonds)

        assertEquals(8, updated.remainingWeapons.size)
        assertEquals(9, updated.maxWeaponRemaining) // Next best is 9
    }

    @Test
    fun `processing a potion reduces healing remaining`() {
        val knowledge = DeckKnowledge.initial()
        val tenOfHearts = potion(10)

        val updated = knowledge.cardProcessed(tenOfHearts)

        assertEquals(8, updated.remainingPotions.size)
        assertEquals(44, updated.totalHealingRemaining) // 54 - 10 = 44
    }

    @Test
    fun `skipping cards tracks them at bottom of deck`() {
        val knowledge = DeckKnowledge.initial()
        val skippedRoom =
            listOf(
                monster(14, Suit.CLUBS),
                weapon(10),
                potion(5),
                monster(2, Suit.SPADES),
            )

        val updated = knowledge.roomSkipped(skippedRoom)

        // Cards are still in remaining (they'll come back)
        assertEquals(44, updated.remainingCards.size)

        // But they're tracked as skipped
        assertEquals(4, updated.skippedCards.size)
        assertEquals(skippedRoom, updated.skippedCards)
    }

    @Test
    fun `skipped cards maintain order for position tracking`() {
        val knowledge = DeckKnowledge.initial()
        val room1 =
            listOf(
                monster(14, Suit.CLUBS),
                monster(13, Suit.CLUBS),
                monster(12, Suit.CLUBS),
                monster(11, Suit.CLUBS),
            )
        val room2 =
            listOf(
                monster(10, Suit.CLUBS),
                monster(9, Suit.CLUBS),
                monster(8, Suit.CLUBS),
                monster(7, Suit.CLUBS),
            )

        var updated = knowledge.roomSkipped(room1)
        updated = updated.roomSkipped(room2)

        // Both rooms at bottom, first skipped room first
        assertEquals(8, updated.skippedCards.size)
        assertEquals(room1 + room2, updated.skippedCards)
    }

    @Test
    fun `processing cards from skipped room removes from skipped list`() {
        val knowledge = DeckKnowledge.initial()
        val ace = monster(14, Suit.CLUBS)
        val skippedRoom = listOf(ace, weapon(10), potion(5), monster(2, Suit.SPADES))

        var updated = knowledge.roomSkipped(skippedRoom)
        updated = updated.cardProcessed(ace)

        // Ace is now processed, not in skipped
        assertEquals(3, updated.skippedCards.size)
        assertTrue(ace !in updated.skippedCards)
        assertTrue(ace in updated.processedCards)

        // And it's removed from remaining
        assertEquals(43, updated.remainingCards.size)
    }

    @Test
    fun `cardsRemainingInDeck excludes skipped cards`() {
        val knowledge = DeckKnowledge.initial()
        val skippedRoom =
            listOf(
                monster(14, Suit.CLUBS),
                weapon(10),
                potion(5),
                monster(2, Suit.SPADES),
            )

        val updated = knowledge.roomSkipped(skippedRoom)

        // 44 total - 4 at bottom = 40 cards could be drawn before skipped cards return
        assertEquals(40, updated.cardsBeforeSkipped)
    }

    @Test
    fun `empty deck has zero for all counts`() {
        var knowledge = DeckKnowledge.initial()

        // Process all cards
        for (card in Deck.create().cards) {
            knowledge = knowledge.cardProcessed(card)
        }

        assertEquals(0, knowledge.remainingMonsters.size)
        assertEquals(0, knowledge.remainingWeapons.size)
        assertEquals(0, knowledge.remainingPotions.size)
        assertEquals(0, knowledge.maxMonsterRemaining)
        assertEquals(0, knowledge.maxWeaponRemaining)
        assertEquals(0, knowledge.totalDamageRemaining)
        assertEquals(0, knowledge.totalHealingRemaining)
    }

    @Test
    fun `expected damage per room calculated correctly`() {
        val knowledge = DeckKnowledge.initial()

        // 26 monsters, 208 total damage, 44 cards
        // In a room of 4, expected monsters = 4 * (26/44) ≈ 2.36
        // Expected damage = 2.36 * (208/26) = 2.36 * 8 ≈ 18.9
        val expected = knowledge.expectedDamagePerRoom()

        // Just verify it's reasonable (roughly 18-19)
        assertTrue(expected > 15.0)
        assertTrue(expected < 22.0)
    }

    @Test
    fun `chance of weapon in next room calculated correctly`() {
        val knowledge = DeckKnowledge.initial()

        // 9 weapons out of 44 cards
        // P(at least 1 weapon in 4 cards) = 1 - P(no weapons in 4)
        // = 1 - (35/44 * 34/43 * 33/42 * 32/41) ≈ 0.59
        val chance = knowledge.chanceOfWeaponInNextRoom()

        assertTrue(chance > 0.55)
        assertTrue(chance < 0.65)
    }

    @Test
    fun `chance of weapon is 1 when 4+ weapons remain with small deck`() {
        var knowledge = DeckKnowledge.initial()

        // Process all but 4 weapons
        val allCards = Deck.create().cards
        val monsters = allCards.filter { it.type == CardType.MONSTER }
        val potions = allCards.filter { it.type == CardType.POTION }
        val weapons = allCards.filter { it.type == CardType.WEAPON }

        for (card in monsters) {
            knowledge = knowledge.cardProcessed(card)
        }
        for (card in potions) {
            knowledge = knowledge.cardProcessed(card)
        }
        // Process 5 weapons, leaving 4
        for (card in weapons.take(5)) {
            knowledge = knowledge.cardProcessed(card)
        }

        // Only 4 cards left, all weapons
        assertEquals(4, knowledge.remainingCards.size)
        assertEquals(1.0, knowledge.chanceOfWeaponInNextRoom(), 0.001)
    }

    @Test
    fun `survival margin calculates correctly`() {
        val knowledge = DeckKnowledge.initial()

        // Survival margin = health + totalHealingRemaining - totalDamageRemaining
        // = 20 + 54 - 208 = -134
        val margin = knowledge.survivalMargin(20)

        assertEquals(-134, margin)
    }

    @Test
    fun `survival margin improves as monsters are processed`() {
        var knowledge = DeckKnowledge.initial()
        val initialMargin = knowledge.survivalMargin(20)

        // Process the Ace of Clubs (14 damage removed)
        knowledge = knowledge.cardProcessed(monster(14, Suit.CLUBS))
        val newMargin = knowledge.survivalMargin(20)

        assertEquals(initialMargin + 14, newMargin)
    }
}
