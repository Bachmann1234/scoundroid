package dev.mattbachmann.scoundroid.data.model

/**
 * Represents the complete state of a Scoundrel game.
 *
 * @property deck The remaining cards in the dungeon deck
 * @property health Current player health (0-20)
 * @property currentRoom The cards currently in the room (4 cards when drawn, 1 after selection)
 * @property equippedWeapon The currently equipped weapon card, if any
 * @property discardPile Cards that have been discarded
 * @property canAvoidRoom Whether the player can avoid the current/next room
 * @property lastRoomAvoided Whether the last room was avoided (for tracking consecutive avoidance)
 */
data class GameState(
    val deck: Deck,
    val health: Int,
    val currentRoom: List<Card>?,
    val equippedWeapon: Card?,
    val discardPile: List<Card>,
    val canAvoidRoom: Boolean,
    val lastRoomAvoided: Boolean,
) {
    companion object {
        const val MAX_HEALTH = 20
        const val ROOM_SIZE = 4
        const val CARDS_TO_SELECT = 3

        /**
         * Creates a new game with a shuffled deck and starting conditions.
         */
        fun newGame(): GameState {
            return GameState(
                deck = Deck.create().shuffle(),
                health = MAX_HEALTH,
                currentRoom = null,
                equippedWeapon = null,
                discardPile = emptyList(),
                canAvoidRoom = true,
                lastRoomAvoided = false,
            )
        }
    }

    /**
     * Draws a new room of cards from the deck.
     * If there's a card remaining from the previous room, it's included.
     */
    fun drawRoom(): GameState {
        val cardsNeeded = if (currentRoom == null) ROOM_SIZE else ROOM_SIZE - 1
        val (drawnCards, remainingDeck) = deck.draw(cardsNeeded)

        val newRoom =
            if (currentRoom != null) {
                currentRoom + drawnCards
            } else {
                drawnCards
            }

        return copy(
            deck = remainingDeck,
            currentRoom = newRoom,
            canAvoidRoom = true,
            lastRoomAvoided = if (currentRoom != null) false else lastRoomAvoided,
        )
    }

    /**
     * Avoids the current room, moving all cards to the bottom of the deck.
     * Can only be done if canAvoidRoom is true.
     */
    fun avoidRoom(): GameState {
        require(currentRoom != null) { "No room to avoid" }
        require(!lastRoomAvoided) { "Cannot avoid room twice in a row" }

        // Move room cards to bottom of deck
        val newDeckCards = deck.cards + currentRoom
        val newDeck = Deck(newDeckCards)

        return copy(
            deck = newDeck,
            currentRoom = null,
            canAvoidRoom = false,
            lastRoomAvoided = true,
        )
    }

    /**
     * Selects cards from the current room to process.
     * The unselected card remains for the next room.
     *
     * @param selectedCards The cards to process (should be 3 out of 4)
     */
    fun selectCards(selectedCards: List<Card>): GameState {
        require(currentRoom != null) { "No room to select from" }
        require(selectedCards.size == CARDS_TO_SELECT) { "Must select exactly $CARDS_TO_SELECT cards" }

        val unselectedCards = currentRoom.filter { it !in selectedCards }
        require(unselectedCards.size == 1) { "Must leave exactly 1 card" }

        return copy(
            currentRoom = unselectedCards,
        )
    }

    /**
     * Reduces health by the specified amount.
     * Health cannot go below 0.
     */
    fun takeDamage(damage: Int): GameState {
        val newHealth = (health - damage).coerceAtLeast(0)
        return copy(health = newHealth)
    }

    /**
     * Increases health by the specified amount.
     * Health cannot exceed MAX_HEALTH (20).
     */
    fun heal(amount: Int): GameState {
        val newHealth = (health + amount).coerceAtMost(MAX_HEALTH)
        return copy(health = newHealth)
    }

    /**
     * Equips a weapon, replacing any currently equipped weapon.
     */
    fun equipWeapon(weapon: Card): GameState {
        require(weapon.type == CardType.WEAPON) { "Can only equip weapon cards" }
        return copy(equippedWeapon = weapon)
    }

    /**
     * Adds a card to the discard pile.
     */
    fun discard(card: Card): GameState {
        return copy(discardPile = discardPile + card)
    }

    /**
     * Returns true if the game is over (health reached 0).
     */
    val isGameOver: Boolean
        get() = health <= 0

    /**
     * Returns true if the player has won (deck is empty and health > 0).
     */
    val isGameWon: Boolean
        get() = deck.isEmpty && health > 0
}
