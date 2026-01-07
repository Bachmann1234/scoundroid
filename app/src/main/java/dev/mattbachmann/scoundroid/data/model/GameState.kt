package dev.mattbachmann.scoundroid.data.model

/**
 * Represents the complete state of a Scoundrel game.
 *
 * @property deck The remaining cards in the dungeon deck
 * @property health Current player health (0-20)
 * @property currentRoom The cards currently in the room (4 cards when drawn, 1 after selection)
 * @property weaponState The currently equipped weapon with degradation tracking, if any
 * @property defeatedMonsters List of monsters defeated (weapon stack)
 * @property discardPile Cards that have been discarded
 * @property canAvoidRoom Whether the player can avoid the current/next room
 * @property lastRoomAvoided Whether the last room was avoided (for tracking consecutive avoidance)
 * @property usedPotionThisTurn Whether a potion has been used this turn (only 1 per turn allowed)
 * @property lastCardProcessed The last card processed (monster/weapon/potion) for special scoring
 */
data class GameState(
    val deck: Deck,
    val health: Int,
    val currentRoom: List<Card>?,
    val weaponState: WeaponState?,
    val defeatedMonsters: List<Card>,
    val discardPile: List<Card>,
    val canAvoidRoom: Boolean,
    val lastRoomAvoided: Boolean,
    val usedPotionThisTurn: Boolean,
    val lastCardProcessed: Card?,
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
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                canAvoidRoom = true,
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
            )
        }
    }

    /**
     * Draws a new room of cards from the deck.
     * If there's a card remaining from the previous room, it's included.
     * Resets the potion flag (new turn = can use potion again).
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
            // New turn = can use potion again
            usedPotionThisTurn = false,
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
     * The weapon starts with no degradation (can defeat any monster).
     */
    fun equipWeapon(weapon: Card): GameState {
        require(weapon.type == CardType.WEAPON) { "Can only equip weapon cards" }
        return copy(
            weaponState = WeaponState(weapon),
            lastCardProcessed = weapon,
        )
    }

    /**
     * Fights a monster, applying damage and weapon degradation.
     *
     * Combat rules:
     * - If no weapon or weapon can't defeat monster: barehanded (full damage)
     * - If weapon can defeat monster: damage = max(0, monster - weapon), weapon degrades
     * - Monster goes to defeated pile
     *
     * @param monster The monster to fight
     * @return New game state after combat
     */
    fun fightMonster(monster: Card): GameState {
        require(monster.type == CardType.MONSTER) { "Can only fight monster cards" }

        val damage: Int
        val newWeaponState: WeaponState?

        if (weaponState != null && weaponState.canDefeat(monster)) {
            // Weapon combat: reduced damage and weapon degrades
            damage = (monster.value - weaponState.weapon.value).coerceAtLeast(0)
            newWeaponState = weaponState.useOn(monster)
        } else {
            // Barehanded combat: full damage, weapon unchanged
            damage = monster.value
            newWeaponState = weaponState
        }

        return copy(
            health = (health - damage).coerceAtLeast(0),
            weaponState = newWeaponState,
            defeatedMonsters = defeatedMonsters + monster,
            lastCardProcessed = monster,
        )
    }

    /**
     * Uses a potion to restore health.
     *
     * Potion rules:
     * - Only the FIRST potion per turn can be used
     * - Potions restore health by their value
     * - Health is capped at MAX_HEALTH (20)
     * - Second potion in same turn is discarded without effect
     * - Tracks last card processed for special scoring
     *
     * @param potion The potion card to use
     * @return New game state after using (or discarding) the potion
     */
    fun usePotion(potion: Card): GameState {
        require(potion.type == CardType.POTION) { "Can only use potion cards" }

        return if (!usedPotionThisTurn) {
            // First potion this turn: restore health
            copy(
                health = (health + potion.value).coerceAtMost(MAX_HEALTH),
                usedPotionThisTurn = true,
                lastCardProcessed = potion,
            )
        } else {
            // Second potion this turn: no effect (but still track it was processed)
            copy(lastCardProcessed = potion)
        }
    }

    /**
     * Adds a card to the discard pile.
     */
    fun discard(card: Card): GameState {
        return copy(discardPile = discardPile + card)
    }

    /**
     * Calculates the current score.
     *
     * Scoring rules:
     * - **Winning** (health > 0): score = remaining health
     * - **Special case**: If health = 20 AND last card processed was a potion,
     *   score = 20 + potion value
     * - **Losing** (health = 0): score = 0 - sum of remaining monsters in deck
     *
     * @return The current score
     */
    fun calculateScore(): Int {
        return if (health > 0) {
            // Winning: score = remaining health
            if (health == MAX_HEALTH &&
                lastCardProcessed != null &&
                lastCardProcessed.type == CardType.POTION
            ) {
                // Special case: full health AND last card was a potion
                health + lastCardProcessed.value
            } else {
                health
            }
        } else {
            // Losing: score = negative sum of remaining monsters
            val remainingMonsterDamage =
                deck.cards
                    .filter { it.type == CardType.MONSTER }
                    .sumOf { it.value }
            -remainingMonsterDamage
        }
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
