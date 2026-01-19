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
 * @property lastRoomAvoided Whether the last room was avoided (for tracking consecutive avoidance)
 * @property usedPotionThisTurn Whether a potion has been used this turn (only 1 per turn allowed)
 */
data class GameState(
    val deck: Deck,
    val health: Int,
    val currentRoom: List<Card>?,
    val weaponState: WeaponState?,
    val defeatedMonsters: List<Card>,
    val discardPile: List<Card>,
    val lastRoomAvoided: Boolean,
    val usedPotionThisTurn: Boolean,
) {
    companion object {
        const val MAX_HEALTH = 20
        const val ROOM_SIZE = 4
        const val CARDS_TO_SELECT = 3

        /**
         * Creates a new game with a shuffled deck and starting conditions.
         *
         * @param random Optional random source for deterministic shuffling (useful for tests)
         */
        fun newGame(random: kotlin.random.Random = kotlin.random.Random): GameState =
            GameState(
                deck = Deck.create().shuffle(random),
                health = MAX_HEALTH,
                currentRoom = null,
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
            )
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
            lastRoomAvoided = if (currentRoom != null) false else lastRoomAvoided,
            // New turn = can use potion again
            usedPotionThisTurn = false,
        )
    }

    /**
     * Avoids the current room, moving all cards to the bottom of the deck.
     * Cannot be done if the last room was also avoided.
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
        return copy(weaponState = WeaponState(weapon))
    }

    /**
     * Fights a monster using the equipped weapon.
     * Weapon must be able to defeat the monster.
     *
     * @param monster The monster to fight
     * @return New game state after combat with reduced damage and degraded weapon
     */
    fun fightMonsterWithWeapon(monster: Card): GameState {
        require(monster.type == CardType.MONSTER) { "Can only fight monster cards" }
        require(weaponState != null && weaponState.canDefeat(monster)) {
            "Weapon cannot defeat this monster"
        }

        val damage = (monster.value - weaponState.weapon.value).coerceAtLeast(0)
        val newWeaponState = weaponState.useOn(monster)

        return copy(
            health = (health - damage).coerceAtLeast(0),
            weaponState = newWeaponState,
            defeatedMonsters = defeatedMonsters + monster,
        )
    }

    /**
     * Fights a monster barehanded, taking full damage.
     * Weapon (if any) is not used and not degraded.
     *
     * @param monster The monster to fight
     * @return New game state after combat with full damage taken
     */
    fun fightMonsterBarehanded(monster: Card): GameState {
        require(monster.type == CardType.MONSTER) { "Can only fight monster cards" }

        return copy(
            health = (health - monster.value).coerceAtLeast(0),
            defeatedMonsters = defeatedMonsters + monster,
        )
    }

    /**
     * Fights a monster, applying damage and weapon degradation.
     * Auto-uses weapon if available and can defeat the monster.
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

        return if (weaponState != null && weaponState.canDefeat(monster)) {
            fightMonsterWithWeapon(monster)
        } else {
            fightMonsterBarehanded(monster)
        }
    }

    /**
     * Uses a potion to restore health.
     *
     * Potion rules:
     * - Only the FIRST potion per turn can be used
     * - Potions restore health by their value
     * - Health is capped at MAX_HEALTH (20)
     * - Second potion in same turn is discarded without effect
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
            )
        } else {
            // Second potion this turn: no effect
            copy(usedPotionThisTurn = true)
        }
    }

    /**
     * Calculates the current score.
     *
     * Scoring rules:
     * - **During play** (health > 0, deck not empty): score = health - remaining monster damage
     *   - Remaining monsters includes both deck AND current room (unprocessed cards)
     *   - This shows projected loss score during gameplay (can be negative)
     * - **Win** (health > 0, deck empty): score = health
     *   - The leftover card does NOT affect win score
     * - **Win with potion bonus**: If health = 20, deck empty, AND the leftover card is a potion,
     *   score = 20 + potion value
     * - **Lose** (health = 0): score = 0 - remaining monsters in deck only
     *
     * @return The current score
     */
    fun calculateScore(): Int {
        val deckMonsterDamage =
            deck.cards
                .filter { it.type == CardType.MONSTER }
                .sumOf { it.value }

        // Check if the leftover card (the one not selected from the final room) is a potion
        val leftoverCard = currentRoom?.singleOrNull()

        return if (health > 0) {
            if (deck.isEmpty) {
                // Game won: score = health, plus potion bonus if applicable
                val baseScore = health
                if (health == MAX_HEALTH &&
                    leftoverCard != null &&
                    leftoverCard.type == CardType.POTION
                ) {
                    baseScore + leftoverCard.value
                } else {
                    baseScore
                }
            } else {
                // During play: show projected score (health minus all remaining monsters)
                val roomMonsterDamage =
                    currentRoom
                        ?.filter { it.type == CardType.MONSTER }
                        ?.sumOf { it.value }
                        ?: 0
                health - deckMonsterDamage - roomMonsterDamage
            }
        } else {
            // Losing: score = negative sum of remaining monsters in deck only
            -deckMonsterDamage
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
