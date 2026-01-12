package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState

/**
 * Feature vector representation of game state for neural network input.
 */
data class GameStateFeatures(
    // Room cards (4 cards, each encoded as type + value)
    // Type: 0=Monster, 1=Weapon, 2=Potion
    val card0Type: Int,
    val card0Value: Int,
    val card1Type: Int,
    val card1Value: Int,
    val card2Type: Int,
    val card2Value: Int,
    val card3Type: Int,
    val card3Value: Int,
    // Player state
    val health: Int,
    val healthFraction: Float, // health / 20
    // Weapon state
    val hasWeapon: Boolean,
    val weaponValue: Int, // 0 if no weapon
    val weaponMaxMonster: Int, // 0 if no weapon, 15 if fresh
    val weaponIsFresh: Boolean,
    // Deck state
    val cardsRemaining: Int,
    val monstersRemaining: Int,
    val weaponsRemaining: Int,
    val potionsRemaining: Int,
    // Game state
    val lastRoomSkipped: Boolean,
    val canSkip: Boolean,
    // Derived features
    val totalMonsterValueInRoom: Int,
    val maxMonsterValueInRoom: Int,
    val totalPotionValueInRoom: Int,
    val maxWeaponValueInRoom: Int,
    val hasWeaponInRoom: Boolean,
    val hasPotionInRoom: Boolean,
) {
    /**
     * Convert to float array for neural network input.
     */
    fun toFloatArray(): FloatArray = floatArrayOf(
        // Card encodings (normalized)
        card0Type.toFloat() / 2f,
        card0Value.toFloat() / 14f,
        card1Type.toFloat() / 2f,
        card1Value.toFloat() / 14f,
        card2Type.toFloat() / 2f,
        card2Value.toFloat() / 14f,
        card3Type.toFloat() / 2f,
        card3Value.toFloat() / 14f,
        // Health
        health.toFloat() / 20f,
        healthFraction,
        // Weapon
        if (hasWeapon) 1f else 0f,
        weaponValue.toFloat() / 10f,
        weaponMaxMonster.toFloat() / 15f,
        if (weaponIsFresh) 1f else 0f,
        // Deck
        cardsRemaining.toFloat() / 44f,
        monstersRemaining.toFloat() / 26f,
        weaponsRemaining.toFloat() / 9f,
        potionsRemaining.toFloat() / 9f,
        // State
        if (lastRoomSkipped) 1f else 0f,
        if (canSkip) 1f else 0f,
        // Derived
        totalMonsterValueInRoom.toFloat() / 56f, // max possible: 14+13+12+11
        maxMonsterValueInRoom.toFloat() / 14f,
        totalPotionValueInRoom.toFloat() / 30f,
        maxWeaponValueInRoom.toFloat() / 10f,
        if (hasWeaponInRoom) 1f else 0f,
        if (hasPotionInRoom) 1f else 0f,
    )

    companion object {
        const val FEATURE_SIZE = 26

        fun fromGameState(state: GameState): GameStateFeatures? {
            val room = state.currentRoom ?: return null
            if (room.size != 4) return null

            val cards = room.sortedBy { it.hashCode() } // Consistent ordering

            fun cardType(card: Card): Int = when (card.type) {
                CardType.MONSTER -> 0
                CardType.WEAPON -> 1
                CardType.POTION -> 2
            }

            val monsters = room.filter { it.type == CardType.MONSTER }
            val weapons = room.filter { it.type == CardType.WEAPON }
            val potions = room.filter { it.type == CardType.POTION }

            val deckMonsters = state.deck.cards.count { it.type == CardType.MONSTER }
            val deckWeapons = state.deck.cards.count { it.type == CardType.WEAPON }
            val deckPotions = state.deck.cards.count { it.type == CardType.POTION }

            return GameStateFeatures(
                card0Type = cardType(cards[0]),
                card0Value = cards[0].value,
                card1Type = cardType(cards[1]),
                card1Value = cards[1].value,
                card2Type = cardType(cards[2]),
                card2Value = cards[2].value,
                card3Type = cardType(cards[3]),
                card3Value = cards[3].value,
                health = state.health,
                healthFraction = state.health / 20f,
                hasWeapon = state.weaponState != null,
                weaponValue = state.weaponState?.weapon?.value ?: 0,
                weaponMaxMonster = state.weaponState?.maxMonsterValue ?: if (state.weaponState != null) 15 else 0,
                weaponIsFresh = state.weaponState?.maxMonsterValue == null,
                cardsRemaining = state.deck.cards.size,
                monstersRemaining = deckMonsters,
                weaponsRemaining = deckWeapons,
                potionsRemaining = deckPotions,
                lastRoomSkipped = state.lastRoomAvoided,
                canSkip = !state.lastRoomAvoided,
                totalMonsterValueInRoom = monsters.sumOf { it.value },
                maxMonsterValueInRoom = monsters.maxOfOrNull { it.value } ?: 0,
                totalPotionValueInRoom = potions.sumOf { it.value },
                maxWeaponValueInRoom = weapons.maxOfOrNull { it.value } ?: 0,
                hasWeaponInRoom = weapons.isNotEmpty(),
                hasPotionInRoom = potions.isNotEmpty(),
            )
        }
    }
}

/**
 * A training example: game state features + the decision made + outcome.
 */
data class TrainingExample(
    val features: GameStateFeatures,
    val roomCards: List<String>, // Card names for debugging
    val decision: Decision,
    val gameWon: Boolean,
    val finalScore: Int,
)

sealed class Decision {
    data class LeaveCard(val cardIndex: Int) : Decision()
    object SkipRoom : Decision()
}
