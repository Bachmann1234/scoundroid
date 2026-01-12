package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.WeaponState

/**
 * Captures a complete game's decision trace for analysis.
 */
data class GameLog(
    val seed: Long,
    val won: Boolean,
    val finalHealth: Int,
    val finalScore: Int,
    val cardsRemainingInDeck: Int,
    val events: List<GameEvent>,
    val deathInfo: DeathInfo?,
)

/**
 * Information about how the player died.
 */
data class DeathInfo(
    val killerMonster: Card,
    val healthBeforeHit: Int,
    val damageTaken: Int,
    val hadWeapon: Boolean,
    val weaponValue: Int?,
    val weaponMaxMonster: Int?,
    val couldWeaponHaveHelped: Boolean,
    val roomsSkipped: Int,
    val potionsUsedThisGame: Int,
)

/**
 * Events that occur during a game.
 */
sealed class GameEvent {
    data class RoomDrawn(
        val roomNumber: Int,
        val cards: List<Card>,
        val health: Int,
        val deckRemaining: Int,
        val weaponState: WeaponState?,
    ) : GameEvent()

    data class RoomSkipped(
        val roomNumber: Int,
        val cards: List<Card>,
        val estimatedDamage: Int,
        val health: Int,
        val reason: String,
    ) : GameEvent()

    data class CardLeftBehind(
        val roomNumber: Int,
        val cardLeft: Card,
        val cardsProcessed: List<Card>,
        val reason: String,
    ) : GameEvent()

    data class CombatResolved(
        val monster: Card,
        val usedWeapon: Boolean,
        val weaponValue: Int?,
        val damageTaken: Int,
        val healthBefore: Int,
        val healthAfter: Int,
    ) : GameEvent()

    data class WeaponEquipped(
        val weapon: Card,
        val previousWeapon: Card?,
        val previousWeaponDegraded: Boolean,
    ) : GameEvent()

    data class WeaponSkipped(
        val weapon: Card,
        val currentWeapon: Card,
        val reason: String,
    ) : GameEvent()

    data class PotionUsed(
        val potion: Card,
        val healthBefore: Int,
        val healthAfter: Int,
        val healingWasted: Int,
    ) : GameEvent()

    data class PotionWasted(
        val potion: Card,
        val reason: String,
    ) : GameEvent()

    data class GameEnded(
        val won: Boolean,
        val finalHealth: Int,
        val score: Int,
    ) : GameEvent()
}

/**
 * Aggregate statistics across many games.
 */
data class AggregateStats(
    val totalGames: Int,
    val wins: Int,
    val losses: Int,
    val winRate: Double,
    // Death statistics
    val deathsByMonsterRank: Map<Int, Int>,
    val averageHealthAtDeath: Double,
    val averageCardsRemainingAtDeath: Double,
    val deathsWithUnusedWeapon: Int,
    val deathsWhereWeaponCouldHaveHelped: Int,
    val averageRoomsSkippedBeforeDeath: Double,
    val averagePotionsUsedBeforeDeath: Double,
    // Score distribution
    val averageWinScore: Double?,
    val averageLossScore: Double,
    val scoreDistribution: Map<IntRange, Int>,
) {
    fun prettyPrint(): String =
        buildString {
            appendLine("=== AGGREGATE STATISTICS ===")
            appendLine(
                "Games: $totalGames | Wins: $wins | Losses: $losses | Win Rate: ${"%.3f".format(winRate * 100)}%",
            )
            appendLine()
            appendLine("--- Death Analysis ---")
            appendLine("Average health at death: ${"%.1f".format(averageHealthAtDeath)}")
            appendLine("Average cards remaining in deck: ${"%.1f".format(averageCardsRemainingAtDeath)}")
            appendLine("Average rooms skipped before death: ${"%.1f".format(averageRoomsSkippedBeforeDeath)}")
            appendLine("Average potions used before death: ${"%.1f".format(averagePotionsUsedBeforeDeath)}")
            appendLine()
            appendLine(
                "Deaths with unused weapon: $deathsWithUnusedWeapon (${"%.1f".format(
                    deathsWithUnusedWeapon * 100.0 / losses,
                )}%)",
            )
            appendLine(
                "Deaths where weapon COULD have helped: $deathsWhereWeaponCouldHaveHelped (${"%.1f".format(
                    deathsWhereWeaponCouldHaveHelped * 100.0 / losses,
                )}%)",
            )
            appendLine()
            appendLine("--- Deaths by Monster Rank ---")
            deathsByMonsterRank.entries.sortedByDescending { it.value }.forEach { (rank, count) ->
                val pct = count * 100.0 / losses
                val bar = "█".repeat((pct / 2).toInt())
                appendLine("  Rank %2d: %4d (%5.1f%%) %s".format(rank, count, pct, bar))
            }
            appendLine()
            appendLine("--- Scores ---")
            averageWinScore?.let { appendLine("Average winning score: ${"%.1f".format(it)}") }
            appendLine("Average losing score: ${"%.1f".format(averageLossScore)}")
        }
}

/**
 * Builder for collecting aggregate statistics.
 */
class AggregateStatsBuilder {
    private var totalGames = 0
    private var wins = 0
    private var losses = 0

    private val deathsByMonsterRank = mutableMapOf<Int, Int>()
    private var totalHealthAtDeath = 0
    private var totalCardsRemainingAtDeath = 0
    private var deathsWithUnusedWeapon = 0
    private var deathsWhereWeaponCouldHaveHelped = 0
    private var totalRoomsSkipped = 0
    private var totalPotionsUsed = 0

    private var totalWinScore = 0
    private var winCount = 0
    private var totalLossScore = 0
    private var lossCount = 0
    private val scores = mutableListOf<Int>()

    fun addGame(log: GameLog) {
        totalGames++
        scores.add(log.finalScore)

        if (log.won) {
            wins++
            totalWinScore += log.finalScore
            winCount++
        } else {
            losses++
            totalLossScore += log.finalScore
            lossCount++

            log.deathInfo?.let { death ->
                deathsByMonsterRank[death.killerMonster.rank.value] =
                    deathsByMonsterRank.getOrDefault(death.killerMonster.rank.value, 0) + 1
                totalHealthAtDeath += death.healthBeforeHit
                totalCardsRemainingAtDeath += log.cardsRemainingInDeck
                totalRoomsSkipped += death.roomsSkipped
                totalPotionsUsed += death.potionsUsedThisGame

                if (death.hadWeapon) {
                    deathsWithUnusedWeapon++
                }
                if (death.couldWeaponHaveHelped) {
                    deathsWhereWeaponCouldHaveHelped++
                }
            }
        }
    }

    fun build(): AggregateStats {
        val scoreRanges =
            listOf(
                Int.MIN_VALUE..-50,
                -50..-20,
                -20..-10,
                -10..0,
                0..5,
                5..10,
                10..15,
                15..20,
            )

        val scoreDistribution =
            scoreRanges.associateWith { range ->
                scores.count { it in range }
            }

        return AggregateStats(
            totalGames = totalGames,
            wins = wins,
            losses = losses,
            winRate = if (totalGames > 0) wins.toDouble() / totalGames else 0.0,
            deathsByMonsterRank = deathsByMonsterRank.toMap(),
            averageHealthAtDeath = if (losses > 0) totalHealthAtDeath.toDouble() / losses else 0.0,
            averageCardsRemainingAtDeath = if (losses > 0) totalCardsRemainingAtDeath.toDouble() / losses else 0.0,
            deathsWithUnusedWeapon = deathsWithUnusedWeapon,
            deathsWhereWeaponCouldHaveHelped = deathsWhereWeaponCouldHaveHelped,
            averageRoomsSkippedBeforeDeath = if (losses > 0) totalRoomsSkipped.toDouble() / losses else 0.0,
            averagePotionsUsedBeforeDeath = if (losses > 0) totalPotionsUsed.toDouble() / losses else 0.0,
            averageWinScore = if (winCount > 0) totalWinScore.toDouble() / winCount else null,
            averageLossScore = if (lossCount > 0) totalLossScore.toDouble() / lossCount else Double.NaN,
            scoreDistribution = scoreDistribution,
        )
    }
}
