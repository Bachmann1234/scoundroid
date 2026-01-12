package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import org.junit.Ignore
import org.junit.Test
import java.io.File
import kotlin.random.Random

class TrainingDataGenerator {
    @Test
    fun `generate training data from heuristic player`() {
        val numGames = 50_000
        val outputFile = File("build/training_data.csv")

        println("Generating training data from $numGames games...")

        var wins = 0
        var totalExamples = 0L

        // Write incrementally to avoid OOM
        outputFile.printWriter().use { out ->
            // Header
            val featureNames = listOf(
                "card0_type", "card0_value", "card1_type", "card1_value",
                "card2_type", "card2_value", "card3_type", "card3_value",
                "health", "health_fraction",
                "has_weapon", "weapon_value", "weapon_max_monster", "weapon_is_fresh",
                "cards_remaining", "monsters_remaining", "weapons_remaining", "potions_remaining",
                "last_room_skipped", "can_skip",
                "total_monster_value", "max_monster_value", "total_potion_value", "max_weapon_value",
                "has_weapon_in_room", "has_potion_in_room",
                "decision_type", "decision_card_index", "game_won", "final_score"
            )
            out.println(featureNames.joinToString(","))

            for (seed in 0L until numGames) {
                val player = DataCollectionPlayer()
                val (finalState, examples) = player.playGameAndCollect(GameState.newGame(Random(seed)))

                if (finalState.isGameWon) wins++

                // Write examples immediately
                for (example in examples) {
                    val features = example.features.toFloatArray()
                    val decisionType = when (example.decision) {
                        is Decision.LeaveCard -> 0
                        Decision.SkipRoom -> 1
                    }
                    val cardIndex = when (val d = example.decision) {
                        is Decision.LeaveCard -> d.cardIndex
                        Decision.SkipRoom -> -1
                    }

                    val row = features.toList() + listOf(
                        decisionType.toFloat(),
                        cardIndex.toFloat(),
                        if (example.gameWon) 1f else 0f,
                        example.finalScore.toFloat()
                    )
                    out.println(row.joinToString(","))
                    totalExamples++
                }

                if (seed % 10000 == 0L && seed > 0) {
                    println("  Processed $seed games, $totalExamples examples, $wins wins")
                }
            }
        }

        println("Total: $totalExamples examples from $numGames games ($wins wins)")
        println("Saved to ${outputFile.absolutePath}")
    }

    @Test
    fun `generate training data from winning games only`() {
        val targetWins = 5_000
        val outputFile = File("build/training_data_wins_only.csv")

        println("Generating training data from $targetWins winning games...")

        val allExamples = mutableListOf<TrainingExample>()
        var seed = 0L
        var wins = 0

        while (wins < targetWins) {
            val player = DataCollectionPlayer()
            val (finalState, examples) = player.playGameAndCollect(GameState.newGame(Random(seed)))

            if (finalState.isGameWon) {
                allExamples.addAll(examples)
                wins++

                if (wins % 1000 == 0) {
                    println("  Found $wins wins after $seed games, collected ${allExamples.size} examples")
                }
            }

            seed++
        }

        println("Total: ${allExamples.size} examples from $wins winning games (searched $seed games)")

        // Write to CSV
        outputFile.printWriter().use { out ->
            val featureNames = listOf(
                "card0_type", "card0_value", "card1_type", "card1_value",
                "card2_type", "card2_value", "card3_type", "card3_value",
                "health", "health_fraction",
                "has_weapon", "weapon_value", "weapon_max_monster", "weapon_is_fresh",
                "cards_remaining", "monsters_remaining", "weapons_remaining", "potions_remaining",
                "last_room_skipped", "can_skip",
                "total_monster_value", "max_monster_value", "total_potion_value", "max_weapon_value",
                "has_weapon_in_room", "has_potion_in_room",
                "decision_type", "decision_card_index", "game_won", "final_score"
            )
            out.println(featureNames.joinToString(","))

            for (example in allExamples) {
                val features = example.features.toFloatArray()
                val decisionType = when (example.decision) {
                    is Decision.LeaveCard -> 0
                    Decision.SkipRoom -> 1
                }
                val cardIndex = when (val d = example.decision) {
                    is Decision.LeaveCard -> d.cardIndex
                    Decision.SkipRoom -> -1
                }

                val row = features.toList() + listOf(
                    decisionType.toFloat(),
                    cardIndex.toFloat(),
                    if (example.gameWon) 1f else 0f,
                    example.finalScore.toFloat()
                )
                out.println(row.joinToString(","))
            }
        }

        println("Saved to ${outputFile.absolutePath}")
    }

    @Test
    fun `quick test data generation`() {
        // Quick test to verify data generation works
        val player = DataCollectionPlayer()
        val (finalState, examples) = player.playGameAndCollect(GameState.newGame(Random(42)))

        println("Game result: ${if (finalState.isGameWon) "WIN" else "LOSS"}, score: ${finalState.calculateScore()}")
        println("Collected ${examples.size} decision examples")

        if (examples.isNotEmpty()) {
            println("\nFirst example:")
            val first = examples.first()
            println("  Room: ${first.roomCards}")
            println("  Decision: ${first.decision}")
            println("  Features: ${first.features.toFloatArray().take(10)}...")
        }

        assert(examples.isNotEmpty()) { "Should collect at least some examples" }
    }
}
