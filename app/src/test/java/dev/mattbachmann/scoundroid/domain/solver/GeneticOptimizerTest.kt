package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import org.junit.Ignore
import org.junit.Test
import java.io.File
import kotlin.random.Random

class GeneticOptimizerTest {
    @Test
    @Ignore("Long running experiment - run manually")
    fun `compare all three fitness strategies`() {
        val output = StringBuilder()

        fun log(s: String) {
            println(s)
            output.appendLine(s)
        }

        log("=== THREE-WAY FITNESS STRATEGY COMPARISON ===")
        log("Population: 30, Games per evaluation: 3000, Generations: 30")
        log("")

        data class StrategyResult(
            val name: String,
            val genome: PlayerGenome,
            val trainingResult: EvaluationResult,
            val validationResult: EvaluationResult,
        )

        val results = mutableListOf<StrategyResult>()

        for (strategy in FitnessStrategy.entries) {
            log("--- RUNNING ${strategy.name} OPTIMIZATION ---")
            val startTime = System.currentTimeMillis()

            val optimizer =
                GeneticOptimizer(
                    populationSize = 30,
                    gamesPerEvaluation = 3000,
                    mutationRate = 0.15,
                    fitnessStrategy = strategy,
                )

            val best =
                optimizer.evolve(
                    generations = 30,
                    startingSeed = 1L,
                ) { gen, bestThisGen, _ ->
                    if (gen % 10 == 0 || gen == 1) {
                        log(
                            "  Gen %2d: %.3f%% wins, avg score=%.1f".format(
                                gen,
                                bestThisGen.result.winRate * 100,
                                bestThisGen.result.averageScore,
                            ),
                        )
                    }
                }

            val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
            log("  Completed in %.1fs".format(elapsed))

            // Validate
            val validation = validateGenome(best.genome, 500_001L..550_000L)

            results.add(
                StrategyResult(
                    name = strategy.name,
                    genome = best.genome,
                    trainingResult = best.result,
                    validationResult = validation,
                ),
            )
            log("")
        }

        // Compare genomes
        log("=== EVOLVED PARAMETERS COMPARISON ===")
        log("")
        log(
            "%-35s | %12s | %12s | %12s".format(
                "Parameter",
                "WIN_FOCUSED",
                "SCORE_ONLY",
                "BALANCED",
            ),
        )
        log("-".repeat(80))

        val genomes = results.map { it.genome }
        log(
            "%-35s | %12d | %12d | %12d".format(
                "skipIfDamageExceedsHealthMinus",
                genomes[0].skipIfDamageExceedsHealthMinus,
                genomes[1].skipIfDamageExceedsHealthMinus,
                genomes[2].skipIfDamageExceedsHealthMinus,
            ),
        )
        log(
            "%-35s | %12.3f | %12.3f | %12.3f".format(
                "skipWithoutWeaponDamageFraction",
                genomes[0].skipWithoutWeaponDamageFraction,
                genomes[1].skipWithoutWeaponDamageFraction,
                genomes[2].skipWithoutWeaponDamageFraction,
            ),
        )
        log(
            "%-35s | %12.3f | %12.3f | %12.3f".format(
                "monsterLeavePenaltyMultiplier",
                genomes[0].monsterLeavePenaltyMultiplier,
                genomes[1].monsterLeavePenaltyMultiplier,
                genomes[2].monsterLeavePenaltyMultiplier,
            ),
        )
        log(
            "%-35s | %12.3f | %12.3f | %12.3f".format(
                "weaponLeavePenaltyIfNeeded",
                genomes[0].weaponLeavePenaltyIfNeeded,
                genomes[1].weaponLeavePenaltyIfNeeded,
                genomes[2].weaponLeavePenaltyIfNeeded,
            ),
        )
        log(
            "%-35s | %12d | %12d | %12d".format(
                "weaponPreservationThreshold",
                genomes[0].weaponPreservationThreshold,
                genomes[1].weaponPreservationThreshold,
                genomes[2].weaponPreservationThreshold,
            ),
        )
        log(
            "%-35s | %12d | %12d | %12d".format(
                "minDamageSavedToUseWeapon",
                genomes[0].minDamageSavedToUseWeapon,
                genomes[1].minDamageSavedToUseWeapon,
                genomes[2].minDamageSavedToUseWeapon,
            ),
        )
        log(
            "%-35s | %12d | %12d | %12d".format(
                "emergencyHealthBuffer",
                genomes[0].emergencyHealthBuffer,
                genomes[1].emergencyHealthBuffer,
                genomes[2].emergencyHealthBuffer,
            ),
        )
        log(
            "%-35s | %12d | %12d | %12d".format(
                "equipFreshWeaponIfDegradedBelow",
                genomes[0].equipFreshWeaponIfDegradedBelow,
                genomes[1].equipFreshWeaponIfDegradedBelow,
                genomes[2].equipFreshWeaponIfDegradedBelow,
            ),
        )
        log("")

        // Validation results
        log("=== VALIDATION ON 50,000 FRESH SEEDS ===")
        log("")
        log(
            "%-15s | %12s | %12s | %12s".format(
                "Metric",
                "WIN_FOCUSED",
                "SCORE_ONLY",
                "BALANCED",
            ),
        )
        log("-".repeat(60))
        log(
            "%-15s | %11.3f%% | %11.3f%% | %11.3f%%".format(
                "Win Rate",
                results[0].validationResult.winRate * 100,
                results[1].validationResult.winRate * 100,
                results[2].validationResult.winRate * 100,
            ),
        )
        log(
            "%-15s | %12.1f | %12.1f | %12.1f".format(
                "Avg Score",
                results[0].validationResult.averageScore,
                results[1].validationResult.averageScore,
                results[2].validationResult.averageScore,
            ),
        )
        log(
            "%-15s | %12d | %12d | %12d".format(
                "Total Wins",
                results[0].validationResult.wins,
                results[1].validationResult.wins,
                results[2].validationResult.wins,
            ),
        )
        log("")

        // Find best of each
        val bestWinRate = results.maxBy { it.validationResult.winRate }
        val bestScore = results.maxBy { it.validationResult.averageScore }

        log("=== SUMMARY ===")
        log("Best win rate: ${bestWinRate.name} (%.3f%%)".format(bestWinRate.validationResult.winRate * 100))
        log("Best avg score: ${bestScore.name} (%.1f)".format(bestScore.validationResult.averageScore))

        if (bestWinRate.name == bestScore.name) {
            log("CONCLUSION: ${bestWinRate.name} is strictly best!")
        } else {
            log("CONCLUSION: Trade-off remains between ${bestWinRate.name} (wins) and ${bestScore.name} (score)")
        }

        // Save results
        File("build/three-way-comparison.txt").writeText(output.toString())
        log("")
        log("Results saved to build/three-way-comparison.txt")
    }

    @Test
    @Ignore("Long running experiment - run manually")
    fun `compare win-focused vs score-focused optimization`() {
        val output = StringBuilder()

        fun log(s: String) {
            println(s)
            output.appendLine(s)
        }

        log("=== WIN-FOCUSED vs SCORE-FOCUSED COMPARISON ===")
        log("Population: 30, Games per evaluation: 3000, Generations: 30")
        log("")

        val startTime = System.currentTimeMillis()

        // Run win-focused optimization
        log("--- RUNNING WIN-FOCUSED OPTIMIZATION ---")
        val winOptimizer =
            GeneticOptimizer(
                populationSize = 30,
                gamesPerEvaluation = 3000,
                mutationRate = 0.15,
                fitnessStrategy = FitnessStrategy.WIN_FOCUSED,
            )

        val winBest =
            winOptimizer.evolve(
                generations = 30,
                startingSeed = 1L,
            ) { gen, bestThisGen, _ ->
                if (gen % 5 == 0 || gen == 1) {
                    log(
                        "  Gen %2d: %.3f%% wins, avg score=%.1f".format(
                            gen,
                            bestThisGen.result.winRate * 100,
                            bestThisGen.result.averageScore,
                        ),
                    )
                }
            }

        val winTime = (System.currentTimeMillis() - startTime) / 1000.0
        log("Win-focused completed in %.1fs".format(winTime))
        log("")

        // Run score-focused optimization
        log("--- RUNNING SCORE-FOCUSED OPTIMIZATION ---")
        val scoreStartTime = System.currentTimeMillis()
        val scoreOptimizer =
            GeneticOptimizer(
                populationSize = 30,
                gamesPerEvaluation = 3000,
                mutationRate = 0.15,
                fitnessStrategy = FitnessStrategy.SCORE_ONLY,
            )

        val scoreBest =
            scoreOptimizer.evolve(
                generations = 30,
                startingSeed = 1L,
            ) { gen, bestThisGen, _ ->
                if (gen % 5 == 0 || gen == 1) {
                    log(
                        "  Gen %2d: %.3f%% wins, avg score=%.1f".format(
                            gen,
                            bestThisGen.result.winRate * 100,
                            bestThisGen.result.averageScore,
                        ),
                    )
                }
            }

        val scoreTime = (System.currentTimeMillis() - scoreStartTime) / 1000.0
        log("Score-focused completed in %.1fs".format(scoreTime))
        log("")

        // Compare genomes
        log("=== EVOLVED PARAMETERS COMPARISON ===")
        log("")
        log("%-35s | %12s | %12s".format("Parameter", "Win-Focused", "Score-Focus"))
        log("-".repeat(65))
        log(
            "%-35s | %12d | %12d".format(
                "skipIfDamageExceedsHealthMinus",
                winBest.genome.skipIfDamageExceedsHealthMinus,
                scoreBest.genome.skipIfDamageExceedsHealthMinus,
            ),
        )
        log(
            "%-35s | %12.3f | %12.3f".format(
                "skipWithoutWeaponDamageFraction",
                winBest.genome.skipWithoutWeaponDamageFraction,
                scoreBest.genome.skipWithoutWeaponDamageFraction,
            ),
        )
        log(
            "%-35s | %12.3f | %12.3f".format(
                "monsterLeavePenaltyMultiplier",
                winBest.genome.monsterLeavePenaltyMultiplier,
                scoreBest.genome.monsterLeavePenaltyMultiplier,
            ),
        )
        log(
            "%-35s | %12.3f | %12.3f".format(
                "weaponLeavePenaltyIfNeeded",
                winBest.genome.weaponLeavePenaltyIfNeeded,
                scoreBest.genome.weaponLeavePenaltyIfNeeded,
            ),
        )
        log(
            "%-35s | %12d | %12d".format(
                "weaponPreservationThreshold",
                winBest.genome.weaponPreservationThreshold,
                scoreBest.genome.weaponPreservationThreshold,
            ),
        )
        log(
            "%-35s | %12d | %12d".format(
                "minDamageSavedToUseWeapon",
                winBest.genome.minDamageSavedToUseWeapon,
                scoreBest.genome.minDamageSavedToUseWeapon,
            ),
        )
        log(
            "%-35s | %12d | %12d".format(
                "emergencyHealthBuffer",
                winBest.genome.emergencyHealthBuffer,
                scoreBest.genome.emergencyHealthBuffer,
            ),
        )
        log(
            "%-35s | %12d | %12d".format(
                "equipFreshWeaponIfDegradedBelow",
                winBest.genome.equipFreshWeaponIfDegradedBelow,
                scoreBest.genome.equipFreshWeaponIfDegradedBelow,
            ),
        )
        log("")

        // Validate both on same seeds
        log("=== VALIDATION ON 50,000 FRESH SEEDS ===")
        val validationRange = 500_001L..550_000L

        val winValidation = validateGenome(winBest.genome, validationRange)
        val scoreValidation = validateGenome(scoreBest.genome, validationRange)

        log("")
        log("%-20s | %12s | %12s".format("Metric", "Win-Focused", "Score-Focus"))
        log("-".repeat(50))
        log(
            "%-20s | %12.3f%% | %12.3f%%".format(
                "Win Rate",
                winValidation.winRate * 100,
                scoreValidation.winRate * 100,
            ),
        )
        log(
            "%-20s | %12.1f | %12.1f".format(
                "Average Score",
                winValidation.averageScore,
                scoreValidation.averageScore,
            ),
        )
        log(
            "%-20s | %12d | %12d".format(
                "Total Wins",
                winValidation.wins,
                scoreValidation.wins,
            ),
        )
        log("")

        // Analysis
        log("=== ANALYSIS ===")
        val scoreDiff = scoreValidation.averageScore - winValidation.averageScore
        val winDiff = (scoreValidation.winRate - winValidation.winRate) * 100
        log("Score difference: %.1f (positive = score-focused is better)".format(scoreDiff))
        log("Win rate difference: %.3f%% (positive = score-focused wins more)".format(winDiff))

        if (scoreDiff > 0 && winDiff >= 0) {
            log("CONCLUSION: Score-focused optimization is strictly better!")
        } else if (scoreDiff > 0 && winDiff < 0) {
            log("CONCLUSION: Trade-off - score-focused has better avg score but fewer wins")
        } else if (scoreDiff <= 0 && winDiff > 0) {
            log("CONCLUSION: Trade-off - win-focused has better avg score but score-focused wins more")
        } else {
            log("CONCLUSION: Win-focused optimization is strictly better or equivalent")
        }

        // Save results
        File("build/fitness-comparison.txt").writeText(output.toString())
        log("")
        log("Results saved to build/fitness-comparison.txt")
    }

    @Test
    @Ignore("Long running optimization - run manually")
    fun `evolve optimal player`() {
        val output = StringBuilder()

        fun log(s: String) {
            println(s)
            output.appendLine(s)
        }

        log("Starting genetic optimization...")
        log("Population: 50, Games per evaluation: 5000, Generations: 50")
        log("")

        val optimizer =
            GeneticOptimizer(
                populationSize = 50,
                gamesPerEvaluation = 5000,
                mutationRate = 0.15,
            )

        val startTime = System.currentTimeMillis()

        val best =
            optimizer.evolve(
                generations = 50,
                startingSeed = 1L,
            ) { gen, bestThisGen, population ->
                val avgFitness = population.map { it.fitness }.average()
                val elapsed = (System.currentTimeMillis() - startTime) / 1000.0

                log(
                    "Gen %2d: Best=%.1f (%.3f%% wins, avg=%.1f) | Pop avg=%.1f | Time: %.1fs".format(
                        gen,
                        bestThisGen.fitness,
                        bestThisGen.result.winRate * 100,
                        bestThisGen.result.averageScore,
                        avgFitness,
                        elapsed,
                    ),
                )
            }

        val totalTime = (System.currentTimeMillis() - startTime) / 1000.0

        log("")
        log("=== OPTIMIZATION COMPLETE ===")
        log("Total time: %.1f seconds".format(totalTime))
        log("")
        log("Best genome found:")
        log("  skipIfDamageExceedsHealthMinus = ${best.genome.skipIfDamageExceedsHealthMinus}")
        log("  skipWithoutWeaponDamageFraction = ${"%.3f".format(best.genome.skipWithoutWeaponDamageFraction)}")
        log("  monsterLeavePenaltyMultiplier = ${"%.3f".format(best.genome.monsterLeavePenaltyMultiplier)}")
        log("  weaponLeavePenaltyIfNeeded = ${"%.3f".format(best.genome.weaponLeavePenaltyIfNeeded)}")
        log("  weaponPreservationThreshold = ${best.genome.weaponPreservationThreshold}")
        log("  minDamageSavedToUseWeapon = ${best.genome.minDamageSavedToUseWeapon}")
        log("  emergencyHealthBuffer = ${best.genome.emergencyHealthBuffer}")
        log("  equipFreshWeaponIfDegradedBelow = ${best.genome.equipFreshWeaponIfDegradedBelow}")
        log("")
        log(
            "Training results: ${best.result.wins}/${best.result.games} wins (${"%.3f".format(
                best.result.winRate * 100,
            )}%)",
        )
        log("")

        // Validate on fresh seeds
        log("Validating on 100,000 fresh seeds...")
        val validationResult = validateGenome(best.genome, 100_001L..200_000L)
        log(
            "Validation: ${validationResult.wins}/${validationResult.games} wins (${"%.3f".format(
                validationResult.winRate * 100,
            )}%)",
        )
        log("Validation avg score: ${"%.1f".format(validationResult.averageScore)}")

        // Compare to default
        log("")
        log("Comparing to default genome...")
        val defaultResult = validateGenome(PlayerGenome.DEFAULT, 100_001L..200_000L)
        log(
            "Default: ${defaultResult.wins}/${defaultResult.games} wins (${"%.3f".format(
                defaultResult.winRate * 100,
            )}%)",
        )

        val improvement =
            if (defaultResult.winRate > 0) {
                validationResult.winRate / defaultResult.winRate
            } else {
                Double.POSITIVE_INFINITY
            }
        log("Improvement: ${"%.1f".format(improvement)}x")

        // Save results
        File("build/ga-results.txt").writeText(output.toString())
        log("")
        log("Results saved to build/ga-results.txt")
    }

    @Test
    fun `quick optimization test`() {
        // Quick test with smaller parameters to verify it works
        val optimizer =
            GeneticOptimizer(
                populationSize = 10,
                gamesPerEvaluation = 500,
                mutationRate = 0.2,
            )

        val best =
            optimizer.evolve(
                generations = 5,
                startingSeed = 1L,
            ) { gen, bestThisGen, _ ->
                println("Gen $gen: fitness=${bestThisGen.fitness}, wins=${bestThisGen.result.wins}")
            }

        println("Best genome: ${best.genome}")
        println("Fitness: ${best.fitness}")

        // Just verify it ran without crashing
        assert(best.fitness > Double.NEGATIVE_INFINITY)
    }

    private fun validateGenome(
        genome: PlayerGenome,
        seedRange: LongRange,
    ): EvaluationResult {
        val player = ParameterizedPlayer(genome)
        var wins = 0
        var totalScore = 0L
        val games = seedRange.count()

        for (seed in seedRange) {
            val game = GameState.newGame(Random(seed))
            val finalState = player.playGame(game)

            val won =
                finalState.health > 0 &&
                    finalState.deck.isEmpty &&
                    (finalState.currentRoom == null || finalState.currentRoom.isEmpty())

            if (won) wins++
            totalScore += finalState.calculateScore()
        }

        val winRate = wins.toDouble() / games
        val avgScore = totalScore.toDouble() / games

        return EvaluationResult(
            wins = wins,
            games = games,
            winRate = winRate,
            averageScore = avgScore,
            fitness = winRate * 10000 + avgScore,
        )
    }
}
