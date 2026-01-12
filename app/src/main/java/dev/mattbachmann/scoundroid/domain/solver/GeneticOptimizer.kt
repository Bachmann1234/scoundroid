package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlin.random.Random

/**
 * Different fitness strategies for optimization.
 */
enum class FitnessStrategy {
    /** Heavily weight wins (original) */
    WIN_FOCUSED,

    /** Only optimize for average score */
    SCORE_ONLY,

    /** Balanced: modest bonus for wins */
    BALANCED,
}

/**
 * Genetic algorithm optimizer for finding optimal PlayerGenome parameters.
 */
class GeneticOptimizer(
    private val populationSize: Int = 50,
    private val gamesPerEvaluation: Int = 5000,
    private val mutationRate: Double = 0.15,
    private val crossoverRate: Double = 0.7,
    private val eliteCount: Int = 5,
    private val rng: Random = Random,
    private val fitnessStrategy: FitnessStrategy = FitnessStrategy.WIN_FOCUSED,
) {
    /**
     * Evolve the population for the given number of generations.
     * Returns the best genome found.
     */
    fun evolve(
        generations: Int,
        startingSeed: Long = 1L,
        onGeneration: (Int, ScoredGenome, List<ScoredGenome>) -> Unit = { _, _, _ -> },
    ): ScoredGenome {
        var population = (0 until populationSize).map { PlayerGenome.random(rng) }
        var bestEver: ScoredGenome? = null

        for (gen in 1..generations) {
            // Evaluate all genomes in parallel
            val seedStart = startingSeed + (gen - 1) * gamesPerEvaluation
            val scored = evaluatePopulation(population, seedStart)

            // Track best ever
            val bestThisGen = scored.maxByOrNull { it.fitness }!!
            if (bestEver == null || bestThisGen.fitness > bestEver.fitness) {
                bestEver = bestThisGen
            }

            onGeneration(gen, bestThisGen, scored)

            if (gen < generations) {
                population = evolvePopulation(scored)
            }
        }

        return bestEver!!
    }

    private fun evaluatePopulation(
        population: List<PlayerGenome>,
        startingSeed: Long,
    ): List<ScoredGenome> =
        runBlocking(Dispatchers.Default) {
            population
                .mapIndexed { index, genome ->
                    async {
                        // Each genome gets a different seed range to avoid overlap
                        val seedOffset = index * gamesPerEvaluation
                        val result = evaluateGenome(genome, startingSeed + seedOffset)
                        ScoredGenome(genome, result)
                    }
                }.map { it.await() }
        }

    private fun evaluateGenome(
        genome: PlayerGenome,
        startingSeed: Long,
    ): EvaluationResult {
        val player = ParameterizedPlayer(genome)
        var wins = 0
        var totalScore = 0L

        for (seed in startingSeed until startingSeed + gamesPerEvaluation) {
            val game = GameState.newGame(Random(seed))
            val finalState = player.playGame(game)

            val won =
                finalState.health > 0 &&
                    finalState.deck.isEmpty &&
                    (finalState.currentRoom == null || finalState.currentRoom.isEmpty())

            if (won) wins++
            totalScore += finalState.calculateScore()
        }

        val winRate = wins.toDouble() / gamesPerEvaluation
        val avgScore = totalScore.toDouble() / gamesPerEvaluation

        // Fitness depends on strategy
        val fitness =
            when (fitnessStrategy) {
                FitnessStrategy.WIN_FOCUSED -> winRate * 10000 + avgScore
                FitnessStrategy.SCORE_ONLY -> avgScore
                FitnessStrategy.BALANCED -> avgScore + (winRate * 500)
            }

        return EvaluationResult(
            wins = wins,
            games = gamesPerEvaluation,
            winRate = winRate,
            averageScore = avgScore,
            fitness = fitness,
        )
    }

    private fun evolvePopulation(scored: List<ScoredGenome>): List<PlayerGenome> {
        val sorted = scored.sortedByDescending { it.fitness }

        val newPopulation = mutableListOf<PlayerGenome>()

        // Elitism: keep the best unchanged
        newPopulation.addAll(sorted.take(eliteCount).map { it.genome })

        // Fill the rest with offspring
        while (newPopulation.size < populationSize) {
            val parent1 = tournamentSelect(scored)
            val parent2 = tournamentSelect(scored)

            var child =
                if (rng.nextDouble() < crossoverRate) {
                    parent1.crossover(parent2, rng)
                } else {
                    if (rng.nextBoolean()) parent1 else parent2
                }

            child = child.mutate(mutationRate, rng)
            newPopulation.add(child)
        }

        return newPopulation
    }

    private fun tournamentSelect(
        scored: List<ScoredGenome>,
        tournamentSize: Int = 3,
    ): PlayerGenome {
        val tournament = (0 until tournamentSize).map { scored.random(rng) }
        return tournament.maxByOrNull { it.fitness }!!.genome
    }
}

data class EvaluationResult(
    val wins: Int,
    val games: Int,
    val winRate: Double,
    val averageScore: Double,
    val fitness: Double,
)

data class ScoredGenome(
    val genome: PlayerGenome,
    val result: EvaluationResult,
) {
    val fitness: Double get() = result.fitness
}
