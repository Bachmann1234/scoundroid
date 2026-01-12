package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import kotlin.random.Random

/**
 * Benchmark comparing InformedPlayer vs HeuristicPlayer.
 * Run with: ./gradlew :app:testDebugUnitTest --tests "*.PlayerBenchmark.runBenchmark"
 */
object PlayerBenchmark {
    @JvmStatic
    fun main(args: Array<String>) {
        runBenchmark(10_000L)
    }

    fun runBenchmark(numSeeds: Long): BenchmarkResult {
        val informedPlayer = InformedPlayer()
        val heuristicPlayer = HeuristicPlayer()

        var informedWins = 0
        var heuristicWins = 0
        var bothWin = 0
        var onlyInformed = 0
        var onlyHeuristic = 0

        val startTime = System.currentTimeMillis()

        for (seed in 1L..numSeeds) {
            val game = GameState.newGame(Random(seed))

            val informedResult = informedPlayer.playGame(game)
            val heuristicResult = heuristicPlayer.playGame(game)

            val informedWon =
                informedResult.health > 0 &&
                    informedResult.deck.isEmpty &&
                    (informedResult.currentRoom == null || informedResult.currentRoom.isEmpty())
            val heuristicWon =
                heuristicResult.health > 0 &&
                    heuristicResult.deck.isEmpty &&
                    (heuristicResult.currentRoom == null || heuristicResult.currentRoom.isEmpty())

            if (informedWon) informedWins++
            if (heuristicWon) heuristicWins++
            if (informedWon && heuristicWon) bothWin++
            if (informedWon && !heuristicWon) onlyInformed++
            if (!informedWon && heuristicWon) onlyHeuristic++
        }

        val elapsed = System.currentTimeMillis() - startTime

        return BenchmarkResult(
            numSeeds = numSeeds,
            elapsedMs = elapsed,
            informedWins = informedWins,
            heuristicWins = heuristicWins,
            bothWin = bothWin,
            onlyInformed = onlyInformed,
            onlyHeuristic = onlyHeuristic,
        )
    }

    data class BenchmarkResult(
        val numSeeds: Long,
        val elapsedMs: Long,
        val informedWins: Int,
        val heuristicWins: Int,
        val bothWin: Int,
        val onlyInformed: Int,
        val onlyHeuristic: Int,
        val onlyInformedSeeds: List<Long> = emptyList(),
        val onlyHeuristicSeeds: List<Long> = emptyList(),
    ) {
        fun print() {
            println("\n╔══════════════════════════════════════════════╗")
            println("║   INFORMED vs HEURISTIC: $numSeeds SEEDS")
            println("╠══════════════════════════════════════════════╣")
            println("║  Time: ${elapsedMs}ms")
            println("║  Informed wins:  $informedWins (${String.format("%.3f", informedWins * 100.0 / numSeeds)}%)")
            println("║  Heuristic wins: $heuristicWins (${String.format("%.3f", heuristicWins * 100.0 / numSeeds)}%)")
            println("╠══════════════════════════════════════════════╣")
            println("║  Both win:       $bothWin")
            println("║  Only Informed:  $onlyInformed")
            println("║  Only Heuristic: $onlyHeuristic")
            println("╚══════════════════════════════════════════════╝\n")
        }
    }

    fun runBenchmarkWithSeeds(numSeeds: Long): BenchmarkResult {
        val informedPlayer = InformedPlayer()
        val heuristicPlayer = HeuristicPlayer()

        var informedWins = 0
        var heuristicWins = 0
        var bothWin = 0
        var onlyInformed = 0
        var onlyHeuristic = 0
        val onlyInformedSeeds = mutableListOf<Long>()
        val onlyHeuristicSeeds = mutableListOf<Long>()

        val startTime = System.currentTimeMillis()

        for (seed in 1L..numSeeds) {
            val game = GameState.newGame(Random(seed))

            val informedResult = informedPlayer.playGame(game)
            val heuristicResult = heuristicPlayer.playGame(game)

            val informedWon =
                informedResult.health > 0 &&
                    informedResult.deck.isEmpty &&
                    (informedResult.currentRoom == null || informedResult.currentRoom.isEmpty())
            val heuristicWon =
                heuristicResult.health > 0 &&
                    heuristicResult.deck.isEmpty &&
                    (heuristicResult.currentRoom == null || heuristicResult.currentRoom.isEmpty())

            if (informedWon) informedWins++
            if (heuristicWon) heuristicWins++
            if (informedWon && heuristicWon) bothWin++
            if (informedWon && !heuristicWon) {
                onlyInformed++
                onlyInformedSeeds.add(seed)
            }
            if (!informedWon && heuristicWon) {
                onlyHeuristic++
                onlyHeuristicSeeds.add(seed)
            }
        }

        val elapsed = System.currentTimeMillis() - startTime

        return BenchmarkResult(
            numSeeds = numSeeds,
            elapsedMs = elapsed,
            informedWins = informedWins,
            heuristicWins = heuristicWins,
            bothWin = bothWin,
            onlyInformed = onlyInformed,
            onlyHeuristic = onlyHeuristic,
            onlyInformedSeeds = onlyInformedSeeds,
            onlyHeuristicSeeds = onlyHeuristicSeeds,
        )
    }
}
