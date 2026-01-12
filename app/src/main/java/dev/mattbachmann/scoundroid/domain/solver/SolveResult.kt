package dev.mattbachmann.scoundroid.domain.solver

/**
 * Result of solving a game seed, containing statistics about all possible play paths.
 *
 * @property totalPaths Total number of distinct decision paths explored
 * @property winningPaths Number of paths that lead to victory
 * @property losingPaths Number of paths that lead to defeat
 * @property maxScore The highest score achievable with optimal play
 * @property minWinningScore The lowest score among winning paths (if any)
 * @property averageWinningScore Average score across all winning paths (if any)
 */
data class SolveResult(
    val totalPaths: Long,
    val winningPaths: Long,
    val losingPaths: Long,
    val maxScore: Int,
    val minWinningScore: Int?,
    val averageWinningScore: Double?,
) {
    /**
     * Whether the seed is winnable with optimal play.
     */
    val isWinnable: Boolean
        get() = winningPaths > 0

    /**
     * Win probability assuming uniform random decisions at each choice point.
     * This is the fraction of paths that lead to victory.
     */
    val winProbability: Double
        get() = if (totalPaths > 0) winningPaths.toDouble() / totalPaths else 0.0

    companion object {
        /**
         * Creates a result for a single winning outcome.
         */
        fun win(score: Int) =
            SolveResult(
                totalPaths = 1,
                winningPaths = 1,
                losingPaths = 0,
                maxScore = score,
                minWinningScore = score,
                averageWinningScore = score.toDouble(),
            )

        /**
         * Creates a result for a single losing outcome.
         */
        fun loss(score: Int) =
            SolveResult(
                totalPaths = 1,
                winningPaths = 0,
                losingPaths = 1,
                maxScore = score,
                minWinningScore = null,
                averageWinningScore = null,
            )

        /**
         * Combines multiple solve results from different decision branches.
         */
        fun combine(results: List<SolveResult>): SolveResult {
            if (results.isEmpty()) {
                return SolveResult(
                    totalPaths = 0,
                    winningPaths = 0,
                    losingPaths = 0,
                    maxScore = Int.MIN_VALUE,
                    minWinningScore = null,
                    averageWinningScore = null,
                )
            }

            val totalPaths = results.sumOf { it.totalPaths }
            val winningPaths = results.sumOf { it.winningPaths }
            val losingPaths = results.sumOf { it.losingPaths }
            val maxScore = results.maxOf { it.maxScore }

            val winningResults = results.filter { it.winningPaths > 0 }
            val minWinningScore = winningResults.mapNotNull { it.minWinningScore }.minOrNull()

            val averageWinningScore =
                if (winningPaths > 0) {
                    val totalWinningScore =
                        results.sumOf { result ->
                            (result.averageWinningScore ?: 0.0) * result.winningPaths
                        }
                    totalWinningScore / winningPaths
                } else {
                    null
                }

            return SolveResult(
                totalPaths = totalPaths,
                winningPaths = winningPaths,
                losingPaths = losingPaths,
                maxScore = maxScore,
                minWinningScore = minWinningScore,
                averageWinningScore = averageWinningScore,
            )
        }
    }
}
