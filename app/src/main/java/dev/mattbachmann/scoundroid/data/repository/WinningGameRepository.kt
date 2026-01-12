package dev.mattbachmann.scoundroid.data.repository

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.LogEntry
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.data.persistence.WinningGame
import dev.mattbachmann.scoundroid.data.persistence.WinningGameDao
import kotlinx.coroutines.flow.Flow
import org.json.JSONArray
import org.json.JSONObject

class WinningGameRepository(
    private val winningGameDao: WinningGameDao,
) {
    fun getAllWinningGames(): Flow<List<WinningGame>> = winningGameDao.getAllWinningGames()

    suspend fun getWinCount(): Int = winningGameDao.getWinCount()

    suspend fun saveWinningGame(
        seed: Long,
        finalHealth: Int,
        actionLog: List<LogEntry>,
    ) {
        val json = serializeActionLog(actionLog)
        val winningGame =
            WinningGame(
                seed = seed,
                finalHealth = finalHealth,
                actionLogJson = json,
            )
        winningGameDao.insert(winningGame)
    }

    suspend fun exportAllWinsAsJson(): String {
        val games = winningGameDao.getAllWinningGamesSync()
        val jsonArray = JSONArray()

        for (game in games) {
            val gameObj =
                JSONObject().apply {
                    put("seed", game.seed)
                    put("finalHealth", game.finalHealth)
                    put("timestamp", game.timestamp)
                    put("actions", JSONArray(game.actionLogJson))
                }
            jsonArray.put(gameObj)
        }

        val export =
            JSONObject().apply {
                put("exportedAt", System.currentTimeMillis())
                put("winCount", games.size)
                put("wins", jsonArray)
            }

        return export.toString(2) // Pretty print with 2-space indent
    }

    suspend fun deleteAll() = winningGameDao.deleteAll()

    companion object {
        fun serializeActionLog(actionLog: List<LogEntry>): String {
            val jsonArray = JSONArray()

            for (entry in actionLog) {
                val obj =
                    JSONObject().apply {
                        put("timestamp", entry.timestamp)
                        when (entry) {
                            is LogEntry.GameStarted -> {
                                put("type", "GameStarted")
                            }
                            is LogEntry.RoomDrawn -> {
                                put("type", "RoomDrawn")
                                put("cardsDrawn", entry.cardsDrawn)
                                put("deckSizeAfter", entry.deckSizeAfter)
                                put("roomCards", serializeCards(entry.roomCards))
                            }
                            is LogEntry.RoomAvoided -> {
                                put("type", "RoomAvoided")
                                put("cardsReturned", entry.cardsReturned)
                                put("roomCards", serializeCards(entry.roomCards))
                            }
                            is LogEntry.CardsSelected -> {
                                put("type", "CardsSelected")
                                put("selectedCards", serializeCards(entry.selectedCards))
                                put("cardLeftBehind", serializeCard(entry.cardLeftBehind))
                            }
                            is LogEntry.MonsterFought -> {
                                put("type", "MonsterFought")
                                put("monster", serializeCard(entry.monster))
                                put("weaponUsed", entry.weaponUsed?.let { serializeCard(it) })
                                put("damageBlocked", entry.damageBlocked)
                                put("damageTaken", entry.damageTaken)
                                put("healthBefore", entry.healthBefore)
                                put("healthAfter", entry.healthAfter)
                            }
                            is LogEntry.WeaponEquipped -> {
                                put("type", "WeaponEquipped")
                                put("weapon", serializeCard(entry.weapon))
                                put("replacedWeapon", entry.replacedWeapon?.let { serializeCard(it) })
                            }
                            is LogEntry.PotionUsed -> {
                                put("type", "PotionUsed")
                                put("potion", serializeCard(entry.potion))
                                put("healthRestored", entry.healthRestored)
                                put("healthBefore", entry.healthBefore)
                                put("healthAfter", entry.healthAfter)
                                put("wasDiscarded", entry.wasDiscarded)
                            }
                        }
                    }
                jsonArray.put(obj)
            }

            return jsonArray.toString()
        }

        private fun serializeCard(card: Card): String = "${card.rank.displayName}${card.suit.symbol}"

        private fun serializeCards(cards: List<Card>): JSONArray {
            val arr = JSONArray()
            cards.forEach { arr.put(serializeCard(it)) }
            return arr
        }

        fun deserializeActionLog(json: String): List<LogEntry> {
            val result = mutableListOf<LogEntry>()
            val jsonArray = JSONArray(json)

            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                val timestamp = obj.getLong("timestamp")
                val type = obj.getString("type")

                val entry: LogEntry =
                    when (type) {
                        "GameStarted" -> LogEntry.GameStarted(timestamp)
                        "RoomDrawn" ->
                            LogEntry.RoomDrawn(
                                timestamp = timestamp,
                                cardsDrawn = obj.getInt("cardsDrawn"),
                                deckSizeAfter = obj.getInt("deckSizeAfter"),
                                roomCards = deserializeCards(obj.optJSONArray("roomCards")),
                            )
                        "RoomAvoided" ->
                            LogEntry.RoomAvoided(
                                timestamp = timestamp,
                                cardsReturned = obj.getInt("cardsReturned"),
                                roomCards = deserializeCards(obj.optJSONArray("roomCards")),
                            )
                        "CardsSelected" ->
                            LogEntry.CardsSelected(
                                timestamp = timestamp,
                                selectedCards = deserializeCards(obj.getJSONArray("selectedCards")),
                                cardLeftBehind = deserializeCard(obj.getString("cardLeftBehind")),
                            )
                        "MonsterFought" ->
                            LogEntry.MonsterFought(
                                timestamp = timestamp,
                                monster = deserializeCard(obj.getString("monster")),
                                weaponUsed = obj.optString("weaponUsed", null)?.let { deserializeCard(it) },
                                damageBlocked = obj.getInt("damageBlocked"),
                                damageTaken = obj.getInt("damageTaken"),
                                healthBefore = obj.getInt("healthBefore"),
                                healthAfter = obj.getInt("healthAfter"),
                            )
                        "WeaponEquipped" ->
                            LogEntry.WeaponEquipped(
                                timestamp = timestamp,
                                weapon = deserializeCard(obj.getString("weapon")),
                                replacedWeapon = obj.optString("replacedWeapon", null)?.let { deserializeCard(it) },
                            )
                        "PotionUsed" ->
                            LogEntry.PotionUsed(
                                timestamp = timestamp,
                                potion = deserializeCard(obj.getString("potion")),
                                healthRestored = obj.getInt("healthRestored"),
                                healthBefore = obj.getInt("healthBefore"),
                                healthAfter = obj.getInt("healthAfter"),
                                wasDiscarded = obj.getBoolean("wasDiscarded"),
                            )
                        else -> continue
                    }
                result.add(entry)
            }

            return result
        }

        private fun deserializeCard(str: String): Card {
            // Format: "K♠" or "10♦"
            val suitSymbol = str.last()
            val rankStr = str.dropLast(1)

            val suit =
                Suit.entries.first { it.symbol == suitSymbol.toString() }
            val rank =
                Rank.entries.first { it.displayName == rankStr }

            return Card(suit, rank)
        }

        private fun deserializeCards(arr: JSONArray?): List<Card> {
            if (arr == null) return emptyList()
            return (0 until arr.length()).map { deserializeCard(arr.getString(it)) }
        }
    }
}
