package dev.mattbachmann.scoundroid.data.persistence

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [HighScore::class, WinningGame::class],
    version = 2,
    exportSchema = false,
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun highScoreDao(): HighScoreDao

    abstract fun winningGameDao(): WinningGameDao

    companion object {
        @Volatile
        private var dbInstance: AppDatabase? = null

        private val MIGRATION_1_2 =
            object : Migration(1, 2) {
                override fun migrate(db: SupportSQLiteDatabase) {
                    db.execSQL(
                        """
                        CREATE TABLE IF NOT EXISTS winning_games (
                            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                            seed INTEGER NOT NULL,
                            timestamp INTEGER NOT NULL,
                            finalHealth INTEGER NOT NULL,
                            actionLogJson TEXT NOT NULL
                        )
                        """.trimIndent(),
                    )
                    db.execSQL(
                        "CREATE UNIQUE INDEX IF NOT EXISTS index_winning_games_seed ON winning_games (seed)",
                    )
                }
            }

        fun getDatabase(context: Context): AppDatabase =
            dbInstance ?: synchronized(this) {
                dbInstance ?: Room
                    .databaseBuilder(
                        context.applicationContext,
                        AppDatabase::class.java,
                        "scoundroid_database",
                    ).addMigrations(MIGRATION_1_2)
                    .build()
                    .also { dbInstance = it }
            }
    }
}
