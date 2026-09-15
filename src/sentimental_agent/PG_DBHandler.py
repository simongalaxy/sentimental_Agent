import json
import os
import pandas as pd
from datetime import datetime
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import List, Tuple
from pprint import pformat

from src.sentimental_agent.Settings import settings
from src.sentimental_agent.logger import Logger


class PG_DBHandler:
    def __init__(self, logger: Logger):
        # logger settings.
        self.logger = logger
         
        # neon database settings.
        self.conn_str = settings.neon_connection_str
        self.db_name = settings.pgdatabase
        
        # Create persistent connection with autocommit
        self.conn = psycopg2.connect(self.conn_str)
        self.conn.autocommit = True
        
        self.logger.info(f"DBHandler initialized and connected to {self.db_name}")
        
        # ensure database and table was created.
        self._ensure_database_exists()
        self._create_table()


    # check whether the database exists.
    def _ensure_database_exists(self) -> None:
        """
        Check if a PostgreSQL database exists. 
        If not, create it. If yes, do nothing.
        """

        # Step 1 — connect to default 'postgres' database
        self.conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with self.conn.cursor() as cur:
            # Step 2 — check if database exists
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (self.db_name,))
            exists = cur.fetchone()

            if exists:
                print(f"Database '{self.db_name}' already exists — skipping creation.")
            else:
                print(f"Database '{self.db_name}' does not exist — creating now...")
                cur.execute(f'CREATE DATABASE "{self.db_name}";')
                print(f"Database '{self.db_name}' created successfully.")

        return

    # create table when needed.
    def _create_table(self) -> None:
        create_table_query = """
        CREATE TABLE IF NOT EXISTS Reviews ( 
            -- Changed AUTO_INCREMENT to SERIAL
            review_id SERIAL PRIMARY KEY, 
            product_location_name VARCHAR(255) NOT NULL, 
            category VARCHAR(100), 
            platform VARCHAR(100), 
            avg_rating DECIMAL(3, 2), 
            total_reviews INT DEFAULT 0, 
            total_helpful_votes INT DEFAULT 0, 
            sentiment VARCHAR(20), 
            review_text TEXT 
        );
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_query)
            self.logger.info("Table - Reviews created (or already exists)")
        except Exception as e:
            self.logger.error(f"Failed to create table: {e}")
            raise
        
        return

    # ---------------------------------------------------------
    # upsert a news article
    # ---------------------------------------------------------
   
    # just add raw data of job info to database.
    def upsert_reviews(self, item: dict) -> int | None:
        """Insert or update a news item. Returns the review_id on success."""
        insert_query = """
        INSERT INTO Reviews (
            product_location_name, 
            category, 
            platform, 
            avg_rating, 
            total_reviews, 
            total_helpful_votes, 
            sentiment, 
            review_text
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (review_text)
        DO UPDATE SET
            product_location_name = EXCLUDED.product_location_name,
            category = EXCLUDED.category,
            platform = EXCLUDED.platform,
            avg_rating = EXCLUDED.avg_rating,
            total_reviews = EXCLUDED.total_reviews,
            total_helpful_votes = EXCLUDED.total_helpful_votes,
            sentiment = EXCLUDED.sentiment
        RETURNING review_id; -- Added RETURNING clause to get the primary key
        """

        # Safeguard fallback values to safely match column data types
        values = (
            item.get("Product/Location_Name"),
            item.get("category"),
            item.get("Platform"),
            item.get("Avg_Rating"),
            item.get("Total_Reviews"),
            item.get("Total_Helpful_Votes"),
            item.get("sentiment"),
            item.get("Review_Text"),
        )

        # Get a safe identifier for error logs (using a fallback default string)
        item_log_id = item.get("id") or "Unknown ID"

        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(insert_query, values)
                row = cur.fetchone()

                if row:
                    # Match your schema's primary key name ('review_id')
                    inserted_id = row["review_id"] 
                    self.logger.info(f"Upserted review with id - {inserted_id}")
                    return inserted_id
                else:
                    self.logger.info(f"No row returned for review with id - {item_log_id}")
                    return None

        except Exception as e:
            # Fixed attribute access crash: item.id -> item_log_id
            self.logger.error(f"Error inserting news with id - {item_log_id}: {e}")
            return None


