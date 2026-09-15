import json
import os
import pandas as pd
from datetime import datetime
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import List, Tuple
from pprint import pformat

from src.Util.Settings import settings
from src.Util.logger import Logger
from src.Data.DataClasses import NewsItem, ParsedQuery

class PG_DBHandler:
    def __init__(self, logger: Logger):
        # logger settings.
        self.logger = logger
         
        # neon database settings.
        self.conn_str = settings.neon_connection_str
        self.db_name = settings.pgdatabase

        # search result export path settings.
        self.searching_results_path = settings.searching_results_path
        
        # Create persistent connection with autocommit
        self.conn = psycopg2.connect(self.conn_str)
        self.conn.autocommit = True
        
        self.logger.info(f"DBHandler initialized and connected to {self.db_name}")
        
        # ensure database and table was created.
        self._ensure_database_exists()
        self._create_table()

        # Create search results folder if it doesn't exist
        self._create_folder()
            
    # methods for report generation.
    def _create_folder(self):
        os.makedirs(self.searching_results_path, exist_ok=True)

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
        CREATE EXTENSION IF NOT EXISTS vector;
        
        CREATE TABLE IF NOT EXISTS GovPressReleases (
            id TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            content TEXT,
            url TEXT NOT NULL,
            published_date DATE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_query)
            self.logger.info("Table GovNews created (or already exists)")
        except Exception as e:
            self.logger.error(f"Failed to create table: {e}")
            raise
        
        return

    # ---------------------------------------------------------
    # upsert a news article
    # ---------------------------------------------------------
   
    # just add raw data of job info to database.
    def upsert_news(self, item: NewsItem) -> None:
        
        """Insert or update a news item. Returns the id on success."""
        insert_query = """
        INSERT INTO GovPressReleases (
            id, title, content, url, published_date
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            url = EXCLUDED.url,
            published_date = EXCLUDED.published_date
        RETURNING id;
        """

        values = (
            item.id,
            item.title,
            item.content,
            item.url,
            item.published_date
        )

        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(insert_query, values)
                row = cur.fetchone()

                if row:
                    inserted_id = row["id"]
                    self.logger.info(f"Inserted news with id - {inserted_id}")
                    return inserted_id
                else:
                    self.logger.info(f"No row returned for news with id - {item.id}")
                    return None

        except Exception as e:
            self.logger.error(f"Error inserting news with id - {item.id}: {e}")
            # Do NOT raise here if you want the pipeline to continue
            # raise  
            return None


    def query_full_text_search(self, parsed_query: ParsedQuery) -> List[dict]:
        base = """
            SELECT
                id,
                published_date,
                title,
                content,
                url
            FROM GovPressReleases
        """
        
        where_clauses = []
        params = []

        # date range.
        if parsed_query.start_date and parsed_query.end_date:
            where_clauses.append("published_date BETWEEN %s AND %s")
            params.append(parsed_query.start_date)
            params.append(parsed_query.end_date)
        elif parsed_query.start_date or parsed_query.end_date:
            where_clauses.append("published_date = %s")
            params.append(parsed_query.start_date)
        else:
            pass
        
        # build tsquery string
        # 1. Departments.
        if parsed_query.departments:
            depts = [f"({item.replace(" ", " <-> ")})" for item in parsed_query.departments]
            tsquery_depts = " | ".join(depts)
            self.logger.info(f"tsquery_depts: {tsquery_depts}")
        else:
            tsquery_depts = None

        # 2. keywords.
        if parsed_query.keywords:
            keywords = [f"({item.replace(" ", " <-> ")})" for item in parsed_query.keywords]
            tsquery_keywords = " | ".join(keywords)
            self.logger.info(f"tsquery_keywords: {tsquery_keywords}")
        else:
            tsquery_keywords = None

        if tsquery_depts and tsquery_keywords:
            tsquery = f"({tsquery_depts}) & ({tsquery_keywords})"
        elif tsquery_depts:
            tsquery = tsquery_depts
        elif tsquery_keywords:
            tsquery = tsquery_keywords
        else:
            tsquery = None

        # keyword Full Text Search filter
        if tsquery:
            where_clauses.append("to_tsvector('english', content) @@ to_tsquery('english', %s)")
            params.append(tsquery)
        
        #--- Assemble WHERE clause ---
        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)
        else:
            where_sql = ""

        # --- Final SQL ---
        sql = base + where_sql + " ORDER BY published_date ASC;"
            
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            search_results = [dict(row) for row in rows]
            # for i, item in enumerate(search_results, start=1):
            #     self.logger.info(f"Data Type of item: {type(item)}")
            #     self.logger.info(f"Query Search Result No.: {i}/{len(search_results)} - /n%s", pformat(item, indent=4))
            self.logger.info(f"Total search results retrieved: {len(search_results)}")
            self.logger.info(f"Third Search results: \n%s", pformat(search_results[3], indent=2))

            return search_results


    def export_to_excel(self, search_results: List[dict], parsed_query: ParsedQuery) -> None:  
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(search_results)

        # Generate filename based on query and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{timestamp}.xlsx"

        # Save to Excel
        df.to_excel(os.path.join(self.searching_results_path, filename), index=False)
        self.logger.info(f"Exported search results to {filename}")

        return