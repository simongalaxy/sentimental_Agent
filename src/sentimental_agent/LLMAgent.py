import os
import json
import asyncio
import instructor
import pandas as pd
from datetime import datetime
from openai import AsyncOpenAI
from pprint import pformat
from typing import List, Any, Dict, AsyncGenerator
from pathlib import Path
import textwrap


from src.sentimental_agent.Settings import settings
from src.sentimental_agent.logger import Logger
from src.sentimental_agent.DataClasses import ClassifiedView, DataProfile, Category

class LLMAgent:
    def __init__(self, logger: Logger):
        # initiate logger.
        self.logger=logger
        
        # cloud ollama llm settings.
        self.model_name = settings.ollama_cloud_model
        self.base_url = settings.ollama_base_url
        self.api_key = settings.ollama_api_key
        self.client = instructor.from_openai(
            AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            ),
            mode=instructor.Mode.JSON_SCHEMA,  # Forces JSON extraction compatible with Ollama
        )

        # local state variables to hold the original query, parsed query, extracted data, and summary.
        self.original_query = None
        self.parsed_query = None
        self.extracted_datas = None

        # summary report path settings.
        self.report_path = settings.report_path

        # Create log folder if it doesn't exist
        self._create_folder()

        
    # methods for report generation.
    def _create_folder(self):
        os.makedirs(self.report_path, exist_ok=True)

 
    def _write_report(self, markdown: str) -> str:
        """Write the generated markdown report to a text file with a timestamped filename, and return the filename."""
        
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # generate filename by daily press release url.
        filename = f"Media_Summary_Report-{current_timestamp}.md"
        
        # generate report in text file.
        with open(os.path.join(self.report_path, filename), "w", encoding="utf-8") as file:
            file.write(markdown + "\n")
            
        return filename


    # 💡 抽離出的共用底層非同步方法
    async def _call_llm(self, model: str, messages: list, response_model: type, **kwargs) -> Any:
        """統一管理所有對 Ollama 的非同步請求與異常處理"""
        try:
            return await self.client.create(
                model=model,
                messages=messages,
                response_model=response_model,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"LLM API call failed [Model: {model}]: {e}")
            raise e


    def _consolidated_views(self, dataprofile: DataProfile) -> str:
        
        parts = []
        for view in dataprofile.sample_views:
            parts.append(f"- {view}")
        sample_views_str ="\n".join(parts)
        consolidated_sample_views = f"Product: {dataprofile.product}\n\nfiltered_views: {sample_views_str}\n"
        # self.logger.info(f"Consolidated sample views: \n%s", consolidated_sample_views)
        
        return consolidated_sample_views


    async def generate_category(self, dataprofile: DataProfile) -> list[str]:
        self.logger.info("Start consolidating categories from views.")
        
        consolidated_sample_views = self._consolidated_views(dataprofile=dataprofile)

        # 💡 FIX 1: Explicitly instruct the LLM on the JSON key it MUST use
        system_instruction = """
        You are a precise data categorization assistant. 
        Analyze the provided list of views and determine a list of high-level category names to group them logically.
        
        CRITICAL: 
        1. You must return a JSON object with a single key named "category" containing a flat list of strings.
        2. DO NOT include "Positive", "Negative", or any general sentiment indicators as a category.
        
        Example Format:
        {
        "category": ["Category A", "Category B", "Category C"]
        }
        """

        user_prompt = f"""
        Categorize the following views into distinct category names.

        Views to process:
        {consolidated_sample_views}
        """

        try:
            # Call LLM with the structured schema wrapper
            category_data = await self._call_llm(
                model=self.model_name,
                messages=[
                    { "role": "system", "content": system_instruction },
                    { "role": "user", "content": user_prompt }
                ],
                response_model=Category, # Enforces mapping validation
                temperature=0.0,
                timeout=15.0,
                max_retries=3
            )
            
            self.logger.info(f"Category schema successfully generated: \n%s", 
                            pformat(category_data.model_dump(by_alias=True), indent=2))
            
            # 💡 FIX 2: Safely extract and return the internal list of strings 
            return category_data.category

        except Exception as e:
            self.logger.error(f"Failed to generate structured category: {str(e)}")
            # Provide a graceful fallback array to prevent application crashes
            return ["General Feedback"]


    async def _categorize_views(self, data: dict, category: List[str]) -> dict[str]:
        self.logger.info("Start consolidating category from views.")

        view = f"Product: {data.get('Product/Location_Name')}\n\nView: {data.get('Review_Text')}\n"
        
        system_instruction = """
        You are a precise view analyst. Your job is to identify the sentiment and category of the view.
        CRITICAL: 
        1. You must provide your 'category' output as a strict JSON array of strings (e.g. ["CategoryName"]), even if there is only one category.
        2. Only classify into category provided in the context. 
        """

        user_prompt = f"""
        Identify the sentiment and category of the following view.

        Available Category: 
        {category}

        View:
        {view}
        """

        try:
            # 乾淨地呼叫抽離後的非同步方法
            response = await self._call_llm(
                model=self.model_name,
                messages=[
                    { "role": "system", "content": system_instruction },
                    { "role": "user", "content": user_prompt }
                ],
                response_model=ClassifiedView,
                context={"allowed_category": category},
                temperature=0.0,
                timeout=15.0,
                max_retries=3
            )

            if response is not None:
                # self.logger.info(f"Classified view generated: \n%s", pformat(response.model_dump(by_alias=True), indent=2))
                data["sentiment"] = response.sentiment
                data['category'] = response.category
            else:
                data["sentiment"] = "Neutral"
                data['category'] = []
                self.logger.error("LLM returned None response.")

        except Exception as e:
            self.logger.error(f"Failed to process view due to error: {e}")
            data["sentiment"] = "Neutral"
            data['category'] = []

        self.logger.info(f"Processed view: \n%s", pformat(data, indent=2))
        self.logger.info("#" * 50)

        return data


    async def categorize_all_views(self, dataprofile: DataProfile) -> AsyncGenerator[List[dict[str]]]:

        semaphore = asyncio.Semaphore(3)   # Tune this (3~6) based on your GPU/RAM

        async def bounded_extract(data: dict):
            async with semaphore:
                return await self._categorize_views(data=data, category=dataprofile.category)

        tasks = [bounded_extract(data) for data in dataprofile.unique_views.to_dict(orient="records")]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out or handle exceptions if an execution failed
        valid_results = [r for r in results if isinstance(r, dict)]

        # dataprofile.processed_df = pd.DataFrame(results)

        # self.logger.info(f"Processed dataframe: \n%s", dataprofile.processed_df.head(20))
        # self.logger.info("#" * 50)

        yield valid_results


    # async def generate_summary(self, search_results: List[dict]) -> None:
    #     self.logger.info("Start generating summary from search results.")
        
    #     # prepare the documents from search results in pgvector.
    #     articles = self._format_articles(rows=search_results)

    #     system_prompt = """
    #     You are an expert Media Analyst and Policy Researcher. Your task is to synthesize unstructured media coverage and news articles into a structured, highly organized media summary report.

    #     You must strictly organize the output using the following hierarchical structure:
    #     1. Department
    #     2. Topic

    #     For EACH individual topic, you must format the content in this exact order:
    #     - **Background Paragraph(s)**: Provide the baseline context, foundational facts, and initial trigger for the topic.
    #     - **Chronological History**: Present a timeline or sequence of events detailing how the topic evolved over time. Ensure dates or timeframes are explicitly noted.
    #     - **Government Stand**: A concise summary of the official government position, policy statements, or actions regarding this topic.
    #     - **Opposite Views**: A summary of criticism, public backlash, opposition party statements, or alternative viewpoints.

    #     CRITICAL CONSTRAINTS:
    #     - Do not mix the Government Stand and Opposite Views into the chronological history; they must remain at the very end of each topic section.
    #     - Maintain a neutral, analytical tone.
    #     - Do not add conversational fluff or introductory meta-commentary (e.g., "Sure, here is your summary"). Start directly with the first department.
    #     """

    #     user_prompt = f"""
    #     Please analyze the following raw media reports and generate a structured media summary according to your system instructions with the following requirements:

    #     1. **Content Guidelines**:
    #     - Cover all significant points from the provided articles.
    #     - Be detailed and comprehensive rather than brief; do not omit important information just to keep it short.
    #     - Include key facts, figures, exact dates, names of officials/agencies, and outcomes.
    #     - Show the progression and evolution of the issue over time within its specific section.

    #     2. **Structure & Style**:
    #     - Group the summaries strictly by Department, then by Topic.
    #     - For each Topic, organize the history chronologically (nested inside that topic).
    #     - Ensure each Topic ends with distinct sections for the Government Stand and Opposite Views.
    #     - Maintain a formal, objective, and professional tone.
    #     - Use clear paragraphs. Use bullet points *only* for lists of actions or key chronological events when appropriate.

    #     ### Articles:
    #     {articles}
    #     """


    #     # user_prompt = f"""Below are the full texts of multiple government news articles related to the query.

    #     # Articles:
    #     # {articles}

    #     # Please create a comprehensive **Media Summary Report** with the following requirements:

    #     # 1. **Structure**:
    #     # - Start with a short executive overview (2-4 sentences).
    #     # - Then organize the main body in strict chronological order (earliest events first).
    #     # - Use clear markdown headings and sub-headings (## Date or ## Topic).
    #     # - Group related developments when logical, but never break chronology.

    #     # 2. **Content Guidelines**:
    #     # - Cover all significant points from the provided articles.
    #     # - Be detailed and comprehensive rather than brief.
    #     # - Include key facts, figures, dates, names of officials/agencies, and outcomes.
    #     # - Show progression and evolution of the issue over time.
    #     # - If multiple departments or topics are involved, create logical sections while keeping the overall timeline intact.

    #     # 3. **Style**:
    #     # - Formal, objective, and professional tone.
    #     # - Clear paragraphs. Use bullet points only for lists of actions or key outcomes when appropriate.
    #     # - Do not omit important information just to keep it short.

    #     # Write the complete report now."""

    #     try:
    #         # ⚡️ FIX: Bypass instructor processing entirely for raw text extraction
    #         # We access the raw async client via .client attribute underneath instructor
    #         response = await self.client.client.chat.completions.create(
    #             model=self.model_name,
    #             messages=[
    #                 { "role": "system", "content": system_prompt },
    #                 { "role": "user", "content": user_prompt }
    #             ],
    #             timeout=30.0  # Reports can take slightly longer to stream
    #         )
    #         summary = response.choices[0].message.content
    #         self._write_report(markdown=summary)
    #         self.logger.info("#"*50)
    #         self.logger.info(f"Generated Summary: \n%s", summary)
    #         self.logger.info("#"*50)

    #     except Exception as e:
    #         self.logger.error(f"Unstructured summary generation failed: {e}")
    #         raise e

    #     return
