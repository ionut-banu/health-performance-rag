# Health & Performance RAG Assistant

> Ask science-backed questions about health, performance, and nutrition — answers grounded in Huberman Lab, Andy Galpin's *Perform* podcast, and real recipe data.

## What is this?

This is a Retrieval-Augmented Generation (RAG) application that lets you query evidence-based health and performance content using natural language. Instead of guessing what science says about sleep, recovery, or nutrition, you get answers grounded in actual transcripts from two leading sources — Andrew Huberman's neuroscience-focused podcast and Andy Galpin's performance science podcast — paired with real recipe and nutrition data so recommendations are actionable, not just theoretical.

Built as the final project for [DataTalks.Club's LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp).

## Why this project?

Most health advice online is either oversimplified clickbait or buried in hours of podcast audio you don't have time to listen to. This project makes that knowledge searchable and queryable — ask a specific question, get an answer with sources, and (where relevant) a recipe that fits.

## Data sources

- **Huberman Lab** podcast transcripts — neuroscience, sleep, hormones, nutrition science
- **Andy Galpin's *Perform*** podcast transcripts — exercise physiology, recovery, performance
- **RecipeNLG** + **USDA FoodData Central** — recipes grounded in real nutritional data

## Tech stack

- **Orchestration:** Apache Airflow
- **Retrieval:** Hybrid search (vector + keyword), PGVector
- **Evaluation:** Retrieval metrics + LLM-as-a-judge
- **Interface:** *(TBD — Streamlit/FastAPI)*
- **Monitoring:** *(TBD)*

## Status

🚧 Actively in development, built module-by-module alongside LLM Zoomcamp 2026.
