# AI Agent Architecture Workshop: Municipal Civic Assistant

Welcome to the AI Agent Architecture Workshop. This repository contains the Jupyter Notebooks, datasets, and ML pipeline configurations for a two-day active learning module designed for college-level Machine Learning students. 

In this workshop, students will transition from traditional predictive machine learning to **agentified AI development**, using the Google Gemini API stack to build an autonomous system capable of reasoning, planning, and executing tool calls.

## 📖 Overview
The capstone of this workshop is the **Kitchener-Waterloo Municipal Challenge**. Students will build an AI-powered civic assistant that helps residents understand which level of government (City, Region, Province, or Federal) is responsible for specific services and guides them through the correct next steps.

## 🛠️ Prerequisites
Before starting the workshop, ensure you have the following installed and configured:
* **Python 3.10+** (managed via Miniconda/Anaconda)
* **Visual Studio Code** (with the Jupyter extension)
* **Data Version Control (DVC)**
* **Google Gemini API Key:** You must generate a free-tier API key via [Google AI Studio](https://aistudio.google.com/).

## 🗓️ Workshop Structure

### Day 1: Foundations of Agentified Development
Day one focuses on the architectural shift from static ML models to dynamic LLM Agents.
* **Concepts Covered:**
  * Introduction to Agentic Architecture (Memory, Planning, Action).
  * NLP basics and prompt engineering.
  * **Function Calling (Tool Use):** Teaching the Gemini model to execute Python code to retrieve external data.
  * Introduction to ML Pipelines and reproducibility using Data Version Control (DVC).
* **Key Files:** * `Day1_API_Setup.ipynb`: Secure environment setup and API authentication.
  * `Day1_Intro_Agents.ipynb`: Foundational tutorial on tool use and Gemini orchestration.

### Day 2: The Municipal Challenge (Kitchener/Waterloo)
Day two is a hands-on lab where students apply Day 1 concepts to a real-world civic problem. Students will build a routing agent that queries local open data portals to answer resident questions (e.g., *"Who do I contact about garbage pickup?"* or *"Is childcare a city or provincial service?"*).
* **Concepts Covered:**
  * Complex tool chaining and routing logic.
  * Ingesting and querying real-world municipal datasets.
  * Managing datasets and prompt versioning using DVC.
* **Data Sources Utilized:**
  * City of Kitchener Open Data Portal
  * Region of Waterloo Open Data Portal
  * Ontario Data Catalogue
  * Government of Canada Open Government Portal

## 🚀 Getting Started
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ProfEspinosaAIML/AI_Agent_Workshop.git](https://github.com/your-username/AI_Agent_Workshop.git)
   cd AI_Agent_Workshop