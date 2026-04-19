# AI Agent Architecture Workshop: Municipal Civic Assistant

A two-day, hands-on workshop for college-level Machine Learning students. Students build an AI-powered civic routing assistant — powered by the **Claude API (Anthropic)** — that tells residents which level of government handles a given service and what to do next.

---

## Overview

The capstone challenge is the **Kitchener-Waterloo Municipal Challenge**: given a resident's free-text question (e.g., *"Who handles garbage pickup?"*), the agent must return a structured JSON response identifying the correct jurisdiction (City, Region, Province, Federal), the responsible government body, and actionable next steps.

The project demonstrates three tiers of AI sophistication:

| Tier | Approach | Grounding |
|------|----------|-----------|
| 1 | Prompt-only baseline | Claude's training knowledge |
| 2 | Retrieval-Augmented (RAG) | Local service catalog injected into context |
| 3 | Tool-calling agent | Claude searches and looks up data via tools |

---

## Repository Structure

```
AI_Agent_Workshop/
├── notebooks/
│   ├── day1/
│   │   └── AI_Agent_Workshop_Day1.ipynb       # Foundations: agents, tools, DVC
│   ├── day2/
│   │   ├── Day2_01_problem_setup_and_data.ipynb
│   │   ├── Day2_02_build_the_service_agent.ipynb
│   │   ├── Day2_03_evaluate_and_pipeline.ipynb
│   │   └── Day2_04_submission_guidelines.ipynb
│   ├── AI_Agent_Workflow.ipynb                # Unified submission notebook
│   └── Claude_API_Setup.ipynb                 # API key setup guide
├── src/                                        # Shared Python modules
│   ├── schema.py        # Input/output TypedDicts + JSON schema
│   ├── retrieval.py     # Tokenization, catalog normalization, RAG
│   ├── tools.py         # Tool functions + Anthropic tool declarations
│   ├── agent.py         # Claude agent (Tier 1 / 2 / 3)
│   ├── evaluation.py    # Metrics, rubric, batch scoring
│   └── pipeline.py      # End-to-end pipeline orchestration
├── scripts/
│   ├── prepare_data.py      # DVC Stage 1: normalize catalog
│   ├── run_agent_eval.py    # DVC Stage 2: run evaluation
│   └── report_metrics.py    # DVC Stage 3: generate report
├── data/
│   ├── service_catalog.csv  # 15 municipal services (source of truth)
│   └── service_catalog.json
├── eval/
│   └── service_eval_set.csv # 8 benchmark questions with ground truth
├── artifacts/               # DVC-managed generated outputs
├── images/
├── dvc.yaml                 # DVC pipeline definition
├── params.yaml              # Pipeline hyperparameters
├── requirements.txt
└── README.md
```

---

## Prerequisites

- Python 3.11+
- An **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com)
- Git + DVC

---

## Quickstart

**1. Clone the repository**

```bash
git clone https://github.com/muthuacumen/AI_Agent_Workshop.git
cd AI_Agent_Workshop
```

**2. Create a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set your API key**

Create a `.env` file in the project root (never commit this file):

```
ANTHROPIC_API_KEY=sk-ant-...
```

Or export it in your shell:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**5. Run the DVC pipeline**

```bash
dvc repro
dvc metrics show
```

**6. Open the unified notebook**

```bash
jupyter notebook notebooks/AI_Agent_Workflow.ipynb
```

---

## DVC Pipeline

Three automated stages:

```
data/service_catalog.csv
        │
        ▼
  prepare_data.py  ──►  artifacts/service_catalog.cleaned.json
                                      │
                                      ▼
                         run_agent_eval.py  ──►  artifacts/metrics.json
                                                          │
                                                          ▼
                                              report_metrics.py  ──►  artifacts/metrics_report.md
```

Parameters in `params.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retrieval.top_k` | 3 | Number of catalog results to retrieve |
| `agent.model_name` | `claude-sonnet-4-6` | Claude model for agent calls |
| `agent.use_tool_calling` | `true` | Enable Tier 3 tool-calling |
| `evaluation.max_examples` | 100 | Max evaluation examples |

---

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| Jurisdiction Accuracy | Correct level of government (City/Region/Province/Federal) |
| Responsible Body Accuracy | Exact government body name match |
| Format Compliance Rate | All 7 schema fields present and valid |
| Avg Reasoning Quality | Heuristic score for explanation completeness (0-1) |
| Source Presence Rate | At least one source URL included |

---

## Day 1 — Foundations

Covers the architectural shift from static ML models to dynamic LLM agents:
- Agentic architecture patterns (Router, Retrieval, Tool-calling, Planner, Critique)
- Prompt engineering and structured output
- Claude function calling / tool use
- DVC for ML pipeline reproducibility

## Day 2 — The Municipal Challenge

Four progressive notebooks:

| Notebook | Focus |
|----------|-------|
| Day2_01 | Problem framing, schema definition, data normalization |
| Day2_02 | Building the agent (3 tiers), improving prompts |
| Day2_03 | Evaluation metrics, pipeline orchestration |
| Day2_04 | Submission requirements and validation checklist |

The **unified notebook** (`notebooks/AI_Agent_Workflow.ipynb`) combines all four notebooks into a single cohesive system, suitable for submission.

---

## Submission

The required submission artifact is `notebooks/AI_Agent_Workflow.ipynb` with all cells executed and saved, pushed to a public GitHub repository named `AI_Agent_Workshop`.

See `notebooks/day2/Day2_04_submission_guidelines.ipynb` for the full checklist.

---

## Team

- Prajesh Bhatt
- KevinKumar Patel 
- Muthuraj Jayakumar

---

*Built with the Claude API (Anthropic) and DVC.*
