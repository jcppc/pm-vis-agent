# PM-VIS-AGENT

**Leveraging Large Language Models for Agentic Process Analytics Assistants: Assessing Accuracy in Process Mining Tasks**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Preprint](https://img.shields.io/badge/Preprint-Elsevier-orange.svg)](#citation)

> **Authors:** Diogo Reis<sup>a</sup>, João Caldeira<sup>b,*</sup>, Marta Jesus<sup>a</sup>
>
> <sup>a</sup> NOVA Information Management School (NOVA IMS), Universidade Nova de Lisboa, Portugal  
> <sup>b</sup> COPELABS, ECATI, Lusófona University, Lisbon, Portugal  
> <sup>*</sup> Corresponding author — joao.caldeira@lusofona.pt

---

## Overview

This repository contains the source code, experimental pipeline, and Streamlit-based tool accompanying the paper *"Leveraging Large Language Models for Agentic Process Analytics Assistants: Assessing Accuracy in Process Mining Tasks"*, submitted to Elsevier.

The study introduces a framework that:

1. **Defines a process analytics question taxonomy** spanning six dimensions — Case Analysis (Q1), Activity Analysis (Q2), Resource Analysis (Q3), Temporal Analysis (Q4), Discovery & Variant Analysis (Q5), and What-If & Predictive Analysis (Q7).
2. **Benchmarks eight LLMs** (four commercial + four small open-source) on their ability to generate executable Python code that produces visual analytics from event logs.
3. **Evaluates outputs** using an **LLM-as-Judge** approach (scored 1–10) combined with a **Visualisation Error Rate (VER)** metric.

The best-performing configuration achieved an LLM-as-Judge score of **μ = 8.18, σ = 2.3** with a VER of **25%** (GPT-4.1, six-shot prompting).

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐     ┌────────────┐
│  NL Question │────▶│  Prompt Building  │────▶│   LLM (Code   │────▶│  Sandbox   │
│              │     │  (Zero/Few/CoT)   │     │  Generation)  │     │  Execution │
└─────────────┘     └──────────────────┘     └───────────────┘     └─────┬──────┘
                                                                         │
                                              ┌───────────────┐         │
                                              │  Evaluation   │◀────────┘
                                              │  (LLM-Judge   │   Code + Output
                                              │   + VER)      │   + Visualisation
                                              └───────────────┘
```

---

## Models Evaluated

### Generation Models

| Class | Model | Release | Context Window |
|-------|-------|---------|----------------|
| Commercial | GPT-4o | Nov 2024 | 128K |
| Commercial | GPT-4.1 | Apr 2025 | 1M |
| Commercial | Claude-3.5-Sonnet | Jan 2024 | 200K |
| Commercial | Claude-3.7-Sonnet | Feb 2025 | 200K |
| Small Open-Source | Qwen2.5-Coder-3B | Nov 2024 | 32K |
| Small Open-Source | Qwen2.5-Coder-7B | Nov 2024 | 32K |
| Small Open-Source | Qwen2.5-3B | Jan 2025 | 32K |
| Small Open-Source | Qwen2.5-7B | Jan 2025 | 128K |

### Evaluation Model

| Model | Release | Context Window |
|-------|---------|----------------|
| Gemini-2.5-Flash | Apr 2025 | 1M |

---

## Prompting Strategies

| Strategy | Description | Tokens |
|----------|-------------|--------|
| **Zero-Shot** | Direct task instruction, no examples | ~161 |
| **Four-Shot** | Includes DFG, histogram, bar chart, and heatmap examples | ~1,223 |
| **Six-Shot** | Extends four-shot with ML feature importance and what-if simulation | ~2,210 |
| **Chain-of-Thought** | Step-by-step reasoning (columns → processing → visualisation → insight) | ~945 |

---

## Datasets

All datasets are sourced from [4TU.ResearchData](https://data.4tu.nl/) in XES format.

| Dataset | Events | Cases | Activities | Resources |
|---------|--------|-------|------------|-----------|
| [Sepsis Cases](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639) | 15,214 | 1,050 | 16 | 26 |
| [Electronic Invoicing](https://data.4tu.nl/collections/BPI_Challenge_2023/6450709) | 309,030 | 20,135 | 9 | 7 |
| [BPI 2020 – Request for Payment](https://data.4tu.nl/articles/dataset/BPI_Challenge_2020_-_Request_For_Payment/12696884) | 36,796 | 6,886 | 19 | 2 |

---

## Key Results

| Class | Model | LLM-as-Judge (μ ± σ) | VER |
|-------|-------|-----------------------|-----|
| Commercial | **GPT-4.1** | **8.18 ± 2.3** | **25%** |
| Commercial | Claude-3.7-Sonnet | 7.79 ± 2.5 | 33% |
| Commercial | Claude-3.5-Sonnet | 7.83 ± 2.4 | 35% |
| Commercial | GPT-4o | 7.35 ± 2.7 | 40% |
| Small Open-Source | Qwen2.5-Coder-7B | 4.52 ± 2.3 | 60% |
| Small Open-Source | Qwen2.5-7B | 4.41 ± 2.6 | 65% |

**Key findings:**
- Six-shot prompting with GPT-4.1 achieved the highest overall score (8.59 ± 2.07, VER 17%).
- Models perform well on tasks solvable with general Python libraries (Q1, Q2, Q4) but struggle with PM4Py-specific tasks (Q5), revealing a domain-knowledge gap.
- Small open-source models (≤ 7B parameters) show significantly lower performance, particularly on discovery and predictive tasks.

---

## Repository Structure

```
pm-vis-agent/
├── process_mining_agent.py      # Core agent class (code generation, execution, evaluation)
├── run_experiments.ipynb         # Experimental pipeline (batch execution across models/prompts)
├── prompts/                      # Prompt templates (zero-shot, few-shot, CoT)
├── data/                         # Event logs (XES format)
├── results/                      # Experimental outputs (code, visualisations, scores)
├── app/                          # Streamlit-based interactive tool
│   └── streamlit_app.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.10+
- API keys for the LLM providers you wish to use (OpenAI, Anthropic, Google, and/or Ollama running locally)

### Setup

```bash
git clone https://github.com/jcppc/pm-vis-agent.git
cd pm-vis-agent
pip install -r requirements.txt
```

### API Keys

```python
from process_mining_agent import ProcessMiningAgent

agent = ProcessMiningAgent(
    generation_prompts=GENERATION_PROMPTS,
    evaluation_prompts=EVALUATION_PROMPTS,
    api_keys={
        'openai': 'sk-...',
        'claude': 'sk-ant-...',
        'google': 'AIza...',
    }
)
```

For local models via [Ollama](https://ollama.com/), ensure the Ollama server is running on `localhost:11434`.

---

## Usage

### Running the Benchmark

```python
results = agent.run_process_mining_batch(
    questions_df=questions,
    output_root="results/sepsis_gpt41_6shot",
    gen_model_type="openai",
    gen_model_name="gpt-4.1-2025-04-14",
    eval_model_type="gemini",
    eval_model_name="gemini-2.5-flash-preview-04-17",
    prompt_name="SIX_SHOT",
    eval_prompt_name="EVALUATION",
    log=event_log,
    df=df,
)
```

Each question produces the following artifacts per output folder:
- `prompt.txt` — full prompt sent to the model
- `codigo.txt` — generated Python code
- `resposta.txt` — execution output
- `nota_final.txt` — LLM-as-Judge score
- `figure_*.png` — generated visualisations

### Interactive Tool (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Upload an XES file, select a model and temperature, type a natural-language question, and receive the generated code and visualisation.

---

## Evaluation Metrics

**LLM-as-Judge (J ∈ [1, 10])** — An evaluator model (Gemini-2.5-Flash) scores the technical quality, logic, and correctness of the generated code relative to the question.

**Visualisation Error Rate (VER)** — Percentage of visualisations that failed to execute:

$$\text{VER} = \frac{E}{T} \times 100$$

where *E* = number of code executions with errors and *T* = total number of expected visualisations.

Each question is evaluated across **5 iterations**; the final score is reported as mean ± standard deviation.

---

## Question Taxonomy

| Category | ID | Focus | Example Topics |
|----------|----|-------|----------------|
| Case Analysis | Q1 | Instance-level metrics | Events per case, case duration distribution, outlier detection |
| Activity Analysis | Q2 | Activity-level metrics | Activity frequency, average duration, waiting times |
| Resource Analysis | Q3 | Resource utilisation | Workload distribution, resource-activity heatmaps |
| Temporal Analysis | Q4 | Time-based patterns | Monthly trends, hourly event distribution, seasonal patterns |
| Discovery & Variant | Q5 | Process structure | DFGs, loop detection, variant comparison, Sankey diagrams |
| What-If & Predictive | Q7 | Decision support | Feature importance for duration prediction, activity removal simulation |

---

## Citation

If you use this framework or code in your research, please cite:

```bibtex
@article{reis2025leveraging,
  title={Leveraging Large Language Models for Agentic Process Analytics Assistants: Assessing Accuracy in Process Mining Tasks},
  author={Reis, Diogo and Caldeira, Jo{\~a}o and Jesus, Marta},
  journal={Preprint submitted to Elsevier},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work was developed at [NOVA IMS](https://www.novaims.unl.pt/) and [COPELABS, Lusófona University](https://copelabs.ulusofona.pt/). The datasets used are publicly available from [4TU.ResearchData](https://data.4tu.nl/).
