# Agent-Alex
# 📊 AI-Powered Data Analyst

An interactive Streamlit application that turns any uploaded dataset into a fully
automated, AI-generated exploratory data analysis (EDA) dashboard — and lets you
chat with your data in plain English.

Instead of hardcoded charts, an LLM agent (Google Gemini or Groq) **writes real
Python analysis code** — pandas, NumPy, Matplotlib, and Seaborn — tailored to the
exact columns and data types of the uploaded file, which is then executed live
inside the app to render the dashboard along with automatically generated,
data-grounded insights.

---

## ✨ Features

- **📁 Flexible data upload** — CSV, TSV, JSON, and Excel (`.xlsx` / `.xls`) files
- **🧠 AI-Generated EDA Dashboard** — one click generates a complete, dataset-specific analysis covering:
  - Describe (summary statistics) with automatic outlier & skew detection
  - Correlation heatmap with narrated insights on the strongest relationships
  - Univariate analysis (histograms, boxplots, count plots, pie charts)
  - Bivariate analysis (scatter + regression, grouped boxplots)
  - Multivariate analysis (pairplots, hue coloring, groupby aggregations)
  - Time series analysis with trend + rolling mean (auto-detects date columns)
  - Key Takeaways — concrete, number-grounded summary bullets
- **🧾 Transparent & reusable code** — view or download the exact Python code the AI generated for your dashboard
- **🔁 One-click regenerate** — if a generated chart errors out, regenerate instantly
- **💬 Chat with your data** — ask free-form questions; the agent writes and executes pandas code and charts on demand, backed by dedicated tools
- **🔐 Safety guardrails** — generated code is checked against a denylist (no file I/O, network access, `eval`/`exec`, `os`/`sys`/`subprocess`) before it ever runs
- **🔀 Provider-agnostic** — swap between Google Gemini and Groq models from the sidebar, no code changes needed

---

## 🛠️ Tech Stack

| Layer                  | Technology                                                        |
|-------------------------|--------------------------------------------------------------------|
| **UI / App Framework**  | [Streamlit](https://streamlit.io/)                                |
| **LLM Orchestration**   | [LangChain](https://www.langchain.com/) (`langchain`, `langchain-core`) |
| **Agent Framework**     | LangChain `create_agent` (tool-calling agent)                     |
| **LLM Providers**       | Google Gemini (`langchain-google-genai`), Groq (`langchain-groq`) |
| **Data Processing**     | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Visualization**       | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |
| **File Handling**       | `openpyxl` (Excel support)                                        |
| **Language**            | Python 3.10+                                                      |

---

## 🚀 Getting Started

### 1. Clone or download the project
Make sure `app.py` and `requirements.txt` are in the same folder.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get API keys
- **Google Gemini** → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Groq** → [console.groq.com/keys](https://console.groq.com/keys)

You only need a key for the provider you plan to use.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Use it
1. Enter your API key in the sidebar and pick a provider (Gemini or Groq).
2. Upload a dataset (CSV / TSV / JSON / Excel).
3. Click **Generate Advanced EDA Dashboard** to get a full AI-written analysis.
4. Switch to the **Chat with your Data** tab to ask follow-up questions or request custom charts.

---

## 📂 Project Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── generated/            # AI-generated dashboard code is saved here at runtime
└── README.md             # Project documentation
```

---

## ⚠️ Notes

- API keys are used only within your session and are never stored or logged.
- The AI-generated dashboard code is inspected against a safety denylist before execution, but it is still LLM-generated code — review it via the "View / download generated code" panel if you're deploying this beyond local/personal use.
- Chart quality depends on the LLM's code correctly matching your dataset; use **Regenerate** if a section fails.

---

## 👨‍💻 Credits

**Designed and developed by Ankit Mishra**
