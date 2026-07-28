"""
AI-Powered Data Analyst — Streamlit App
----------------------------------------
Upload a dataset (CSV / TSV / JSON / Excel) and let an LLM-backed agent
run automated EDA, answer natural-language questions about the data, and
generate charts on demand.

Run with:  streamlit run app.py
"""

# ---------------------------------------------------------------------------
# Step 3: Load all modules
# ---------------------------------------------------------------------------
import os
import io
import contextlib
import traceback

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, required for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

SYSTEM_PROMPT = """You are an expert data analyst assistant embedded in a
Streamlit app. A pandas DataFrame has already been loaded for the user.

You have three tools:
- get_dataframe_overview: inspect shape, dtypes, missing values, sample rows.
- run_pandas_code: execute pandas/numpy code to compute an answer. Always
  assign your final answer to a variable called `result`.
- create_visualization: execute matplotlib/seaborn code to draw a chart.

Guidelines:
- Always call get_dataframe_overview first if you are unsure about column
  names or types before writing analysis code.
- Prefer using the tools over guessing; never fabricate numbers.
- Keep code simple, use only pandas, numpy, matplotlib, seaborn (already
  imported in the execution environment as pd, np, plt, sns) and the
  dataframe `df`. Do not import other packages, read/write files, or use
  the network.
- After using tools, answer the user in clear, concise natural language.
  Summarize numeric results instead of just dumping raw output.
"""


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_defaults = {
    "df": None,
    "filename": None,
    "chat_history": [],     # list of {"role", "content", "figure"}
    "last_figure": None,    # scratch slot tools write into
    "eda_report": None,     # cached AI narrative from the last EDA run
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Sidebar: configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

provider = st.sidebar.selectbox("LLM Provider", ["Google Gemini", "Groq"])

google_api_key = st.sidebar.text_input(
    "Google API Key",
    type="password",
    value=os.environ.get("GOOGLE_API_KEY", ""),
    help="Required if using Gemini. Get one at https://aistudio.google.com/apikey",
)
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=os.environ.get("GROQ_API_KEY", ""),
    help="Required if using Groq. Get one at https://console.groq.com/keys",
)

gemini_model_name = st.sidebar.text_input("Gemini model", value="gemini-2.0-flash")
groq_model_name = st.sidebar.text_input("Groq model", value="llama-3.3-70b-versatile")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Reset conversation"):
    st.session_state.chat_history = []
    st.session_state.last_figure = None
    st.rerun()

st.sidebar.caption("API keys are used only for this session and are never stored or logged.")


# ---------------------------------------------------------------------------
# Step 4: Model creation
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_llm(provider: str, google_key: str, google_model: str, groq_key: str, groq_model: str):
    """Create and cache the chat model for the selected provider."""
    if provider == "Google Gemini":
        if not google_key:
            raise ValueError("Please provide a Google API key in the sidebar.")
        return ChatGoogleGenerativeAI(model=google_model, google_api_key=google_key, temperature=0)
    else:
        if not groq_key:
            raise ValueError("Please provide a Groq API key in the sidebar.")
        return ChatGroq(model=groq_model, api_key=groq_key, temperature=0)


# ---------------------------------------------------------------------------
# Step 5: Agent creation — tools + agent
# ---------------------------------------------------------------------------
def _exec_namespace():
    return {"pd": pd, "np": np, "plt": plt, "sns": sns, "df": st.session_state.df}


@tool
def get_dataframe_overview() -> str:
    """Return a text overview of the uploaded dataframe: shape, dtypes,
    missing value counts, sample rows, and summary statistics. Use this
    tool first to understand the dataset before writing analysis code."""
    df = st.session_state.df
    if df is None:
        return "No dataframe has been loaded yet."
    buf = io.StringIO()
    df.info(buf=buf)
    missing = df.isnull().sum()
    missing_str = missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values."
    return (
        f"Shape: {df.shape}\n\n"
        f"Info:\n{buf.getvalue()}\n\n"
        f"Missing values:\n{missing_str}\n\n"
        f"Sample rows:\n{df.head(5).to_string()}\n\n"
        f"Summary statistics:\n{df.describe(include='all').to_string()}"
    )


@tool
def run_pandas_code(code: str) -> str:
    """Execute python/pandas code against the uploaded dataframe (available
    as `df`) to compute an analytical answer. pandas is `pd`, numpy is `np`.
    Assign your final answer to a variable named `result`. Do not read or
    write files, and do not import extra packages. Use this for computation
    only, not for plots (use create_visualization for plots)."""
    if st.session_state.df is None:
        return "No dataframe has been loaded yet."
    ns = _exec_namespace()
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, ns)
        result = ns.get("result", None)
        output = stdout_capture.getvalue()
        if result is not None:
            return f"Result:\n{result}" + (f"\n\nPrinted output:\n{output}" if output else "")
        elif output:
            return f"Printed output:\n{output}"
        return "Code executed but produced no output. Assign the answer to a variable named `result`."
    except Exception as e:
        return f"Error executing code: {e}\n{traceback.format_exc(limit=2)}"


@tool
def create_visualization(code: str) -> str:
    """Execute matplotlib/seaborn code to create a chart from the dataframe
    (available as `df`). pandas is `pd`, numpy is `np`, matplotlib.pyplot is
    `plt`, seaborn is `sns`. Build the plot with plt/sns calls; do not call
    plt.show(). The resulting figure is automatically captured and shown
    to the user."""
    if st.session_state.df is None:
        return "No dataframe has been loaded yet."
    ns = _exec_namespace()
    try:
        plt.close("all")
        exec(code, ns)
        fig = plt.gcf()
        if not fig.get_axes():
            return "No plot was created. Make sure to call a plotting function like plt.plot(...) or sns.barplot(...)."
        st.session_state.last_figure = fig
        return "Visualization created successfully and will be displayed to the user."
    except Exception as e:
        return f"Error creating visualization: {e}\n{traceback.format_exc(limit=2)}"


def build_agent(llm):
    """Assemble the tool-calling agent. Lightweight, so rebuilt each call."""
    tools = [get_dataframe_overview, run_pandas_code, create_visualization]
    return create_agent(model=llm, tools=tools)


def extract_text(response) -> str:
    """Robustly pull the assistant's text out of a create_agent response,
    regardless of whether content is a plain string or a list of content
    blocks (Gemini/Groq return slightly different shapes)."""
    try:
        msg = response["messages"][-1]
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(p for p in parts if p).strip()
        return str(content)
    except Exception:
        return "Sorry, I couldn't parse the agent's response."


def get_agent():
    """Build (or fail loudly on) the LLM + agent using current sidebar settings."""
    llm = build_llm(provider, google_api_key, gemini_model_name, groq_api_key, groq_model_name)
    return build_agent(llm)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(uploaded_file):
    """Load an uploaded Streamlit file (CSV / TSV / JSON / XLSX / XLS)
    into a pandas DataFrame based on its extension."""
    name = uploaded_file.name
    ext = name.split(".")[-1].lower()
    try:
        if ext == "csv":
            return pd.read_csv(uploaded_file)
        elif ext == "tsv":
            return pd.read_csv(uploaded_file, sep="\t")
        elif ext in ("xlsx", "xls"):
            return pd.read_excel(uploaded_file)
        elif ext == "json":
            return pd.read_json(uploaded_file)
        else:
            st.error(f"Unsupported file type: .{ext}")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def perform_eda(df: pd.DataFrame) -> dict:
    """Compute a structured EDA summary for the given dataframe."""
    numeric_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(include=["object", "category"])
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": numeric_df.describe().T if not numeric_df.empty else None,
        "categorical_summary": (
            {col: df[col].value_counts().head(5).to_dict() for col in cat_df.columns}
            if not cat_df.empty else {}
        ),
    }


def generate_ai_insights(agent, df: pd.DataFrame, report: dict) -> str:
    """Ask the agent to write a short natural-language insights report
    based on the computed EDA summary."""
    sample = df.sample(min(5, len(df))).to_string()
    prompt = f"""{SYSTEM_PROMPT}

Based on the dataset sample and EDA summary below, write a concise report
(short paragraphs + bullet points) covering: what the dataset appears to
represent, data quality issues (missing values, duplicates), notable
patterns, and 3-5 suggested next analysis steps. Do not write code, only
the written report.

Sample rows:
{sample}

Shape: {report['shape']}
Missing values (top 10): {dict(list(report['missing_values'].items())[:10])}
Duplicate rows: {report['duplicates']}
"""
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return extract_text(response)


# ---------------------------------------------------------------------------
# Step 6: App UI
# ---------------------------------------------------------------------------
st.title("📊 AI-Powered Data Analyst")
st.caption("Upload a dataset and let an AI agent explore, analyze, and visualize it for you.")

uploaded_file = st.file_uploader(
    "Upload a CSV, TSV, JSON, or Excel file",
    type=["csv", "tsv", "xlsx", "xls", "json"],
)

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    _df = load_dataset(uploaded_file)
    if _df is not None:
        st.session_state.df = _df
        st.session_state.filename = uploaded_file.name
        st.session_state.eda_report = None
        st.session_state.chat_history = []
        st.success(f"Loaded **{uploaded_file.name}** — {_df.shape[0]} rows × {_df.shape[1]} columns")

if st.session_state.df is not None:
    df = st.session_state.df

    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    tab_eda, tab_chat = st.tabs(["📋 Automated EDA", "💬 Chat with your Data"])

    # ------------------------------------------------------------------ #
    # Tab 1: Automated EDA
    # ------------------------------------------------------------------ #
    with tab_eda:
        run_eda = st.button("Run Automated EDA", type="primary")

        if run_eda:
            with st.spinner("Analyzing dataset..."):
                report = perform_eda(df)

                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", report["shape"][0])
                col2.metric("Columns", report["shape"][1])
                col3.metric("Duplicate rows", report["duplicates"])

                st.subheader("Column Types & Missing Values")
                st.dataframe(
                    pd.DataFrame({
                        "dtype": report["dtypes"],
                        "missing_values": report["missing_values"],
                        "missing_%": report["missing_pct"],
                    }),
                    use_container_width=True,
                )

                if report["numeric_summary"] is not None:
                    st.subheader("Numeric Summary")
                    st.dataframe(report["numeric_summary"], use_container_width=True)

                if report["categorical_summary"]:
                    st.subheader("Top Categorical Values")
                    for col, vals in report["categorical_summary"].items():
                        st.write(f"**{col}**")
                        st.write(vals)

                st.subheader("🧠 AI Insights")
                try:
                    agent = get_agent()
                    with st.spinner("Generating AI insights..."):
                        insights = generate_ai_insights(agent, df, report)
                    st.session_state.eda_report = insights
                    st.markdown(insights)
                except Exception as e:
                    st.warning(f"Could not generate AI insights: {e}")
        elif st.session_state.eda_report:
            st.markdown(st.session_state.eda_report)
        else:
            st.info("Click **Run Automated EDA** to generate a full report.")

    # ------------------------------------------------------------------ #
    # Tab 2: Chat
    # ------------------------------------------------------------------ #
    with tab_chat:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("figure") is not None:
                    st.pyplot(msg["figure"])

        user_query = st.chat_input(
            "Ask a question about your data (e.g. 'What are the top 5 products by sales?')"
        )

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query, "figure": None})
            with st.chat_message("user"):
                st.markdown(user_query)

            try:
                agent = get_agent()
                st.session_state.last_figure = None
                with st.spinner("Thinking..."):
                    response = agent.invoke({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_query},
                        ]
                    })
                answer = extract_text(response)
                fig = st.session_state.last_figure
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    if fig is not None:
                        st.pyplot(fig)
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "figure": fig})
            except Exception as e:
                error_msg = f"⚠️ Something went wrong: {e}"
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "figure": None})
else:
    st.info("👆 Upload a dataset to get started.")
