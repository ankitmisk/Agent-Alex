"""
AI-Powered Data Analyst — Streamlit App
----------------------------------------
Upload a dataset (CSV / TSV / JSON / Excel). An LLM agent (Gemini or Groq)
writes a complete, advanced, dataset-specific EDA dashboard as Python code
(pandas/numpy/matplotlib/seaborn/streamlit), which is saved to disk and
executed live to render the dashboard with automatic insights. A separate
chat tab lets you ask free-form questions about the data.

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

GEN_DIR = os.path.join(os.getcwd(), "generated")
os.makedirs(GEN_DIR, exist_ok=True)
GEN_EDA_PATH = os.path.join(GEN_DIR, "generated_eda_dashboard.py")

CHAT_SYSTEM_PROMPT = """You are an expert data analyst assistant embedded in a
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
- If the user asks for a chart, plot, graph, visualization, or to "show" /
  "plot" / "visualize" something, you MUST call the create_visualization
  tool to actually render it. Never just describe what a chart would look
  like in text instead of calling the tool.
- In create_visualization code, always create the figure explicitly first,
  e.g. `plt.figure(figsize=(8,5))`, then call the plotting function
  (plt.*, sns.*, or df.plot), and call `plt.tight_layout()` at the end.
- After using tools, answer the user in clear, concise natural language.
  Summarize numeric results instead of just dumping raw output.
"""

# Operations the AI-generated EDA code is not allowed to use.
_FORBIDDEN_TOKENS = [
    "import os", "os.", "sys.", "subprocess", "shutil", "socket",
    "open(", "eval(", "exec(", "__import__", "input(", "requests",
    "urllib", "pathlib", "Path(", "pickle", "globals(", "locals(",
    "compile(", "importlib", "del ",
]


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_defaults = {
    "df": None,
    "filename": None,
    "chat_history": [],      # list of {"role", "content", "figure"}
    "last_figure": None,     # scratch slot chat tools write into
    "eda_code": None,        # last AI-generated dashboard source
    "eda_code_error": None,  # traceback if the generated code failed
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
# AI-generated advanced EDA dashboard
# ---------------------------------------------------------------------------
def build_eda_codegen_prompt(df: pd.DataFrame) -> str:
    """Craft the prompt that asks the LLM to write the full EDA dashboard
    function, giving it exact column names/dtypes so it never hallucinates
    a column that doesn't exist."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    return f"""You are an expert Python data analyst and Streamlit developer.
Write ONE complete Python function called `run_eda_dashboard(df)` that takes
a pandas DataFrame `df` and renders a complete, advanced, automated EDA
dashboard directly inside a Streamlit app.

Assume these are already imported and available in scope — do NOT add your
own import statements: pandas as pd, numpy as np, matplotlib.pyplot as plt,
seaborn as sns, streamlit as st.

Do NOT read or write files, use the network, use os/sys/subprocess, use
eval/exec, or call plt.show().

Dataset structure (use these exact column names, never invent new ones):
Shape: {df.shape}
Columns and dtypes: {df.dtypes.astype(str).to_dict()}
Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}
Sample rows:
{df.sample(min(5, len(df))).to_string()}

The function must include ALL of the following sections, each under its own
st.subheader(...), and EACH section wrapped in its own
`try:` / `except Exception as e: st.warning(f"Skipped <section>: {{e}}")`
block so that one failing section never stops the rest from rendering:

1. Describe — st.dataframe of df.describe(include='all').T, followed by a
   short st.markdown bullet list of automatic, data-grounded insights (e.g.
   name any skewed numeric columns, columns with the most missing values,
   columns with potential outliers via the IQR method) — compute these
   values in code, do not write generic text.
2. Correlation — a seaborn heatmap of numeric column correlations (only if
   there are 2+ numeric columns), followed by 2-3 sentences via st.markdown
   naming the actual strongest positive and negative correlated pairs and
   their coefficients.
3. Univariate analysis — for every numeric column, show a histogram+KDE and
   a boxplot side by side using st.columns(2), with a st.caption() noting
   skew direction; for every categorical column with <= 30 unique values,
   show a horizontal count plot (bar) and a pie chart of its top 8
   categories side by side using st.columns(2), with a st.caption() noting
   the dominant category and its share.
4. Bivariate analysis — compute the pair of numeric columns with the
   strongest absolute correlation and show a scatter plot with a regression
   line, with the correlation coefficient in the chart title; if there is
   at least one categorical column with <= 10 unique values and one numeric
   column, also show a boxplot of that numeric column grouped by that
   category.
5. Multivariate analysis — a seaborn pairplot of up to 4 numeric columns
   (choose the ones with highest variance), optionally colored (hue) by a
   categorical column with <= 6 unique values if one exists; and a groupby
   bar chart grouping by the categorical column with the fewest unique
   values (between 2 and 10), aggregating (mean) the numeric column most
   correlated with the others, sorted descending, as a horizontal bar
   chart.
6. Time series analysis — ONLY if some column can plausibly be parsed as a
   date (test with pd.to_datetime on a sample, requiring >80% success
   rate): resample the most relevant numeric column to a sensible frequency
   and plot the trend with a rolling-mean overlay, with a st.markdown
   sentence describing the trend direction using real computed numbers. If
   no datetime-like column exists, just call
   st.info("No time-series-like column detected in this dataset.").
7. Key Takeaways — a final st.subheader("Key Takeaways") with 4-6
   st.markdown bullet points summarizing the most important, concrete
   findings from all sections above, referencing actual column names and
   numbers (never generic advice).

Formatting rules:
- Every matplotlib figure: create with `fig, ax = plt.subplots(figsize=(w,h))`,
  call `plt.tight_layout()`, display with `st.pyplot(fig)`, then
  `plt.close(fig)`. For the pairplot use `g = sns.pairplot(...)` then
  `st.pyplot(g.fig)` then `plt.close(g.fig)`.
- Use only the real column names given above.
- Ground every insight sentence in a value actually computed in the code
  (via an f-string), never a vague canned sentence.

Return ONLY the Python code for the `run_eda_dashboard(df)` function inside
a single ```python fenced code block — no explanation before or after it.
"""


def extract_code_block(text: str) -> str:
    """Pull code out of a ```python ... ``` (or bare ``` ... ```) fenced
    block returned by the LLM."""
    if "```" not in text:
        return text.strip()
    code = text.split("```")[1]
    lines = code.split("\n", 1)
    if lines[0].strip().lower() in ("python", "py"):
        code = lines[1] if len(lines) > 1 else ""
    return code.strip()


def is_code_safe(code: str):
    """Very lightweight denylist check on the AI-generated code before it
    is ever executed. Not a sandbox — just a guard against obviously
    disallowed operations the prompt already told the model to avoid."""
    lowered = code.lower()
    for token in _FORBIDDEN_TOKENS:
        if token.lower() in lowered:
            return False, token
    return True, ""


def generate_eda_dashboard_code(agent, df: pd.DataFrame) -> str:
    """Ask the agent to write the full dashboard function and return the
    extracted source code."""
    prompt = build_eda_codegen_prompt(df)
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    raw = extract_text(response)
    return extract_code_block(raw)


def save_generated_code(code: str) -> str:
    """Persist the generated dashboard code to disk, mirroring the
    file-based workflow of the original script (so it can also be
    downloaded / inspected / reused outside the app)."""
    with open(GEN_EDA_PATH, "w") as f:
        f.write(code)
    return GEN_EDA_PATH


def run_generated_eda_code(code: str, df: pd.DataFrame):
    """Execute the AI-generated code and call its run_eda_dashboard(df)
    function so it renders straight into the current Streamlit app."""
    ns = {"pd": pd, "np": np, "plt": plt, "sns": sns, "st": st, "df": df}
    exec(code, ns)
    if "run_eda_dashboard" not in ns or not callable(ns["run_eda_dashboard"]):
        raise RuntimeError("The generated code did not define a callable `run_eda_dashboard(df)` function.")
    ns["run_eda_dashboard"](df)


# ---------------------------------------------------------------------------
# Step 6: App UI
# ---------------------------------------------------------------------------
st.title("📊 AI-Powered Data Analyst")
st.caption("Upload a dataset and let an AI agent write and run a full EDA dashboard, or chat with your data.")

uploaded_file = st.file_uploader(
    "Upload a CSV, TSV, JSON, or Excel file",
    type=["csv", "tsv", "xlsx", "xls", "json"],
)

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    _df = load_dataset(uploaded_file)
    if _df is not None:
        st.session_state.df = _df
        st.session_state.filename = uploaded_file.name
        st.session_state.chat_history = []
        st.session_state.eda_code = None
        st.session_state.eda_code_error = None
        st.success(f"Loaded **{uploaded_file.name}** — {_df.shape[0]} rows × {_df.shape[1]} columns")

if st.session_state.df is not None:
    df = st.session_state.df

    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    tab_eda, tab_chat = st.tabs(["🧠 AI-Generated EDA Dashboard", "💬 Chat with your Data"])

    # ------------------------------------------------------------------ #
    # Tab 1: AI-generated advanced EDA dashboard
    # ------------------------------------------------------------------ #
    with tab_eda:
        st.write(
            "The AI writes a complete, dataset-specific analysis "
            "(describe, correlation, univariate, bivariate, multivariate, "
            "time series, and automatic insights) as real Python code, "
            "then runs it below."
        )
        col_gen, col_regen = st.columns([1, 1])
        generate_clicked = col_gen.button("🪄 Generate Advanced EDA Dashboard", type="primary")
        regenerate_clicked = col_regen.button("🔁 Regenerate")

        if generate_clicked or regenerate_clicked:
            try:
                agent = get_agent()
                with st.spinner("Asking the AI to write the analysis code..."):
                    code = generate_eda_dashboard_code(agent, df)
                safe, bad_token = is_code_safe(code)
                if not safe:
                    st.error(
                        f"The generated code contained a disallowed operation "
                        f"('{bad_token}'). Click Regenerate to try again."
                    )
                else:
                    save_generated_code(code)
                    st.session_state.eda_code = code
                    st.session_state.eda_code_error = None
            except Exception as e:
                st.error(f"Could not generate the dashboard: {e}")

        if st.session_state.eda_code:
            with st.expander("🧾 View / download generated code"):
                st.code(st.session_state.eda_code, language="python")
                st.download_button(
                    "Download generated_eda_dashboard.py",
                    st.session_state.eda_code,
                    file_name="generated_eda_dashboard.py",
                    mime="text/x-python",
                )

            st.markdown("---")
            try:
                run_generated_eda_code(st.session_state.eda_code, df)
                st.session_state.eda_code_error = None
            except Exception as e:
                st.session_state.eda_code_error = f"{e}\n{traceback.format_exc(limit=3)}"

            if st.session_state.eda_code_error:
                st.error("The generated code raised an error while running:")
                st.code(st.session_state.eda_code_error)
                st.info("Click **Regenerate** above to have the AI try again.")
        else:
            st.info("Click **Generate Advanced EDA Dashboard** to let the AI build it.")

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
                            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
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
