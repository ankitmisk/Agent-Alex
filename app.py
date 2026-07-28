# """
# AI-Powered Data Analyst — Streamlit App
# ----------------------------------------
# Upload a dataset (CSV / TSV / JSON / Excel) and let an LLM-backed agent
# run automated EDA, answer natural-language questions about the data, and
# generate charts on demand.

# Run with:  streamlit run app.py
# """

# # ---------------------------------------------------------------------------
# # Step 3: Load all modules
# # ---------------------------------------------------------------------------
# import os
# import io
# import contextlib
# import traceback

# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")  # headless backend, required for Streamlit
# import matplotlib.pyplot as plt
# import seaborn as sns

# import streamlit as st

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
# from langchain.agents import create_agent
# from langchain_core.tools import tool


# # ---------------------------------------------------------------------------
# # Page config
# # ---------------------------------------------------------------------------
# st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

# SYSTEM_PROMPT = """You are an expert data analyst assistant embedded in a
# Streamlit app. A pandas DataFrame has already been loaded for the user.

# You have three tools:
# - get_dataframe_overview: inspect shape, dtypes, missing values, sample rows.
# - run_pandas_code: execute pandas/numpy code to compute an answer. Always
#   assign your final answer to a variable called `result`.
# - create_visualization: execute matplotlib/seaborn code to draw a chart.

# Guidelines:
# - Always call get_dataframe_overview first if you are unsure about column
#   names or types before writing analysis code.
# - Prefer using the tools over guessing; never fabricate numbers.
# - Keep code simple, use only pandas, numpy, matplotlib, seaborn (already
#   imported in the execution environment as pd, np, plt, sns) and the
#   dataframe `df`. Do not import other packages, read/write files, or use
#   the network.
# - If the user asks for a chart, plot, graph, visualization, or to "show" /
#   "plot" / "visualize" something, you MUST call the create_visualization
#   tool to actually render it. Never just describe what a chart would look
#   like in text instead of calling the tool.
# - In create_visualization code, always create the figure explicitly first,
#   e.g. `plt.figure(figsize=(8,5))`, then call the plotting function
#   (plt.*, sns.*, or df.plot), and call `plt.tight_layout()` at the end.
# - After using tools, answer the user in clear, concise natural language.
#   Summarize numeric results instead of just dumping raw output.
# """


# # ---------------------------------------------------------------------------
# # Session state initialisation
# # ---------------------------------------------------------------------------
# _defaults = {
#     "df": None,
#     "filename": None,
#     "chat_history": [],     # list of {"role", "content", "figure"}
#     "last_figure": None,    # scratch slot tools write into
#     "eda_report": None,     # cached AI narrative from the last EDA run
# }
# for _k, _v in _defaults.items():
#     if _k not in st.session_state:
#         st.session_state[_k] = _v


# # ---------------------------------------------------------------------------
# # Sidebar: configuration
# # ---------------------------------------------------------------------------
# st.sidebar.title("⚙️ Configuration")

# provider = st.sidebar.selectbox("LLM Provider", ["Google Gemini", "Groq"])

# google_api_key = st.sidebar.text_input(
#     "Google API Key",
#     type="password",
#     value=os.environ.get("GOOGLE_API_KEY", ""),
#     help="Required if using Gemini. Get one at https://aistudio.google.com/apikey",
# )
# groq_api_key = st.sidebar.text_input(
#     "Groq API Key",
#     type="password",
#     value=os.environ.get("GROQ_API_KEY", ""),
#     help="Required if using Groq. Get one at https://console.groq.com/keys",
# )

# gemini_model_name = st.sidebar.text_input("Gemini model", value="gemini-2.0-flash")
# groq_model_name = st.sidebar.text_input("Groq model", value="llama-3.3-70b-versatile")

# st.sidebar.markdown("---")
# if st.sidebar.button("🗑️ Reset conversation"):
#     st.session_state.chat_history = []
#     st.session_state.last_figure = None
#     st.rerun()

# st.sidebar.caption("API keys are used only for this session and are never stored or logged.")


# # ---------------------------------------------------------------------------
# # Step 4: Model creation
# # ---------------------------------------------------------------------------
# @st.cache_resource(show_spinner=False)
# def build_llm(provider: str, google_key: str, google_model: str, groq_key: str, groq_model: str):
#     """Create and cache the chat model for the selected provider."""
#     if provider == "Google Gemini":
#         if not google_key:
#             raise ValueError("Please provide a Google API key in the sidebar.")
#         return ChatGoogleGenerativeAI(model=google_model, google_api_key=google_key, temperature=0)
#     else:
#         if not groq_key:
#             raise ValueError("Please provide a Groq API key in the sidebar.")
#         return ChatGroq(model=groq_model, api_key=groq_key, temperature=0)


# # ---------------------------------------------------------------------------
# # Step 5: Agent creation — tools + agent
# # ---------------------------------------------------------------------------
# def _exec_namespace():
#     return {"pd": pd, "np": np, "plt": plt, "sns": sns, "df": st.session_state.df}


# @tool
# def get_dataframe_overview() -> str:
#     """Return a text overview of the uploaded dataframe: shape, dtypes,
#     missing value counts, sample rows, and summary statistics. Use this
#     tool first to understand the dataset before writing analysis code."""
#     df = st.session_state.df
#     if df is None:
#         return "No dataframe has been loaded yet."
#     buf = io.StringIO()
#     df.info(buf=buf)
#     missing = df.isnull().sum()
#     missing_str = missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values."
#     return (
#         f"Shape: {df.shape}\n\n"
#         f"Info:\n{buf.getvalue()}\n\n"
#         f"Missing values:\n{missing_str}\n\n"
#         f"Sample rows:\n{df.head(5).to_string()}\n\n"
#         f"Summary statistics:\n{df.describe(include='all').to_string()}"
#     )


# @tool
# def run_pandas_code(code: str) -> str:
#     """Execute python/pandas code against the uploaded dataframe (available
#     as `df`) to compute an analytical answer. pandas is `pd`, numpy is `np`.
#     Assign your final answer to a variable named `result`. Do not read or
#     write files, and do not import extra packages. Use this for computation
#     only, not for plots (use create_visualization for plots)."""
#     if st.session_state.df is None:
#         return "No dataframe has been loaded yet."
#     ns = _exec_namespace()
#     stdout_capture = io.StringIO()
#     try:
#         with contextlib.redirect_stdout(stdout_capture):
#             exec(code, ns)
#         result = ns.get("result", None)
#         output = stdout_capture.getvalue()
#         if result is not None:
#             return f"Result:\n{result}" + (f"\n\nPrinted output:\n{output}" if output else "")
#         elif output:
#             return f"Printed output:\n{output}"
#         return "Code executed but produced no output. Assign the answer to a variable named `result`."
#     except Exception as e:
#         return f"Error executing code: {e}\n{traceback.format_exc(limit=2)}"


# @tool
# def create_visualization(code: str) -> str:
#     """Execute matplotlib/seaborn code to create a chart from the dataframe
#     (available as `df`). pandas is `pd`, numpy is `np`, matplotlib.pyplot is
#     `plt`, seaborn is `sns`. Build the plot with plt/sns calls; do not call
#     plt.show(). The resulting figure is automatically captured and shown
#     to the user."""
#     if st.session_state.df is None:
#         return "No dataframe has been loaded yet."
#     ns = _exec_namespace()
#     try:
#         plt.close("all")
#         exec(code, ns)
#         fig = plt.gcf()
#         if not fig.get_axes():
#             return "No plot was created. Make sure to call a plotting function like plt.plot(...) or sns.barplot(...)."
#         st.session_state.last_figure = fig
#         return "Visualization created successfully and will be displayed to the user."
#     except Exception as e:
#         return f"Error creating visualization: {e}\n{traceback.format_exc(limit=2)}"


# def build_agent(llm):
#     """Assemble the tool-calling agent. Lightweight, so rebuilt each call."""
#     tools = [get_dataframe_overview, run_pandas_code, create_visualization]
#     return create_agent(model=llm, tools=tools)


# def extract_text(response) -> str:
#     """Robustly pull the assistant's text out of a create_agent response,
#     regardless of whether content is a plain string or a list of content
#     blocks (Gemini/Groq return slightly different shapes)."""
#     try:
#         msg = response["messages"][-1]
#         content = getattr(msg, "content", None)
#         if content is None and isinstance(msg, dict):
#             content = msg.get("content")
#         if isinstance(content, str):
#             return content.strip()
#         if isinstance(content, list):
#             parts = []
#             for block in content:
#                 if isinstance(block, dict) and block.get("type") == "text":
#                     parts.append(block.get("text", ""))
#                 elif isinstance(block, str):
#                     parts.append(block)
#             return "\n".join(p for p in parts if p).strip()
#         return str(content)
#     except Exception:
#         return "Sorry, I couldn't parse the agent's response."


# def get_agent():
#     """Build (or fail loudly on) the LLM + agent using current sidebar settings."""
#     llm = build_llm(provider, google_api_key, gemini_model_name, groq_api_key, groq_model_name)
#     return build_agent(llm)


# # ---------------------------------------------------------------------------
# # Data loading
# # ---------------------------------------------------------------------------
# def load_dataset(uploaded_file):
#     """Load an uploaded Streamlit file (CSV / TSV / JSON / XLSX / XLS)
#     into a pandas DataFrame based on its extension."""
#     name = uploaded_file.name
#     ext = name.split(".")[-1].lower()
#     try:
#         if ext == "csv":
#             return pd.read_csv(uploaded_file)
#         elif ext == "tsv":
#             return pd.read_csv(uploaded_file, sep="\t")
#         elif ext in ("xlsx", "xls"):
#             return pd.read_excel(uploaded_file)
#         elif ext == "json":
#             return pd.read_json(uploaded_file)
#         else:
#             st.error(f"Unsupported file type: .{ext}")
#             return None
#     except Exception as e:
#         st.error(f"Failed to read file: {e}")
#         return None


# # ---------------------------------------------------------------------------
# # EDA
# # ---------------------------------------------------------------------------
# def perform_eda(df: pd.DataFrame) -> dict:
#     """Compute a structured EDA summary for the given dataframe."""
#     numeric_df = df.select_dtypes(include=np.number)
#     cat_df = df.select_dtypes(include=["object", "category"])
#     return {
#         "shape": df.shape,
#         "dtypes": df.dtypes.astype(str).to_dict(),
#         "missing_values": df.isnull().sum().to_dict(),
#         "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
#         "duplicates": int(df.duplicated().sum()),
#         "numeric_summary": numeric_df.describe().T if not numeric_df.empty else None,
#         "categorical_summary": (
#             {col: df[col].value_counts().head(5).to_dict() for col in cat_df.columns}
#             if not cat_df.empty else {}
#         ),
#     }


# def generate_ai_insights(agent, df: pd.DataFrame, report: dict) -> str:
#     """Ask the agent to write a short natural-language insights report
#     based on the computed EDA summary."""
#     sample = df.sample(min(5, len(df))).to_string()
#     prompt = f"""{SYSTEM_PROMPT}

# Based on the dataset sample and EDA summary below, write a concise report
# (short paragraphs + bullet points) covering: what the dataset appears to
# represent, data quality issues (missing values, duplicates), notable
# patterns, and 3-5 suggested next analysis steps. Do not write code, only
# the written report.

# Sample rows:
# {sample}

# Shape: {report['shape']}
# Missing values (top 10): {dict(list(report['missing_values'].items())[:10])}
# Duplicate rows: {report['duplicates']}
# """
#     response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
#     return extract_text(response)


# # ---------------------------------------------------------------------------
# # Dashboard helpers (deterministic — no LLM involved, always renders)
# # ---------------------------------------------------------------------------
# def detect_column_types(df: pd.DataFrame):
#     """Split columns into numeric / categorical / datetime, including
#     object columns that look like dates on inspection of a sample."""
#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
#     categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
#     datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

#     for col in list(categorical_cols):
#         sample = df[col].dropna().head(50)
#         if sample.empty:
#             continue
#         parsed = pd.to_datetime(sample, errors="coerce")
#         if parsed.notna().mean() > 0.8:
#             datetime_cols.append(col)
#             categorical_cols.remove(col)
#     return numeric_cols, categorical_cols, datetime_cols


# def render_overview_section(df, numeric_cols):
#     st.subheader("Describe")
#     st.dataframe(df.describe(include="all").T, use_container_width=True)

#     st.subheader("Correlation Matrix")
#     if len(numeric_cols) >= 2:
#         corr = df[numeric_cols].corr(numeric_only=True)
#         size = min(1 + len(numeric_cols) * 0.8, 12)
#         fig, ax = plt.subplots(figsize=(size, size * 0.85))
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
#         plt.tight_layout()
#         st.pyplot(fig)
#     else:
#         st.info("Need at least 2 numeric columns to compute a correlation matrix.")


# def render_univariate_section(df, numeric_cols, categorical_cols):
#     col = st.selectbox("Select a column", df.columns.tolist(), key="univ_col")
#     c1, c2 = st.columns(2)

#     if col in numeric_cols:
#         with c1:
#             fig, ax = plt.subplots(figsize=(6, 4))
#             sns.histplot(df[col].dropna(), kde=True, ax=ax)
#             ax.set_title(f"Distribution of {col}")
#             plt.tight_layout()
#             st.pyplot(fig)
#         with c2:
#             fig, ax = plt.subplots(figsize=(6, 4))
#             sns.boxplot(x=df[col].dropna(), ax=ax)
#             ax.set_title(f"Boxplot of {col}")
#             plt.tight_layout()
#             st.pyplot(fig)
#         st.write(df[col].describe())
#     else:
#         top_n = st.slider("Top N categories", 3, 20, 10, key="univ_topn")
#         vc = df[col].value_counts().head(top_n)
#         with c1:
#             fig, ax = plt.subplots(figsize=(6, 4))
#             sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, orient="h")
#             ax.set_title(f"Count plot of {col}")
#             ax.set_xlabel("Count")
#             plt.tight_layout()
#             st.pyplot(fig)
#         with c2:
#             fig, ax = plt.subplots(figsize=(6, 4))
#             ax.pie(vc.values, labels=vc.index.astype(str), autopct="%1.1f%%", startangle=90)
#             ax.set_title(f"Share of {col}")
#             plt.tight_layout()
#             st.pyplot(fig)
#         st.write(vc)


# def render_bivariate_section(df, numeric_cols, categorical_cols):
#     cols = df.columns.tolist()
#     c1, c2 = st.columns(2)
#     x_col = c1.selectbox("X variable", cols, key="biv_x")
#     y_col = c2.selectbox("Y variable", cols, index=min(1, len(cols) - 1), key="biv_y")

#     if x_col in numeric_cols and y_col in numeric_cols:
#         fig, ax = plt.subplots(figsize=(7, 5))
#         sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter_kws={"alpha": 0.5})
#         corr_val = df[[x_col, y_col]].corr().iloc[0, 1]
#         ax.set_title(f"{x_col} vs {y_col}  (corr = {corr_val:.2f})")
#         plt.tight_layout()
#         st.pyplot(fig)
#     elif (x_col in categorical_cols and y_col in numeric_cols) or (y_col in categorical_cols and x_col in numeric_cols):
#         cat_col, num_col = (x_col, y_col) if x_col in categorical_cols else (y_col, x_col)
#         top_cats = df[cat_col].value_counts().head(10).index
#         plot_df = df[df[cat_col].isin(top_cats)]
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.boxplot(data=plot_df, x=cat_col, y=num_col, ax=ax)
#         ax.set_title(f"{num_col} by {cat_col}")
#         plt.xticks(rotation=45, ha="right")
#         plt.tight_layout()
#         st.pyplot(fig)
#     elif x_col in categorical_cols and y_col in categorical_cols:
#         ct = pd.crosstab(df[x_col], df[y_col])
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
#         ax.set_title(f"{x_col} vs {y_col} (counts)")
#         plt.tight_layout()
#         st.pyplot(fig)
#     else:
#         st.info("Pick two different columns to compare.")


# def render_multivariate_section(df, numeric_cols, categorical_cols):
#     st.subheader("Pairplot")
#     default_sel = numeric_cols[: min(4, len(numeric_cols))]
#     selected = st.multiselect("Numeric columns (2–5 recommended)", numeric_cols, default=default_sel, key="mv_cols")
#     hue_col = st.selectbox("Optional hue (categorical)", ["None"] + categorical_cols, key="mv_hue")
#     if len(selected) >= 2:
#         if st.button("Generate Pairplot"):
#             with st.spinner("Generating pairplot..."):
#                 hue = None if hue_col == "None" else hue_col
#                 plot_cols = selected + ([hue] if hue else [])
#                 g = sns.pairplot(df[plot_cols].dropna(), hue=hue)
#                 st.pyplot(g.fig)
#     else:
#         st.info("Select at least 2 numeric columns.")

#     st.markdown("---")
#     st.subheader("Groupby Aggregation")
#     c1, c2, c3 = st.columns(3)
#     group_options = categorical_cols if categorical_cols else df.columns.tolist()
#     agg_options = numeric_cols if numeric_cols else df.columns.tolist()
#     group_col = c1.selectbox("Group by", group_options, key="gb_group")
#     agg_col = c2.selectbox("Aggregate column", agg_options, key="gb_agg")
#     agg_func = c3.selectbox("Aggregation", ["mean", "sum", "count", "median", "min", "max"], key="gb_func")
#     if group_col and agg_col:
#         grouped = df.groupby(group_col)[agg_col].agg(agg_func).sort_values(ascending=False).head(15)
#         fig, ax = plt.subplots(figsize=(8, 5))
#         sns.barplot(x=grouped.values, y=grouped.index.astype(str), ax=ax)
#         ax.set_title(f"{agg_func} of {agg_col} by {group_col}")
#         plt.tight_layout()
#         st.pyplot(fig)
#         st.dataframe(grouped)


# def render_timeseries_section(df, numeric_cols, datetime_cols):
#     if not datetime_cols:
#         st.info("No date/time-like column was detected in this dataset.")
#         return
#     if not numeric_cols:
#         st.info("No numeric column available to plot over time.")
#         return

#     c1, c2, c3 = st.columns(3)
#     date_col = c1.selectbox("Date column", datetime_cols, key="ts_date")
#     value_col = c2.selectbox("Value column", numeric_cols, key="ts_value")
#     freq = c3.selectbox("Resample frequency", ["D", "W", "M", "Q", "Y"], index=2, key="ts_freq")

#     ts_df = df[[date_col, value_col]].copy()
#     ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
#     ts_df = ts_df.dropna(subset=[date_col]).set_index(date_col).sort_index()

#     if ts_df.empty:
#         st.warning("Could not parse any valid dates from that column.")
#         return

#     resampled = ts_df[value_col].resample(freq).sum()
#     rolling = resampled.rolling(window=3, min_periods=1).mean()

#     fig, ax = plt.subplots(figsize=(10, 5))
#     resampled.plot(ax=ax, marker="o", label=value_col)
#     rolling.plot(ax=ax, linestyle="--", label="Rolling mean (3)")
#     ax.set_title(f"{value_col} over time ({freq})")
#     ax.legend()
#     plt.tight_layout()
#     st.pyplot(fig)


# # ---------------------------------------------------------------------------
# # Step 6: App UI
# # ---------------------------------------------------------------------------
# st.title("📊 AI-Powered Data Analyst")
# st.caption("Upload a dataset and let an AI agent explore, analyze, and visualize it for you.")

# uploaded_file = st.file_uploader(
#     "Upload a CSV, TSV, JSON, or Excel file",
#     type=["csv", "tsv", "xlsx", "xls", "json"],
# )

# if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
#     _df = load_dataset(uploaded_file)
#     if _df is not None:
#         st.session_state.df = _df
#         st.session_state.filename = uploaded_file.name
#         st.session_state.eda_report = None
#         st.session_state.chat_history = []
#         st.success(f"Loaded **{uploaded_file.name}** — {_df.shape[0]} rows × {_df.shape[1]} columns")

# if st.session_state.df is not None:
#     df = st.session_state.df

#     with st.expander("🔍 Data Preview", expanded=True):
#         st.dataframe(df.head(20), use_container_width=True)

#     tab_eda, tab_dash, tab_chat = st.tabs(
#         ["📋 Automated EDA", "📈 Visual Dashboard", "💬 Chat with your Data"]
#     )

#     # ------------------------------------------------------------------ #
#     # Tab 1: Automated EDA
#     # ------------------------------------------------------------------ #
#     with tab_eda:
#         run_eda = st.button("Run Automated EDA", type="primary")

#         if run_eda:
#             with st.spinner("Analyzing dataset..."):
#                 report = perform_eda(df)

#                 col1, col2, col3 = st.columns(3)
#                 col1.metric("Rows", report["shape"][0])
#                 col2.metric("Columns", report["shape"][1])
#                 col3.metric("Duplicate rows", report["duplicates"])

#                 st.subheader("Column Types & Missing Values")
#                 st.dataframe(
#                     pd.DataFrame({
#                         "dtype": report["dtypes"],
#                         "missing_values": report["missing_values"],
#                         "missing_%": report["missing_pct"],
#                     }),
#                     use_container_width=True,
#                 )

#                 if report["numeric_summary"] is not None:
#                     st.subheader("Numeric Summary")
#                     st.dataframe(report["numeric_summary"], use_container_width=True)

#                 if report["categorical_summary"]:
#                     st.subheader("Top Categorical Values")
#                     for col, vals in report["categorical_summary"].items():
#                         st.write(f"**{col}**")
#                         st.write(vals)

#                 st.subheader("🧠 AI Insights")
#                 try:
#                     agent = get_agent()
#                     with st.spinner("Generating AI insights..."):
#                         insights = generate_ai_insights(agent, df, report)
#                     st.session_state.eda_report = insights
#                     st.markdown(insights)
#                 except Exception as e:
#                     st.warning(f"Could not generate AI insights: {e}")
#         elif st.session_state.eda_report:
#             st.markdown(st.session_state.eda_report)
#         else:
#             st.info("Click **Run Automated EDA** to generate a full report.")

#     # ------------------------------------------------------------------ #
#     # Tab 2: Visual Dashboard (deterministic — always renders, no LLM)
#     # ------------------------------------------------------------------ #
#     with tab_dash:
#         numeric_cols, categorical_cols, datetime_cols = detect_column_types(df)

#         section = st.radio(
#             "Choose analysis type",
#             [
#                 "Overview (Describe & Correlation)",
#                 "Univariate",
#                 "Bivariate",
#                 "Multivariate & Groupby",
#                 "Time Series",
#             ],
#             horizontal=True,
#         )
#         st.markdown("---")

#         if section == "Overview (Describe & Correlation)":
#             render_overview_section(df, numeric_cols)
#         elif section == "Univariate":
#             render_univariate_section(df, numeric_cols, categorical_cols)
#         elif section == "Bivariate":
#             render_bivariate_section(df, numeric_cols, categorical_cols)
#         elif section == "Multivariate & Groupby":
#             render_multivariate_section(df, numeric_cols, categorical_cols)
#         elif section == "Time Series":
#             render_timeseries_section(df, numeric_cols, datetime_cols)

#     # ------------------------------------------------------------------ #
#     # Tab 3: Chat
#     # ------------------------------------------------------------------ #
#     with tab_chat:
#         for msg in st.session_state.chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])
#                 if msg.get("figure") is not None:
#                     st.pyplot(msg["figure"])

#         user_query = st.chat_input(
#             "Ask a question about your data (e.g. 'What are the top 5 products by sales?')"
#         )

#         if user_query:
#             st.session_state.chat_history.append({"role": "user", "content": user_query, "figure": None})
#             with st.chat_message("user"):
#                 st.markdown(user_query)

#             try:
#                 agent = get_agent()
#                 st.session_state.last_figure = None
#                 with st.spinner("Thinking..."):
#                     response = agent.invoke({
#                         "messages": [
#                             {"role": "system", "content": SYSTEM_PROMPT},
#                             {"role": "user", "content": user_query},
#                         ]
#                     })
#                 answer = extract_text(response)
#                 fig = st.session_state.last_figure
#                 with st.chat_message("assistant"):
#                     st.markdown(answer)
#                     if fig is not None:
#                         st.pyplot(fig)
#                 st.session_state.chat_history.append({"role": "assistant", "content": answer, "figure": fig})
#             except Exception as e:
#                 error_msg = f"⚠️ Something went wrong: {e}"
#                 with st.chat_message("assistant"):
#                     st.markdown(error_msg)
#                 st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "figure": None})
# else:
#     st.info("👆 Upload a dataset to get started.")
###############################################################
# AI Data Analyst Agent
# Streamlit Application
###############################################################

import os
import importlib
import pandas as pd
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

###############################################################
# Page Config
###############################################################

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

st.title("🤖 AI Data Analyst Agent")
st.write("Upload any CSV or Excel file and let AI generate code for reading and performing EDA.")

###############################################################
# API KEY
###############################################################

GOOGLE_API_KEY = "YOUR_API_KEY"

###############################################################
# LLM
###############################################################

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GOOGLE_API_KEY
)

###############################################################
# Dummy Tool
###############################################################

def temp_tool():
    """Dummy Tool"""
    return "Hello"

###############################################################
# Agent
###############################################################

agent = create_agent(
    model=llm,
    tools=[temp_tool]
)

###############################################################
# Function : Generate File Loader
###############################################################

def load_dataset_agent(uploaded_file):

    prompt = f"""
You are an expert python developer.

Generate ONLY executable python code.

Create a function

def read_uploaded_file(file):

Requirements:

1. Detect extension automatically.
2. Read CSV using pandas.read_csv().
3. Read XLS/XLSX using pandas.read_excel().
4. Return dataframe.
5. No explanation.
6. No markdown.
"""

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    text = response["messages"][-1].content

    if isinstance(text, list):
        text = text[-1]["text"]

    if "```" in text:
        code = text.split("```")[1].replace("python", "")
    else:
        code = text

    with open("file_loader.py", "w", encoding="utf8") as f:
        f.write(code)

    return True

###############################################################
# Read Dataset
###############################################################

def read_file(uploaded_file):

    import file_loader
    importlib.reload(file_loader)

    return file_loader.read_uploaded_file(uploaded_file)

###############################################################
# Generate EDA Module
###############################################################

def generate_eda_agent(df):

    sample = df.head().to_string()

    prompt = f"""
You are a Senior Data Analyst.

Generate ONLY executable python code.

Create one function

def perform_eda(df):

Function should return dictionary.

Include

Shape

Rows

Columns

Missing Values

Duplicate Rows

Data Types

Numerical Summary

Categorical Summary

No markdown.

No explanation.

Sample Data

{sample}
"""

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    text = response["messages"][-1].content

    if isinstance(text, list):
        text = text[-1]["text"]

    if "```" in text:
        code = text.split("```")[1].replace("python", "")
    else:
        code = text

    with open("basic_eda.py", "w", encoding="utf8") as f:
        f.write(code)

    return True

###############################################################
# Execute EDA
###############################################################

def run_eda(df):

    import basic_eda
    importlib.reload(basic_eda)

    return basic_eda.perform_eda(df)

###############################################################
# Sidebar
###############################################################

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Choose Dataset",
    type=["csv", "xlsx", "xls"]
)

###############################################################
# Main Workflow
###############################################################

if uploaded_file is not None:

    st.success("Dataset Uploaded Successfully")

    if st.button("🚀 Run AI Agent"):

        with st.spinner("Generating File Loader..."):

            load_dataset_agent(uploaded_file)

        df = read_file(uploaded_file)

        st.success("Dataset Loaded")

        st.subheader("Preview")

        st.dataframe(df.head())

        with st.spinner("Generating EDA Agent..."):

            generate_eda_agent(df)

        result = run_eda(df)

        st.success("EDA Completed")

        st.header("EDA Report")

        st.write("### Shape")
        st.write(result["Shape"])

        st.write("### Missing Values")
        st.dataframe(result["Missing Values"])

        st.write("### Duplicate Rows")
        st.write(result["Duplicate Rows"])

        st.write("### Data Types")
        st.dataframe(result["Data Types"])

        st.write("### Numerical Summary")
        st.dataframe(result["Numerical Summary"])

        st.write("### Categorical Summary")
        st.dataframe(result["Categorical Summary"])
