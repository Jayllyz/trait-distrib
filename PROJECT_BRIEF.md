# Project Brief — Traitement Distribué (ESGI 5AL2)

Summary of the two course documents (`Support de cours .pdf` and `Sujet.pdf`), plus how they map to **our** project: digit recognition on the Kaggle **Digit Recognizer** dataset with PySpark.

---

## 1. The assignment (Sujet)

Course: *Introduction aux Traitements Distribués* — 5ème année AL classe 2.

**Goal:** get hands-on with Spark and its data-processing APIs.

**Required deliverables:**

1. **Read & parse the data with PySpark**
2. **Statistical analysis of the dataset with PySpark**
3. **Data visualization** — free choice of technology, **Plotly and Matplotlib recommended**
4. *(Optional)* **Machine Learning models** in Python to make predictions

**Constraints:**

- Prerequisites: **Java 17**, **Python 3**
- Groups of **4–5 people**
- Topic is free but must be **validated by the teacher**

**Grading (from the course support):** 40% practical exercises, **60% final mini-project (code + oral presentation)**.

---

## 2. Course content summary (Support de cours)

Instructor: Yidhir Moudoub (yidhir.moudoub@outlook.com). 5 sessions × 3h.

### Session 1 — Distributed systems & Hadoop
- **Distributed system**: independent computers cooperating to appear as one coherent system. Distinguish *parallel processing* (multiple CPUs, one machine) from *distributed processing* (multiple machines over a network).
- Key properties: **scalability, fault tolerance, transparency, concurrency** (example: Google indexing billions of pages).
- **MapReduce** model (Google, 2004): Map = transform, Reduce = aggregate; execution distributed automatically.
- **Hadoop / HDFS**: distributed, replicated, fault-tolerant file system on cheap hardware; optimized for data/compute colocation.

### Session 2 — MapReduce limits & Apache Spark
- MapReduce limitations: slow (disk I/O between phases), complex to develop, no interactive/real-time use.
- **Why Spark**: in-memory processing (up to ~100x faster), simpler APIs (RDD, DataFrames, SQL), supports batch + streaming + ML.
- Architecture: **Driver** (entry point), **Executors/Workers**, **Cluster Manager** (e.g. YARN), **DAG Scheduler**.
- **RDD**: fundamental abstraction — distributed collection with *transformations* (map, filter…) and *actions* (collect, count, reduce…), fault-tolerant via the **DAG** lineage.

### Session 3 — Spark SQL & DataFrames
- DataFrames vs RDD: tabular structure with a **schema** (named columns), high-level API, better optimization.
- **Spark SQL**: run SQL queries over DataFrames, temp/permanent views, reads HDFS/Hive/JDBC.
- **Catalyst Optimizer**: logical/physical query planning, column resolution, type checking, in-memory management.
- Data formats: **CSV, JSON, Parquet, Avro**.
  - `spark.read.csv("f.csv", header=True, inferSchema=True)` / `df.write.parquet("out.parquet")`

### Session 4 — Spark Streaming
- Batch vs streaming; streaming = continuous data (sensors, logs, events) with latency/volume/reliability constraints.
- **Micro-batching**, DStream vs **Structured Streaming**, Kafka integration.
- *(Likely not needed for our project — our dataset is static CSV.)*

### Session 5 — Mini-project & oral presentation
- Group project with Spark + oral presentation (Q&A, evaluation, feedback).

---

## 3. Our project: Digit Recognizer

**Dataset** (already in `digit-recognizer/`, from Kaggle):
- `train.csv` — **42,000 rows × 785 columns**: `label` (0–9) + `pixel0`…`pixel783` (28×28 grayscale images, values 0–255)
- `test.csv` — same pixel columns, no label
- `sample_submission.csv` — Kaggle submission format

**Existing code** (`main.py`): creates a local SparkSession, loads `train.csv` with header + inferred schema, prints schema and first rows. Stack: Python ≥3.14, `pyspark ≥4.2`, `kagglehub`, managed with **uv** (dev tools: ruff, ty).

### Plan mapped to the required deliverables

| # | Requirement | What we do |
|---|-------------|------------|
| 1 | Read & parse with PySpark | Load train/test CSV into DataFrames, define/validate schema (label + 784 pixels), reshape rows into 28×28 images where needed. ✅ started in `main.py` |
| 2 | Statistical analysis with PySpark | Class distribution (count per digit 0–9), pixel statistics (mean/std/min/max, ink density per digit), average image per digit, dark-pixel ratios, Spark SQL queries on a temp view |
| 3 | Visualization | **Matplotlib** for charts: sample digits, mean image per class, class-distribution histogram, pixel-intensity heatmaps |
| 4 | ML (optional) | Train a classifier (Spark MLlib logistic regression / random forest, or scikit-learn) to predict digits; evaluate accuracy + confusion matrix |
| + | Interactive UI | Second library so users can "play" with the project: draw a digit and get a live prediction. Candidate libs: **Tkinter** (canvas drawing, stdlib) or **Streamlit**/**Gradio** (web UI) — to decide |

### Points to show off at the oral (ties back to the course)
- Spark architecture used: local driver, DataFrame API, Catalyst-optimized queries.
- Contrast DataFrame approach vs raw RDD/MapReduce (why it's simpler and faster).
- Use Spark SQL for at least part of the analysis (temp view + SQL queries).
- Optionally write intermediate results to **Parquet** to demonstrate format knowledge.
