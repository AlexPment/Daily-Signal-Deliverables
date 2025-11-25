# 🧠 Mini-ForgeDAN Lab

**Educational AI Jailbreaking & Safety Exploration**

> ⚠️ **DISCLAIMER & CREDITS**
> This project is an **educational red-teaming lab**, inspired by the paper:
> **“FORGEDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models” – Siyang Cheng et al., 2025 (arXiv: 2511.13548v1).**
>
> All code here is intended **only for learning, research, and safety evaluation of your own local test models.**
>
> * **Do not use this code to attack, bypass, or interfere with any production AI systems, services, or third-party infrastructure.**
> * **Do not use it to generate or deploy real malware, ransomware, or harmful content.**
>
> By using this repository you agree that you are responsible for complying with all applicable laws, terms of service, and ethical guidelines. The author of this repo is not responsible for misuse.

---

## 🔍 Overview

This repo reproduces, in a simplified form, the ideas from ForgeDAN:

1. **v1: `simple_forgedan.py`** — a toy “jailbreak detector” using naive keyword rules.
2. **v2: `real_forgedan.py`** — a mini ForgeDAN clone:

   * multi-strategy prompt mutations
   * semantic fitness via embeddings
   * dual behavior/content judges
3. **v3: `AI_real_forgedan.py`** — an **AI-vs-AI** setup:

   * one LLM is the **target** (safety-aligned)
   * another LLM is the **generator** (attacker prompt-engineer)

Supporting scripts help you:

* test your environment (`test_ollama_llm.py`, `test_embeddings.py`)
* train the judges (`train_judges.py` + `judge_training.csv`)
* store judge models (`behavior_judge.joblib`, `content_judge.joblib`)

The recommended way to use this repo is to follow the **same journey**:

1. Start with v1 → see why naive detectors lie.
2. Move to v2 → see real dual-channel jailbreaks.
3. Move to v3 → see how one AI can learn to jailbreak another.

---

## ⚙️ Prerequisites

* Python 3.10+ (matching your `pyvenv.cfg`)
* A virtual environment (already created in your project; see `pyvenv.cfg`)
* [Ollama](https://ollama.ai) installed and running locally
* A local LLM available in Ollama (e.g. `gemma:2b-instruct`)

Install Python dependencies in your venv:

```bash
pip install sentence-transformers scikit-learn rich
```

Pull the model in Ollama (adjust name if needed):

```bash
ollama pull gemma:2b-instruct
```

---

## 🧪 Step 0 – Sanity Checks

### 0.1 `test_ollama_llm.py`

**What it does:**
Simple script to check that:

* Ollama is reachable
* The target model (e.g. `gemma:2b-instruct`) loads and responds

**How to run:**

```bash
python test_ollama_llm.py
```

You should see a short, coherent answer from the model. If this fails, fix Ollama/model before doing anything else.

---

### 0.2 `test_embeddings.py`

**What it does:**

* Loads the sentence-transformer model (MiniLM)
* Encodes a couple of strings
* Prints a cosine similarity score

**How to run:**

```bash
python test_embeddings.py
```

If this works, embeddings are configured correctly.

---

## 🧸 Step 1 – v1: Toy Script (`simple_forgedan.py`)

**Goal:**
See why a naive keyword-based “detector” is useless for real safety.

**What it does:**

* Sends a few prompts like *“Is this ransomware?”* to the LLM
* Checks the output with a **single `if "ransomware" in output.lower()`** line
* Prints `SUCCESS` in green if the word appears, otherwise `FAIL`

This will happily mark **refusals** as “SUCCESS” just because the model mentioned the word.

**How to run:**

```bash
python simple_forgedan.py
```

**What to look for:**

* Lots of green `SUCCESS` lines, even when the model is clearly *refusing* to help.
* This illustrates the core lesson: **string-matching ≠ safety.**

---

## 🧬 Step 2 – v2: Mini-ForgeDAN (`real_forgedan.py`)

**Goal:**
Build a small ForgeDAN-style engine with:

* multi-strategy mutations
* semantic fitness
* dual judges (behavior + content)
* evolutionary search over prompts

### 2.1 `judge_training.csv`

**What it is:**
A CSV containing training examples for the judges.
Columns:

* `text` – model output
* `behavior_label` – `"refuse"` or `"comply"`
* `content_label` – `"safe"` or `"harmful"`

You can inspect and extend this file with more examples (e.g. dual-channel leaks you discover during experiments).

---

### 2.2 `train_judges.py`

**What it does:**

* Reads `judge_training.csv`
* Encodes all `text` using MiniLM
* Trains two LogisticRegression models:

  * behavior judge
  * content judge
* Saves them as:

  * `behavior_judge.joblib`
  * `content_judge.joblib`

**How to run:**

```bash
python train_judges.py
```

You should see training + classification reports in the terminal.
If all goes well, you’ll find the two `.joblib` files in the project directory.

---

### 2.3 `real_forgedan.py`

**What it does:**

* Implements **Version 2 (v2)**, the mini-ForgeDAN:

  1. **Mutations:**

     * character, word, and sentence-level mutators
  2. **Semantic fitness:**

     * loads a list of harmful reference descriptions
     * computes cosine similarity between each model answer and those references
  3. **Judges:**

     * loads `behavior_judge.joblib` & `content_judge.joblib`
     * classifies each output as `refuse/comply` and `safe/harmful`
  4. **Evolution loop:**

     * builds a population of mutated prompts
     * queries the target LLM
     * scores, judges, selects elites, mutates again
     * stops if it finds `comply + harmful` (“jailbreak”) or reaches max generations

**How to run:**

```bash
python real_forgedan.py
```

**What to look for:**

* Generations printed with a table of:

  * prefix (truncated)
  * fitness
  * behavior label
  * content label
* Cases where the model **refuses** on the surface and still leaks harmful structure underneath
* Instances where judges may mislabel harmful outputs as “safe” (a key safety insight)

---

## 🤖 Step 3 – v3: AI vs AI (`AI_real_forgedan.py`)

**Goal:**
Explore an **adaptive attacker**: use a second LLM to automatically refine jailbreak prompts based on feedback.

**What it does:**

* Adds a **Generator LLM** which:

  * reads:

    * the unsafe goal
    * the current prompt
    * the target’s last answer
    * judge labels (behavior/content)
    * semantic fitness score
  * produces a **new, “better” jailbreak prompt**
* Keeps the rest of the pipeline:

  * target LLM
  * judges
  * semantic fitness
* Runs a loop:

  1. Send prompt to target
  2. Judge + score the answer
  3. If `comply + harmful` → mark jailbreak success
  4. Otherwise, ask the generator to refine the prompt and repeat

**How to run:**

```bash
python AI_real_forgedan.py
```

**What to look for:**

* The generator’s prompts becoming:

  * more narrative (hacker stories, incident-response cases)
  * more emotionally rich
  * less obviously malicious
* The target model gradually:

  * refusing less
  * leaking more
  * and, in some runs, almost never refusing

At the end of each run, the script prints a **recap table**:

* total iterations
* number of `refuse/comply`
* number of `safe/harmful`
* whether any success was detected
* if success: the **winning prompt** and its answer

This is your “AI-vs-AI” lab: it shows how easily one model can learn to push another beyond its alignment limits.

---

## 📁 Other Files

### `pyvenv.cfg`

* Standard venv configuration file (created by Python’s `venv` module).
* You don’t need to edit this; it’s just part of the virtual environment.

---

## 🧵 Suggested Journey for New Users

1. **Read** the ForgeDAN paper (or at least the abstract + core design sections).
2. **Set up** your environment:

   * install Python deps,
   * install & test Ollama,
   * run `test_ollama_llm.py` and `test_embeddings.py`.
3. **Play with v1:** `python simple_forgedan.py`

   * See why keyword detection is misleading.
4. **Train the judges:** `python train_judges.py`
5. **Run v2:** `python real_forgedan.py`

   * Observe dual-channel leaks and judge failures.
6. **Run v3:** `python AI_real_forgedan.py`

   * Watch how the generator LLM evolves jailbreak prompts and how the target model’s behavior changes.
7. **Optionally extend:**

   * Add more entries to `judge_training.csv` (e.g. misclassified outputs).
   * Retrain the judges and see if detection improves.
   * Swap in different local LLMs (but always respect their usage policies).

---

## 🧷 Final Reminder

This repo is a **controlled laboratory**, not a weapon.

Use it to:

* understand how jailbreaks work,
* test your own LLM safety mechanics,
* and build better defensive strategies.

Do **not** point this at real systems, APIs, or anything you do not own and control.

If you share or fork this repo, please keep the **disclaimer and credit** to the ForgeDAN authors intact.

Happy experimenting — and break things **only** in the lab.

