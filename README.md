# RecommendationSystemDive
---

## TL;DR

This project explores a simple two-stage music retrieval pipeline: a fast **text-based candidate search** (SBERT) over generated captions and tags, followed by an **audio–text reranking** using CLAP embeddings to pick tracks that actually sound like the query. The dataset used is the Jamendo MTG autotagging subset. Embeddings are precomputed. This repo contains code, notebooks, examples and small demos showing the system in action.

---

# Part 1 — Research & data story (what I did and why)

### 1) Data collection

* **Source:** MTG Jamendo Dataset — autotagging_moodtheme subset. (See: [https://github.com/MTG/mtg-jamendo-dataset](Bogdanov D, Won M, Tovstogan P, Porter A, Serra X. The MTG-Jamendo dataset for automatic music tagging. Paper presented at: ML4MD Machine Learning for Music Discovery Workshop at ICML2019; 2019 Jun 15; Long Beach, California.))
* **Subset used:** `autotagging_moodtheme.tsv` (≈18,486 rows, 59 mood/theme tags).
* **Why I chose it:**

  * Public, well-annotated dataset specifically meant for music tagging and research — friendly for exploratory work and reproducibility.
  * Provides tag-level metadata that suits semantic retrieval experiments (moods, themes, instrument info).
  * I experimented with Spotify API but found metadata access increasingly limited and audio download impractical due to copyright and API restrictions — hence a public research dataset is the pragmatic alternative.

### 2) Data engineering

* Merged tags and metadata from different dataset subsets into a single consolidated DataFrame (track id, artists, album, mood/theme (compleate), genre (almost complete),instrumental flags (half missing)).
* Cleaned and normalized tags (lowercasing, de-duplication, simple stemming/lemmatization where appropriate).
* Generated rich textual captions for tracks by expanding sparse tag lists using a text-generation model (**Qwen2-audio** / LLM-assisted captioning). This step helps SBERT see more descriptive text per track.
* Extracted keywords from captions (TF-IDF + heuristics) to emphasize important words for retrieval.
* Built a final dataset with consolidated metadata fields, a caption column and a keywords column describing audio, that the embedding pipeline consumes.

### 3) Embedding building

* **Design choice:** use *two* embedding spaces rather than a single joint space.

  * **Text-only embeddings** (SBERT) are fast to compute and cheap to search; they serve as a quick candidate generator. SBERT is robust for semantic similarity on text (captions/tags). Used FAISS idexing for speed too.
  * **CLAP audio-text embeddings** are used for **reranking** because they measure cross-modal (audio↔text) compatibility — this helps weed out results that read well in text but do not sound like the query. (This model was used: https://github.com/LAION-AI/CLAP?tab=readme-ov-file)
* **Why not single-space only:** single-space approaches can be elegant but often require heavier computation and more complex training/fine-tuning. The two-stage approach is a pragmatic engineering tradeoff: fast candidate retrieval + higher-quality final ranking.
* **Practicality:** embeddings were precomputed and stored. 

---

# Part 2 — Main pipeline (high-level overview)

**Pipeline flow (high-level):**

1. **User query (text)** — the user writes a short prompt, e.g. "chill piano with sax and late-night vibe".
2. **Text candidate search** — embed the query with SBERT and perform a fast nearest-neighbor search over precomputed text embeddings to produce N candidate tracks (e.g., N=300).
3. **CLAP rerank** — compute CLAP text embedding for the query and compare it against precomputed CLAP audio embeddings for the N candidates. Sort by CLAP similarity and present top-K final results.
4. **Output** — return metadata and paths to locally stored audio.

```
User query -> SBERT search (fast) -> candidate set -> CLAP rerank (precise) -> final list
```

**Why this is nice:** quick response during exploration, while maintaining good audio-text alignment in the final results.

---

# Examples:

### 1. Prompt: "Dark aesthetic rock piece with female vocals and electro guitar"
Top result:
[▶️ Download audio](examples\rock_example.mp3)

### 2. Prompt: "A fast techno with futuristic vibe and synth"
Top result:
[▶️ Download audio](examples\techno_futuristic.mp3)

### 3. I also got a video recorded to showcase how I use the recomendation system to recieve a valid recommendations:

[![Watch the video](https://img.youtube.com/vi/pfGHCjmyatg/maxresdefault.jpg)](https://youtu.be/pfGHCjmyatg)


Prompt: "Morning yoga soundtrack with sounds of ocean waves, nature"  
Top result: [▶️ Download audio](examples\yoga_nature.mp3)



---

# How to try this project (guide)

> Important: I cannot include the full dataset in the repo. You can reproduce results if you: 1) download the Jamendo MTG dataset yourself (audios). 2) clone this repo. 3) use my precomputed embeddings and run the pipeline. 4) test results by plaing the audio. 

> Silly, but I don't have money to host everything.. (If you are desparate to try, you can cotanct me with your prompt, or I will do a short version of the dataset, compute emmbeddings for them and push here)


---

# Project status, limitations & ethics

* Status: experimental, educational. I built it as a learning project for my portfolio — please treat this as a research prototype, not a production system.
* Limitations: results depend heavily on captions quality and dataset coverage. The Jamendo dataset is smaller and biased toward free/shared tracks; it will not match commercial catalogs.
* Ethics: do not redistribute copyrighted audio; follow dataset license. Any usage beyond experimentation should consider artist rights.

---
