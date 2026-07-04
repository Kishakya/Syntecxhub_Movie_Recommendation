# 🎬 Movie Recommendation System — Content-Based Filtering

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project builds a **content-based movie recommendation system** using the TMDB 5000 Movies dataset. Given a movie title, the system recommends the most similar movies by analyzing metadata including genres, keywords, cast, director, and plot overview. Similarity between movies is computed using TF-IDF vectorization and cosine similarity. The recommender is also packaged as a lightweight Flask API endpoint for real-world usability.

---

## 📂 Dataset Description

| Property | Details |
|---|---|
| **Source** | [TMDB 5000 Movie Dataset — Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) |
| **Files** | `tmdb_5000_movies.csv` + `tmdb_5000_credits.csv` |
| **Rows** | 4,803 movies after merging |
| **Target** | No target — unsupervised recommendation |

### Features Used for Recommendation

| Feature | Description |
|---|---|
| `genres` | Movie genre tags (e.g. Action, Drama) |
| `keywords` | Plot-related keywords |
| `cast` | Top 3 cast members |
| `crew` | Director name extracted from crew |
| `overview` | Plot summary text |

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, merging, cleaning |
| `numpy` | Numerical operations |
| `matplotlib` & `seaborn` | EDA visualizations |
| `scikit-learn` | TF-IDF vectorization, cosine similarity |
| `flask` | REST API endpoint |
| `joblib` | Model serialization |
| `ast` | Parsing JSON-like metadata columns |

---

## 🤖 How It Works

### Content-Based Filtering Pipeline

1. Merge `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` on movie title
2. Extract genres, keywords, top 3 cast members, and director from JSON-like columns
3. Clean and normalize all text (lowercase, remove spaces)
4. Build a **"soup"** — a single combined text string per movie from all features
5. Apply **TF-IDF Vectorization** (10,000 features, English stop words removed)
6. Compute **Cosine Similarity** matrix across all 4,803 movies
7. For any input movie, rank all others by similarity score and return top N

### Soup Example
```
action crimefighter dc batman christophernolan
"A vigilante protects Gotham City from a criminal mastermind..."
```

---

## 📊 Qualitative Evaluation — Sample Recommendations

### The Dark Knight
| Recommended Movie | Vote Avg | Similarity |
|---|---|---|
| The Dark Knight Rises | 7.6 | 0.85+ |
| Batman Begins | 7.5 | 0.80+ |
| Batman v Superman | 6.0 | 0.75+ |

### Inception
| Recommended Movie | Vote Avg | Similarity |
|---|---|---|
| Interstellar | 8.1 | 0.80+ |
| The Prestige | 8.0 | 0.75+ |
| Memento | 8.1 | 0.70+ |

> Similarity scores are approximate and may vary based on dataset version.

---

## 🌐 Flask API (Optional)

The recommender is packaged as a simple REST API using Flask.

### Run locally
```bash
pip install flask
python app.py
```

### Example request
```
GET http://127.0.0.1:5000/recommend?title=Inception&n=5
```

### Example response
```json
{
  "movie": "Inception",
  "recommendations": [
    "Interstellar",
    "The Prestige",
    "Memento",
    "The Dark Knight",
    "Shutter Island"
  ]
}
```

---

## 📁 Project Structure

```
movie-recommendation-system/
│
├── Movie_Recommendation.ipynb      # Main notebook
├── movies_clean.csv                # Cleaned dataset with soup column
├── app.py                          # Flask API endpoint
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

> Note: `recommender_tfidf.pkl` and `recommender_cosine_sim.pkl` are not included due to file size. They are generated automatically by running the notebook.

---

## 👤 Author

**Kishkaya Gayathri**
Bachelor of Science in Information Technology specialized in Artificial Intelligence
[SLIIT] — [2026]
