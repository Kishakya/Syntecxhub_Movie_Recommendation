
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load saved assets
tfidf      = joblib.load("recommender_tfidf.pkl")
cosine_sim = joblib.load("recommender_cosine_sim.pkl")
movies     = pd.read_csv("movies_clean.csv")
indices    = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

@app.route("/recommend", methods=["GET"])
def recommend():
    title = request.args.get("title", "")
    n     = int(request.args.get("n", 5))

    if title not in indices:
        return jsonify({"error": f"Movie not found: {title}"}), 404

    idx         = indices[title]
    sim_scores  = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    movie_idxs  = [i[0] for i in sim_scores]
    results     = movies["title"].iloc[movie_idxs].tolist()

    return jsonify({"movie": title, "recommendations": results})

if __name__ == "__main__":
    app.run(debug=True)
