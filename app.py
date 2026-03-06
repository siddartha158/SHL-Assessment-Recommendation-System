"""
SHL Assessment Recommendation API
Flask backend exposing /health and /recommend endpoints
"""
import json
import sys
import os

# Add directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from recommender import SHLRecommender

app = Flask(__name__)

# Initialize recommender once at startup
CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shl_catalog.json")
recommender = SHLRecommender(CATALOG_PATH)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Assessment recommendation endpoint.
    Accepts: {"query": "string"}
    Returns: {"recommended_assessments": [...]}
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field in request body"}), 400

    query = str(data["query"]).strip()
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        results = recommender.get_recommendations(query, top_k=10)
        # Ensure between 5 and 10 results
        results = results[:10]

        formatted = []
        for r in results:
            formatted.append({
                "url": r["url"],
                "name": r["name"],
                "adaptive_support": r["adaptive_support"],
                "description": r["description"],
                "duration": r["duration"],
                "remote_support": r["remote_support"],
                "test_type": r["test_type"],
            })

        return jsonify({"recommended_assessments": formatted}), 200

    except Exception as e:
        return jsonify({"error": f"Recommendation failed: {str(e)}"}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)