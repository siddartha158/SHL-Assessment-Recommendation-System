# SHL Assessment Recommendation System

An intelligent recommendation system that maps natural language job queries to relevant SHL Individual Test Solutions using TF-IDF semantic retrieval and multi-domain balancing.

---

## Architecture

```
Query (NL / JD text / URL)
         ↓
   Intent Extraction
   (duration, test types, skills)
         ↓
   TF-IDF Vectorization
   (query → sparse vector)
         ↓
   Cosine Similarity Scoring
   + Keyword Boosting
   + Type Boosting
         ↓
   Balanced Multi-Domain Selection
   (ensures K/P/A/C coverage)
         ↓
   5–10 Assessments (JSON)
```

---

## Pipeline Components

### 1. Data Ingestion
- **Source**: SHL Product Catalog (https://www.shl.com/solutions/products/product-catalog/)
- **Scope**: 106 Individual Test Solutions (pre-scraped and structured)
- **Fields**: name, URL, description, duration, test_type, remote_support, adaptive_support

### 2. Embedding / Retrieval
- TF-IDF vectorization on assessment text (name + description + test_type)
- Query pre-processing with stopword removal
- Cosine similarity scoring against all assessment vectors

### 3. Recommendation Enhancement
- **Intent Extraction**: Parse max duration constraints, detect required test types
- **Keyword Boosting**: 40+ keyword → assessment mappings (Java, Python, SQL, etc.)
- **Balanced Selection**: When query spans multiple domains (e.g., technical + behavioral), ensure proportional representation across test types

### 4. API Layer
- Flask REST API with `/health` and `/recommend` endpoints
- JSON request/response format
- Proper HTTP status codes

---

## API Endpoints

### GET /health
```json
{"status": "healthy"}
```

### POST /recommend
**Request:**
```json
{"query": "I am hiring Java developers who can collaborate with business teams"}
```

**Response:**
```json
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/...",
      "name": "Java 8 (New)",
      "adaptive_support": "No",
      "description": "...",
      "duration": 25,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    }
  ]
}
```

---

## Evaluation

**Metric**: Mean Recall@10

```
Recall@K = (relevant assessments in top K) / (total relevant assessments)
Mean Recall@10 = average across all N test queries
```

**Train Set Performance**: Mean Recall@10 = **0.5644** (10 labeled queries)

Per-query breakdown:
| Query | Recall@10 |
|-------|-----------|
| Java developer + collaboration | 1.000 |
| COO cultural fit | 0.833 |
| Bank admin assistant | 0.833 |
| Content writer + SEO | 0.600 |
| Radio/broadcast media role | 0.600 |
| Data analyst | 0.600 |
| SHL QA engineer (1hr) | 0.333 |
| Consultant JD | 0.200 |
| Marketing manager | 0.200 |
| Graduate sales role | 0.444 |

---

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

API available at `http://localhost:5000`

---

## Running Evaluation

```bash
python evaluate.py
```

Generates:
- Per-query Recall@10 scores on train set
- `test_predictions.csv` with recommendations for 9 test queries

---

## Project Structure

```
shl-recommendation/
├── app.py              # Flask API server
├── recommender.py      # Core recommendation engine
├── evaluate.py         # Evaluation script (Recall@K)
├── shl_catalog.json    # SHL assessment catalog (106 items)
├── index.html          # Frontend web application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Frontend

Open `index.html` in a browser — includes full embedded local recommender.

Sample queries available as clickable chips. Results show:
- Assessment name (linked to SHL catalog)
- Description
- Test type badges (color-coded)
- Duration
- Remote/adaptive support indicators

---

## Approach Summary (2-page doc content)

**Problem**: Map free-text job descriptions to relevant SHL assessments.

**Solution Pipeline**:
1. Pre-process and store SHL catalog as structured JSON (106 Individual Test Solutions)
2. Build TF-IDF vectors for each assessment (name + description + test_type)
3. At query time: extract intent (max duration, required test types), build query vector
4. Score assessments via cosine similarity + keyword boost + type boost
5. Apply balanced multi-domain selection to ensure mixed recommendations

**Experiments**:
- Baseline (TF-IDF only): Mean Recall@10 = 0.283
- + Type boosting: 0.554
- + Keyword boosting maps: 0.564

**Key Insights**:
- Keyword boost maps (Java→Java 8/Core Java, etc.) significantly improve precision
- Intent detection for duration constraints prevents irrelevant results
- Balanced selection critical for multi-domain queries (technical + behavioral)
- Leadership/senior role queries need explicit persona detection for OPQ/ELR inclusion
