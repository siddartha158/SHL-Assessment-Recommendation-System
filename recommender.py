"""
SHL Assessment Recommendation Engine
Uses TF-IDF + keyword matching + balanced recommendation logic
"""
import json
import re
import math
from collections import Counter, defaultdict


def load_catalog(path="shl_catalog.json"):
    with open(path) as f:
        return json.load(f)


def preprocess(text):
    """Lowercase, remove punctuation, tokenize."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    stopwords = {
        'a','an','the','and','or','but','in','on','at','to','for','of','with',
        'is','are','was','were','be','been','being','have','has','had','do',
        'does','did','will','would','can','could','should','may','might',
        'i','we','you','he','she','they','it','this','that','these','those',
        'my','your','our','their','its','who','what','which','when','where',
        'how','if','then','than','so','as','up','down','not','no','any','all',
        'some','more','most','also','just','only','very','well','new','get',
        'use','need','want','look','make','take','go','know','good','best',
        'years','year','experience','work','role','job','position','hire',
        'hiring','looking','seeking','find','help','test','assessment',
        'candidate','application','screen','recommend','suggestion','please',
    }
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def build_tfidf(catalog):
    """Build TF-IDF vectors for all assessments."""
    docs = []
    for item in catalog:
        text = f"{item['name']} {item['description']} {' '.join(item['test_type'])}"
        docs.append(preprocess(text))

    N = len(docs)
    df = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    idf = {term: math.log((N + 1) / (count + 1)) + 1 for term, count in df.items()}

    vectors = []
    for doc in docs:
        tf = Counter(doc)
        total = sum(tf.values()) or 1
        vec = {term: (count / total) * idf.get(term, 1) for term, count in tf.items()}
        vectors.append(vec)

    return vectors, idf


def cosine_similarity(vec1, vec2):
    common = set(vec1) & set(vec2)
    if not common:
        return 0.0
    dot = sum(vec1[k] * vec2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# Keyword → catalog name substring boosting
KEYWORD_BOOST_MAP = {
    'java': ['java', 'automata fix'],
    'python': ['python'],
    'sql': ['sql', 'automata - sql', 'data warehousing', 'sql server'],
    'javascript': ['javascript', 'js'],
    'selenium': ['selenium', 'automata - selenium'],
    'excel': ['excel', 'microsoft excel'],
    'tableau': ['tableau'],
    'power bi': ['power bi'],
    'coo': ['enterprise leadership', 'opq leadership', 'opq team', 'global skills'],
    'ceo': ['enterprise leadership', 'opq leadership', 'global skills'],
    'cfo': ['enterprise leadership', 'financial professional'],
    'cultural': ['occupational personality', 'opq', 'global skills'],
    'analyst': ['analyst', 'verify numerical', 'verify verbal', 'verify numerical', 'inductive reasoning'],
    'cognitive': ['verify numerical', 'verify verbal', 'inductive reasoning', 'deductive reasoning', 'cognitive', 'shl verify'],
    'personality': ['occupational personality', 'opq', 'motivation questionnaire', 'work strengths', 'personality assessment'],
    'english': ['english comprehension', 'written english', 'svar', 'business communication', 'workplace english'],
    'seo': ['search engine optimization'],
    'content writ': ['content writing', 'written english', 'english comprehension', 'search engine optimization'],
    'marketing': ['marketing', 'digital advertising', 'search engine optimization'],
    'bank': ['bank administrative', 'financial professional', 'verify numerical'],
    'admin': ['administrative professional', 'bank administrative', 'general entry level'],
    'data entry': ['data entry', 'basic computer literacy'],
    'sales': ['entry level sales', 'sales representative', 'technical sales', 'customer facing', 'writex'],
    'customer service': ['customer service', 'customer support', 'call center', 'contact center'],
    'customer support': ['customer support', 'customer service', 'call center', 'contact center'],
    'leadership': ['enterprise leadership', 'opq leadership', 'leadership assessment', 'manager 8'],
    'manager': ['manager 8', 'leadership assessment', 'enterprise leadership'],
    'product manag': ['product management', 'jira', 'scrum', 'agile', 'confluence'],
    'scrum': ['scrum', 'agile', 'jira'],
    'agile': ['agile', 'scrum', 'jira'],
    'devops': ['devops', 'aws', 'azure'],
    'aws': ['aws'],
    'azure': ['microsoft azure'],
    'machine learning': ['machine learning', 'data science', 'python'],
    'data science': ['data science', 'python', 'machine learning', 'r programming'],
    'css': ['css3', 'html/css', 'htmlcss'],
    'html': ['html/css', 'htmlcss', 'css3'],
    'react': ['react'],
    'angular': ['angular'],
    'communication': ['business communication', 'written english', 'interpersonal', 'svar', 'communication skills'],
    'collaborate': ['teamwork', 'interpersonal', 'occupational personality'],
    'teamwork': ['teamwork', 'interpersonal'],
    'consul': ['professional 7', 'verify numerical', 'verify verbal', 'occupational personality', 'shl verify'],
    'radio': ['verify verbal', 'english comprehension', 'marketing', 'interpersonal', 'communication'],
    'sound': ['verify verbal', 'english comprehension', 'marketing', 'interpersonal'],
    'broadcast': ['verify verbal', 'english comprehension', 'marketing'],
    'drupal': ['drupal'],
    'jira': ['jira'],
    'graduate': ['graduate', 'entry level'],
    'data analyst': ['sql server', 'automata - sql', 'tableau', 'excel', 'data warehousing', 'python', 'professional 7'],
    'brand': ['marketing', 'digital advertising', 'verify', 'shl verify', 'writex', 'manager 8'],
    'content strategy': ['marketing', 'digital advertising', 'writex', 'search engine'],
    'positioning': ['marketing', 'digital advertising', 'manager 8', 'excel'],
    'inductive': ['shl verify interactive', 'inductive reasoning', 'verify'],
    'verify': ['shl verify', 'verify numerical', 'verify verbal'],
    'numerical': ['verify numerical', 'shl verify interactive numerical', 'numerical reasoning', 'analyst short'],
    'spoken english': ['svar', 'english comprehension', 'written english'],
    'communication skill': ['business communication', 'interpersonal', 'svar', 'written english', 'communication skills'],
    'business communication': ['business communication', 'interpersonal', 'written english'],
    'interpersonal': ['interpersonal', 'teamwork', 'occupational personality'],
    'junior sales': ['entry level sales', 'business communication', 'svar', 'interpersonal', 'english comprehension'],
    'sales graduate': ['entry level sales', 'business communication', 'svar', 'interpersonal', 'english comprehension'],
    'new graduate sales': ['entry level sales', 'business communication', 'svar', 'interpersonal'],

    'fresher': ['graduate', 'entry level'],
    'new grad': ['graduate', 'entry level'],
}


def extract_query_intent(query_text):
    """Extract key signals from query for better matching."""
    query_lower = query_text.lower()
    intent = {
        'max_duration': None,
        'test_types_needed': [],
    }

    # Duration extraction
    duration_patterns = [
        (r'(\d+)\s*hour', 60),
        (r'(\d+)\s*min(?:utes?)?', 1),
        (r'max(?:imum)?\s+(?:duration|time)\s+(?:of\s+)?(\d+)', 1),
        (r'(?:within|less than|under|max|maximum)\s+(\d+)\s*min', 1),
    ]
    for pattern, mult in duration_patterns:
        m = re.search(pattern, query_lower)
        if m:
            intent['max_duration'] = int(m.group(1)) * mult
            break

    # Test type signals
    if any(kw in query_lower for kw in ['cognitive', 'aptitude', 'reasoning', 'numerical', 'verbal', 'logical', 'inductive', 'deductive']):
        intent['test_types_needed'].append('Ability & Aptitude')
    if any(kw in query_lower for kw in ['personality', 'behavior', 'behaviour', 'soft skill', 'attitude', 'trait', 'opq']):
        intent['test_types_needed'].append('Personality & Behavior')
    if any(kw in query_lower for kw in ['java', 'python', 'sql', 'javascript', 'coding', 'programming', 'technical',
                                         'excel', 'software', 'react', 'angular', 'node', 'c#', 'dotnet', 'aws',
                                         'azure', 'devops', 'selenium', 'automation', 'tableau']):
        intent['test_types_needed'].append('Knowledge & Skills')
    if any(kw in query_lower for kw in ['competen', 'leadership', 'management', 'communication', 'teamwork',
                                         'collaboration', 'stakeholder', 'interpersonal', 'collaborate', 'cross-functional']):
        intent['test_types_needed'].append('Competencies')
    if any(kw in query_lower for kw in ['sales', 'selling', 'revenue', 'commercial']):
        if 'Personality & Behavior' not in intent['test_types_needed']:
            intent['test_types_needed'].append('Personality & Behavior')

    # Leadership/senior roles
    if any(kw in query_lower for kw in ['coo', 'ceo', 'cfo', 'cto', 'senior', 'director', 'executive',
                                          'manager', 'head of', 'vp ', 'vice president', 'cultural', 'global']):
        for tt in ['Personality & Behavior', 'Competencies', 'Ability & Aptitude']:
            if tt not in intent['test_types_needed']:
                intent['test_types_needed'].append(tt)

    # Graduate/entry level
    if any(kw in query_lower for kw in ['graduate', 'fresher', 'entry level', 'new grad', 'junior']):
        for tt in ['Ability & Aptitude', 'Personality & Behavior']:
            if tt not in intent['test_types_needed']:
                intent['test_types_needed'].append(tt)

    return intent


def recommend(query_text, catalog, tfidf_vectors, idf, top_k=10):
    """Main recommendation function with balanced multi-domain logic."""
    intent = extract_query_intent(query_text)
    max_dur = intent['max_duration']
    needed_types = intent['test_types_needed']
    query_lower = query_text.lower()

    # Build query vector
    q_tokens = preprocess(query_text)
    q_tf = Counter(q_tokens)
    q_total = sum(q_tf.values()) or 1
    q_vec = {term: (count / q_total) * idf.get(term, 1) for term, count in q_tf.items()}

    # Keyword boost map for this query
    keyword_boosts = {}
    for kw, catalog_names in KEYWORD_BOOST_MAP.items():
        if kw in query_lower:
            for i, item in enumerate(catalog):
                name_lower = item['name'].lower()
                for cname in catalog_names:
                    if cname in name_lower:
                        keyword_boosts[i] = keyword_boosts.get(i, 0) + 0.3

    # Score all assessments
    scores = []
    for i, (item, vec) in enumerate(zip(catalog, tfidf_vectors)):
        sim = cosine_similarity(q_vec, vec)

        type_boost = sum(0.12 for tt in item['test_type'] if tt in needed_types)
        kw_boost = keyword_boosts.get(i, 0)

        duration_ok = True
        if max_dur is not None and item['duration'] > max_dur:
            duration_ok = False

        final_score = (sim + type_boost + kw_boost) if duration_ok else 0.0
        scores.append((i, final_score))

    scores.sort(key=lambda x: x[1], reverse=True)

    # Balanced multi-domain selection
    if len(needed_types) > 1:
        results = _balanced_pick(scores, catalog, needed_types, max_dur, top_k)
    else:
        results = []
        for idx, score in scores:
            if score > 0:
                results.append(catalog[idx])
            if len(results) >= top_k:
                break

    # Ensure minimum 5 results
    if len(results) < 5:
        for idx, score in scores:
            item = catalog[idx]
            if item not in results:
                results.append(item)
            if len(results) >= 5:
                break

    return results[:top_k]


def _balanced_pick(scores, catalog, needed_types, max_dur, top_k):
    """Pick assessments ensuring each needed type has representation."""
    per_type = defaultdict(list)
    all_passing = []

    for idx, score in scores:
        if score <= 0:
            continue
        item = catalog[idx]
        if max_dur is not None and item['duration'] > max_dur:
            continue
        all_passing.append((idx, score))
        for tt in item['test_type']:
            if tt in needed_types:
                per_type[tt].append((idx, score))

    slots_per_type = max(1, top_k // len(needed_types))
    selected_indices = set()
    results = []

    for tt in needed_types:
        count = 0
        for idx, _ in per_type[tt]:
            if idx not in selected_indices:
                results.append(catalog[idx])
                selected_indices.add(idx)
                count += 1
            if count >= slots_per_type:
                break

    for idx, _ in all_passing:
        if idx not in selected_indices:
            results.append(catalog[idx])
            selected_indices.add(idx)
        if len(results) >= top_k:
            break

    return results


class SHLRecommender:
    """Main recommender class to initialize once and reuse."""

    def __init__(self, catalog_path="shl_catalog.json"):
        self.catalog = load_catalog(catalog_path)
        self.tfidf_vectors, self.idf = build_tfidf(self.catalog)

    def get_recommendations(self, query: str, top_k: int = 10):
        recs = recommend(query, self.catalog, self.tfidf_vectors, self.idf, top_k)
        return [
            {
                "name": r["name"],
                "url": r["url"],
                "description": r["description"],
                "duration": r["duration"],
                "test_type": r["test_type"],
                "remote_support": r["remote_support"],
                "adaptive_support": r["adaptive_support"],
            }
            for r in recs
        ]
