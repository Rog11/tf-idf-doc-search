from flask import Flask, request, jsonify, render_template
import uuid
import re
import string
import math
from typing import List, Dict, Tuple
from collections import defaultdict

import re
import math
from typing import List, Dict, Tuple
from collections import defaultdict

class TFIDFSearcher:
    def __init__(self, stop_words: set):
        self.documents = {}           # doc_id → original_text
        self.sentences = {}           # doc_id → list of (sentence_text, start_char, end_char)
        self.token_spans = {}         # doc_id → [(token, start_char, end_char), ...]
        self.tokens = {}              # doc_id → [tokens]
        self.vocab = set()
        
        # Inverted index: term → {doc_id: [token_positions]}
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        
        self.df = defaultdict(int)    # document frequency
        self.N = 0                    # total documents
        self.stop_words = stop_words
        
        self.sentence_vectors = {}    # (doc_id, sent_idx) → tf-idf vector dict {term: weight}

        # NEW: Language detection
        self.language_profiles = {}   # lang → letter freq dict
        self.doc_languages = {}       # doc_id → lang

    def get_letter_freq(self, text: str) -> Dict[str, float]:
        lowered = text.lower()
        letters = [c for c in lowered if c.isalpha()]
        if not letters:
            return {}
        from collections import Counter
        count = Counter(letters)
        total = len(letters)
        freq = {c: count[c] / total for c in count}
        return freq

    def detect_language(self, text: str) -> str:
        freq = self.get_letter_freq(text)
        if not freq or not self.language_profiles:
            return 'unknown'
        scores = {lang: self.cosine_similarity(freq, prof) 
                  for lang, prof in self.language_profiles.items()}
        return max(scores, key=scores.get) if scores else 'unknown'

    def preprocess_with_offsets(self, text: str) -> List[Tuple[str, int, int]]:
        original_text = text
        lowered = original_text.lower()
        
        # Remove HTML/XML tags if any
        lowered = re.sub(r'<[^>]+>', ' ', lowered)
        
        # Normalize whitespace but keep newlines for line calculation
        lowered = re.sub(r'[ \t]+', ' ', lowered)
        
        # Tokenize: find word spans (alphanumeric, handles accents for French)
        tokens_with_spans = []
        for match in re.finditer(r'\b\w+\b', lowered):
            start, end = match.start(), match.end()
            token = lowered[start:end]
            if token not in self.stop_words and len(token) > 1:
                tokens_with_spans.append((token, start, end))
        
        return tokens_with_spans

    def split_into_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        # Simple regex split (handles .!?; improve with better logic if needed)
        sentences = []
        start = 0
        for match in re.finditer(r'[.!?;]\s*', text):
            end = match.end()
            sent = text[start:end].strip()
            if sent:
                sentences.append((sent, start, end))
            start = end
        if start < len(text):
            sentences.append((text[start:].strip(), start, len(text)))
        return sentences

    def compute_sentence_vectors(self, doc_id: str):
        # Build TF-IDF vectors for each sentence using global vocab
        for sent_idx, (sent_text, _, _) in enumerate(self.sentences[doc_id]):
            sent_tokens = [t for t, _, _ in self.preprocess_with_offsets(sent_text)]  # Tokenize sentence
            vector = {}
            term_freq = defaultdict(int)
            for token in sent_tokens:
                term_freq[token] += 1
            for token in term_freq:
                tf = term_freq[token] / len(sent_tokens) if sent_tokens else 0
                idf = math.log(self.N / (1 + self.df.get(token, 0)))  # Use doc-level IDF for simplicity
                vector[token] = tf * idf
            self.sentence_vectors[(doc_id, sent_idx)] = vector

    def add_document(self, doc_id: str, text: str):
        self.documents[doc_id] = text
        self.sentences[doc_id] = self.split_into_sentences(text)  # New: split into sentences
        spans = self.preprocess_with_offsets(text)
        self.token_spans[doc_id] = spans
        self.tokens[doc_id] = [t for t, s, e in spans]
        self.N += 1
        
        # Build inverted index with positions in token list
        for pos, (token, _, _) in enumerate(spans):
            self.inverted_index[token][doc_id].append(pos)
            self.vocab.add(token)

        # NEW: Detect language for new documents
        if doc_id not in ['alice', 'candideEn', 'candideFr']:
            lang = self.detect_language(text)
            self.doc_languages[doc_id] = lang

    def build_index(self):
        # 1. fill DF
        for term in self.vocab:
            self.df[term] = len(self.inverted_index[term])

        # 2. now build *all* sentence vectors
        for doc_id in self.documents:
            self.compute_sentence_vectors(doc_id)

    def compute_tf_idf(self, term: str, doc_id: str) -> float:
        if doc_id not in self.inverted_index[term]:
            return 0.0
        
        tf = len(self.inverted_index[term][doc_id])  # term frequency in doc
        idf = math.log(self.N / (1 + self.df[term])) # smooth IDF
        return tf * idf

    def score_document(self, query_tokens: List[str], doc_id: str) -> Tuple[float, Dict[str, float]]:
        term_scores = {token: self.compute_tf_idf(token, doc_id) for token in query_tokens}
        total_score = sum(term_scores.values())
        return total_score, term_scores

    def get_line_number(self, text: str, char_offset: int) -> int:
        return text[:char_offset].count('\n') + 1

    def get_term_locations(self, doc_id: str, query_tokens: List[str]) -> List[Tuple[int, int, int]]:
        locations = []
        seen_tokens = set(query_tokens)
        for token in seen_tokens:
            if doc_id in self.inverted_index[token]:
                for pos in self.inverted_index[token][doc_id]:
                    _, start, end = self.token_spans[doc_id][pos]
                    line = self.get_line_number(self.documents[doc_id], start)
                    locations.append((start, end, line))
        return sorted(locations, key=lambda x: x[0])

    def make_snippet(self, doc_id: str, locations: List[Tuple[int, int, int]], context: int = 50) -> str:
        text = self.documents[doc_id]
        snippets = []
        for start, end, line in locations[:3]:  # show first 3 matches
            s = max(0, start - context)
            e = min(len(text), end + context)
            before = text[s:start].replace('\n', ' ')
            match = text[start:end]
            after = text[end:e].replace('\n', ' ')
            snippet = f"{before}**{match}**{after}"
            if s > 0: snippet = "..." + snippet
            if e < len(text): snippet += "..."
            snippets.append(snippet)
        return " | ".join(snippets)

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        intersection = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[term] * vec2[term] for term in intersection)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0

    def _word_search(self, query: str, top_k: int = 5) -> List[Dict]:
        # Preprocess query using same logic (but no spans needed)
        lowered_query = query.lower()
        query_tokens = [match.group(0) for match in re.finditer(r'\b\w+\b', lowered_query) 
                        if match.group(0) not in self.stop_words and len(match.group(0)) > 1]
        
        if not query_tokens:
            return []
        
        # Get candidate docs
        candidate_docs = set()
        for token in query_tokens:
            candidate_docs.update(self.inverted_index[token].keys())
        
        # Rank candidates
        results = []
        for doc_id in candidate_docs:
            score, term_scores = self.score_document(query_tokens, doc_id)
            if score > 0:
                results.append((doc_id, score, term_scores))
        
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        # Build rich results with locations
        final_results = []
        for doc_id, score, term_scores in top_results:
            locations = self.get_term_locations(doc_id, query_tokens)
            final_results.append({
                'doc_id': doc_id,
                'score': round(score, 4),
                'term_scores': {k: round(v, 4) for k, v in term_scores.items()},
                'snippet': self.make_snippet(doc_id, locations),
                'locations': locations,  # list of (char_start, char_end, line)
                'language': self.doc_languages.get(doc_id, 'unknown')  # NEW
            })
        
        return final_results

    def search(self, query: str, top_k: int = 5, mode: str = 'sentence') -> List[Dict]:
        if mode == 'word':  # Fallback to existing word search
            return self._word_search(query, top_k)

        # Sentence mode
        lowered_query = query.lower()
        query_tokens = [match.group(0) for match in re.finditer(r'\b\w+\b', lowered_query) 
                        if match.group(0) not in self.stop_words and len(match.group(0)) > 1]
        
        query_vector = {}
        term_freq = defaultdict(int)
        for token in query_tokens:
            term_freq[token] += 1
        for token in term_freq:
            tf = term_freq[token] / len(query_tokens) if query_tokens else 0
            idf = math.log(self.N / (1 + self.df.get(token, 0)))
            query_vector[token] = tf * idf

        results = []
        for doc_id in self.documents:
            for sent_idx, _ in enumerate(self.sentences[doc_id]):
                sent_vector = self.sentence_vectors.get((doc_id, sent_idx), {})
                similarity = self.cosine_similarity(query_vector, sent_vector)
                if similarity > 0:
                    results.append((doc_id, sent_idx, similarity))

        results.sort(key=lambda x: x[2], reverse=True)
        top_results = results[:top_k]

        final_results = []
        for doc_id, sent_idx, sim in top_results:
            sent_text, start, end = self.sentences[doc_id][sent_idx]
            locations = self.get_term_locations(doc_id, query_tokens)  # Existing, filter to sentence if needed
            final_results.append({
                'doc_id': doc_id,
                'sentence_idx': sent_idx,
                'similarity_score': round(sim, 4),
                'snippet': sent_text,  # Full sentence as snippet, or use make_snippet
                'locations': [loc for loc in locations if start <= loc[0] < end],  # Filter locations to this sentence
                'language': self.doc_languages.get(doc_id, 'unknown')  # NEW
            })

        return final_results

    # -------------------------------------------------
# 1. Keep the old endpoint (top-50) – but return *float* idf
# -------------------------------------------------
    def get_overall_tf_idf(self, top_k: int = 50) -> List[Dict]:
        idf_list = []
        for term in self.vocab:
            df = self.df[term]
            idf = math.log(self.N / (1 + df))          # <-- raw float
            # ---- NEW: never round to zero ----
            if idf < 0.0001 and idf > -0.0001:          # tiny values become 0.0001
                idf = 0.0001
            term_tf_idf = {}
            for doc_id in self.documents:
                tf = len(self.inverted_index[term].get(doc_id, []))
                term_tf_idf[doc_id] = round(tf * idf, 6)   # keep 6 decimals
            idf_list.append({
                'term': term,
                'df': df,
                'idf': round(idf, 6),                      # <-- float, not string
                'tf_idf': term_tf_idf
            })
        idf_list.sort(key=lambda x: x['idf'], reverse=True)
        return idf_list[:top_k]

app = Flask(__name__)

# Load stop words
with open('StopWords.txt', 'r', encoding='utf-8') as f:
    stop_words = set(line.strip() for line in f if line.strip())

searcher = TFIDFSearcher(stop_words)

# Load initial documents
with open('Alice.txt', 'r', encoding='utf-8') as f:
    searcher.add_document('alice', f.read())

with open('CandideEn.txt', 'r', encoding='utf-8') as f:
    searcher.add_document('candideEn', f.read())

with open('CandideFr.txt', 'r', encoding='utf-8') as f:
    searcher.add_document('candideFr', f.read())

# NEW: Compute language profiles from initial documents
en_text = searcher.documents['alice'] + searcher.documents['candideEn']
fr_text = searcher.documents['candideFr']
searcher.language_profiles = {
    'en': searcher.get_letter_freq(en_text),
    'fr': searcher.get_letter_freq(fr_text)
}
searcher.doc_languages = {
    'alice': 'en',
    'candideEn': 'en',
    'candideFr': 'fr'
}

searcher.build_index()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add_document', methods=['POST'])
def add_document():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    doc_id = str(uuid.uuid4())
    searcher.add_document(doc_id, text)
    searcher.build_index()  # Rebuild for simplicity; optimize for production
    return jsonify({'doc_id': doc_id}), 201

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    mode = data.get('mode', 'sentence')  # Default to sentence
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    results = searcher.search(query, mode=mode)
    return jsonify(results), 200

@app.route('/overall', methods=['GET'])
def overall():
    results = searcher.get_overall_tf_idf()
    return jsonify(results), 200

# -------------------------------------------------
# TF-distribution per document
# -------------------------------------------------
@app.route('/tf_histogram', methods=['GET'])
def tf_histogram():
    result = []
    for doc_id in searcher.documents:
        tf_counts = defaultdict(int)
        for term, positions in searcher.inverted_index.items():
            tf = len(positions.get(doc_id, []))
            if tf > 0:
                tf_counts[tf] += 1
        data = [[tf, count] for tf, count in sorted(tf_counts.items())]
        result.append({"doc_id": doc_id, "tf_counts": data})
    return jsonify(result)

# -------------------------------------------------
# Rank-Frequency (Zipf) data per document
# -------------------------------------------------
@app.route('/rank_frequency', methods=['GET'])
def rank_frequency():
    result = []
    for doc_id in searcher.documents:
        # Build TF for every term in this doc
        term_tf = {}
        for term, positions in searcher.inverted_index.items():
            tf = len(positions.get(doc_id, []))
            if tf > 0:
                term_tf[term] = tf

        # Sort by frequency descending → rank
        sorted_items = sorted(term_tf.items(), key=lambda x: x[1], reverse=True)
        data = [[rank+1, tf] for rank, (term, tf) in enumerate(sorted_items)]
        result.append({"doc_id": doc_id, "rank_freq": data})
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)