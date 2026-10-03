# R006A selective postings retrieval

Isolated retrieval-only benchmark. It reuses the frozen E001 fold-0 cohort,
fold assignment, hashing vectorizers, TF-IDF construction, cached exact K10
candidates, and official grouped macro F0.5 scorer. It never writes P001.

The coarse priority is `query_tfidf_weight / sqrt(document_frequency)`, with
deterministic tie breaks by weight, lower DF, then feature index. Whole posting
lists are consumed in priority order until the soft posting budget is met. If
fewer than M distinct documents have been seen, lower-priority lists continue
until M candidates exist or all query postings are exhausted. The partial-score
top M is reranked by exact full-vector float32 TF-IDF cosine; exact-score ties
use ascending corpus row index.
