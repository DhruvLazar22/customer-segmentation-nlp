# Multi-Modal Customer Segmentation: NLP + K-Means + LSTM

Groups customers by **what they say** in reviews (semantic clustering) and **how their sentiment changes** over time (LSTM). This goes further than traditional RFM or TF-IDF segmentation.

**Course:** Advanced Cloud Computing, VIT Vellore
**Team:** Surendra Patil, Yuval Gupta, Dhruv Lazar
**Tools:** Python, SentenceTransformers, scikit-learn, TensorFlow/Keras, PCA, Matplotlib

---

## Pipeline

```
Amazon reviews → clean / tokenize / lemmatize → SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
     → K-Means (k = 6) → PCA 2-D visualisation
     → LSTM sentiment model → sentiment per cluster → customer personas
```

| Stage | Details |
|---|---|
| Data | Amazon Customer Reviews (Kaggle): review text, star ratings, product info |
| Preprocessing | Remove HTML and symbols, tokenize, remove stopwords, lemmatize, lowercase |
| Embeddings | `all-MiniLM-L6-v2`. Similar meanings land close together ("fast delivery" ≈ "quick shipping"). |
| Clustering | K-Means, with **k = 6** chosen using the elbow method + silhouette score |
| Sentiment | Embedding → LSTM → dropout → sigmoid output. Binary cross-entropy loss, Adam optimizer. |

## Results

- **About 90% validation accuracy** for sentiment classification, with stable convergence.
- Clearly separated clusters in the PCA plot, matching recognisable customer groups: satisfied, price-sensitive, quality complaints, service issues, mixed.
- Sentiment heatmaps per cluster show which customer groups are happy and which are at risk.

## Why it matters

Businesses can find out **why** each customer group is happy or unhappy, not just how often they buy. That supports targeted marketing, retention and product fixes.

## Future work

- Real-time streaming reviews
- Transformer-based sentiment model
- Deploying the pipeline as a cloud service
