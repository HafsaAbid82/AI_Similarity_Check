from sentence_transformers import SentenceTransformer, util
HF_Summary = open("HF_Summary.txt", "r")
HFSummary = HF_Summary.read()
OpenAI_Summary = open("OpenAI_Summary.txt", "r")
OpenAISummary = OpenAI_Summary.read()
query = [HFSummary]
docs = [OpenAISummary]
model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
query_emb = model.encode(query)
doc_emb = model.encode(docs)
scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
doc_score_pairs = list(zip(docs, scores))
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
for doc, score in doc_score_pairs:
    print(score)
