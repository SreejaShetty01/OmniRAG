# Five-minute OmniRAG demo script

**Execution status:** steps 1–3 are a proposed demo flow and have not been run as part of M1-P5. Steps 4–5 are explicitly NOT RUN and require separate approvals.

1. **Ingest (about 1 min):** choose a small approved public document in an isolated local demo stack, upload it through the normal UI/API, and show ingestion status and resulting source metadata. Do not use protected or production data.
2. **Cited answer (about 1 min):** ask a question whose answer appears in that document. Show the answer, citation, evidence Details, profile, and request ID.
3. **Abstention (about 1 min):** ask an unsupported question and show the actual abstention result. Do not script or claim an outcome before execution.
4. **MiniCheck verifier catch — NOT RUN:** verifier model has not been downloaded or executed. Do not demonstrate or claim this capability until separately approved and a real catch is verified.
5. **Backup/restore — NOT RUN:** drill awaits explicit approval. Do not present a successful restore until separate-stack restore, integrity checks, five reference queries, top-K comparison, and ACL checks pass.

Before presenting, record the build revision and environment. Mask identity signatures, API keys, and shared secrets from terminal or browser output.
