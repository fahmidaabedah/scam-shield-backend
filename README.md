# Scam Shield API

Takes a message (English/Bangla/Banglish) and tells you if it's a scam, what kind, and why. Same
model and cleaning as the Scam Message Detector notebook, wrapped as an API.

```
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

`POST /classify` with `{"message": "..."}` returns the verdict, scam score, type, and the words that
triggered it. `GET /health` for uptime checks.

Data is synthetic, so this is a prototype, not a real-world accuracy claim.
