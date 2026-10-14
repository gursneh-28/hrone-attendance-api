# Employee Attendance & Analytics API

FastAPI + MongoDB. All code is in `app/main.py`.

## Run

Needs Python 3.11+ and MongoDB 6.0 or newer.

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
set MONGO_URI=mongodb://localhost:27017   # Linux/macOS: export ...
set MONGO_DB=attendance_db
uvicorn app.main:app --port 8000
```

`MONGO_URI` and `MONGO_DB` are read from the environment (a local `.env` is also loaded, but real environment
variables win). Indexes are created at startup and the call is idempotent. `GET /health` returns 200 only when MongoDB
answers a ping, otherwise 503.

## Status

Every endpoint in the contract is implemented, including the five analytics endpoints and `/admin/explain/{endpoint}`.
Nothing is missing.

I checked it against a local MongoDB 7 (Docker), including a generated database of about 96,000 attendance records
holding only the default `_id` index: the app is ready in about 2.5 s, the explain output of every endpoint shows an
`IXSCAN` and no `COLLSCAN`, and each request answers in well under a second.

## Choices where the contract left room

- **Punch-out earlier than every punch-in** returns 404 (no record has a punch-in at or before that instant). A
  punch-out equal to the punch-in, or more than 24 hours after it, returns 422.
- **Concurrent corrections (PATCH)**: the update applies only if the record is unchanged since it was read; the loser gets
  409, so no history entry is lost.
- **Moving average (trend)** is the mean of the reported, already rounded, `attendance_rate` values. Rounding is exact
  half-up everywhere (rates are handled as whole numbers of 1/10000).
- **Department summary** ignores logs of employees who joined after the month, consistent with the headcount.
- **Trend range**: up to 92 days counting both ends.