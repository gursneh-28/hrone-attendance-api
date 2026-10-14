"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience; real env vars win)

Time convention used everywhere in this file:
  * Inside the app, every instant is a timezone-aware UTC datetime, truncated to whole seconds.
  * The API speaks integer epoch MILLISECONDS; we convert only at the boundary (from_ms / to_ms).
  * IST (UTC+05:30) is used only to get a calendar date or to place a shift time (R1).
"""
import os
import json
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from bson import json_util
from bson.son import SON
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()  # override=False by default, so real environment variables take priority

# tz_aware=True: reads come back as aware UTC datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
EPOCH_MS_MIN = 100_000_000_000      # anything smaller looks like seconds -> 422
EPOCH_MS_MAX = 4_102_444_800_000    # year 2100

PRESENCE = ("PRESENT", "WFH", "ON_DUTY")
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]
# strict=True rejects floats ("1.5e12"), strings and booleans; ge/le reject seconds and out-of-range values.
EpochMs = Annotated[int, Field(strict=True, ge=EPOCH_MS_MIN, le=EPOCH_MS_MAX)]

HHMM = r"^([01]\d|2[0-3]):[0-5]\d$"
YMD = r"^\d{4}-\d{2}-\d{2}$"
MONTH_RE = r"^\d{4}-(0[1-9]|1[0-2])$"


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


def ensure_indexes() -> None:
    """Idempotent: creating an index that already exists is a no-op."""
    # employees: emp_code is the business key -> unique (this is what makes duplicate creates race-safe).
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="uq_emp_code")
    # employees filtered by department and sorted by emp_code.
    db.employees.create_index([("department", ASCENDING), ("emp_code", ASCENDING)], name="dept_emp_code")
    # department headcount: employees joined by a given day, grouped by department (answered from the index alone).
    db.employees.create_index([("joined_on", ASCENDING), ("department", ASCENDING)], name="joined_dept")
    # department trend: every joined_on of a department, read from the index alone.
    db.employees.create_index([("department", ASCENDING), ("joined_on", ASCENDING)], name="dept_joined")
    # attendance: one record per (emp_code, date) -> unique (race-safe punch-in; also serves emp_code filters).
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uq_emp_date")
    # attendance list: default sort is date desc, emp_code asc.
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)], name="date_desc_emp")
    # attendance list filtered by status.
    db.attendance_logs.create_index(
        [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="status_date_emp")
    # punch-out: latest record of an employee whose punch_in is at or before a given instant.
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="emp_punch_in")


 
# Time helpers (the boundary between epoch milliseconds and aware datetimes)
 
def from_ms(ms: int) -> datetime:
    """Epoch milliseconds -> aware UTC datetime, truncated to whole seconds (R1)."""
    return (EPOCH + timedelta(milliseconds=ms)).replace(microsecond=0)


def to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Datetime -> integer epoch milliseconds. A naive value is treated as UTC (that is how Mongo stores it)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)


MAX_SKIP = 2**62  # far beyond any real collection, but small enough for MongoDB's 64-bit integers


def page_skip(page: int, page_size: int) -> int:
    """Documents to skip for a page. Capped so an absurd `page` returns an empty page instead of overflowing."""
    return min((page - 1) * page_size, MAX_SKIP)


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def now_millis() -> datetime:
    """Current time at millisecond precision (what Mongo stores), for fields like created_at and history.at."""
    now = datetime.now(UTC)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


def parse_hhmm(value: str) -> tuple[int, int]:
    h, m = value.split(":")
    return int(h), int(m)


def is_overnight(shift_start: str, shift_end: str) -> bool:
    return parse_hhmm(shift_end) <= parse_hhmm(shift_start)


def attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """R1: the IST calendar date of the punch-in, except that on an overnight shift a punch-in earlier
    than shift_end belongs to the shift that started the previous day."""
    local = punch_in.astimezone(IST)
    d = local.date()
    if is_overnight(shift_start, shift_end) and local.time() < time(*parse_hhmm(shift_end)):
        d -= timedelta(days=1)
    return d.isoformat()


def shift_window(date_str: str, shift_start: str, shift_end: str) -> tuple[datetime, datetime]:
    """The shift's start and end instants (aware UTC) for the attendance date. An overnight shift ends the next day."""
    d = date.fromisoformat(date_str)
    end_day = d + timedelta(days=1) if is_overnight(shift_start, shift_end) else d
    start = datetime.combine(d, time(*parse_hhmm(shift_start)), tzinfo=IST)
    end = datetime.combine(end_day, time(*parse_hhmm(shift_end)), tzinfo=IST)
    return start.astimezone(UTC), end.astimezone(UTC)


 
# Business-rule helpers (R2 - R5)
 
def compute_late_minutes(punch_in: datetime, shift_start_dt: datetime) -> int:
    """R2: late only if strictly more than 600 s after shift start; then whole minutes since shift start."""
    seconds = int((punch_in - shift_start_dt).total_seconds())
    return seconds // 60 if seconds > 600 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600 rounded to 2 decimals half-up (Decimal avoids float and banker's rounding)."""
    seconds = int((punch_out - punch_in).total_seconds())
    return float((Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(punch_out: datetime, shift_end_dt: datetime) -> int:
    """R3: whole minutes after shift end, counted only if at least 30."""
    minutes = int((punch_out - shift_end_dt).total_seconds()) // 60
    return minutes if minutes >= 30 else 0


def compute_half_day(work_hours: Optional[float]) -> bool:
    """R5: after punch-out, true when the rounded work_hours < 4.50."""
    return work_hours is not None and work_hours < 4.5


 
# Serialisation (Mongo document -> API shape)
 
def serialize_employee(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": to_ms(doc.get("created_at")),
    }


def serialize_history_entry(entry: dict) -> dict:
    changes = {}
    for field, change in (entry.get("changes") or {}).items():
        before, after = change.get("from"), change.get("to")
        if field in ("punch_in", "punch_out"):
            before, after = to_ms(before), to_ms(after)
        changes[field] = {"from": before, "to": after}
    return {"at": to_ms(entry["at"]), "by": entry.get("by"), "reason": entry.get("reason"), "changes": changes}


def serialize_attendance(doc: dict) -> dict:
    """Never exposes _id; fills in the defaults for legacy records (missing = 0 / false / [])."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_ms(doc.get("punch_in")),
        "punch_out": to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": [serialize_history_entry(h) for h in doc.get("history") or []],
    }


 
# Calendar and rounding helpers for the analytics endpoints
 
def round_half_up(value, places: int = 2) -> float:
    """R8: half-up rounding. Going through str() uses the shortest repr of a float, so 2.675 stays 2.675."""
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def month_bounds(month: str) -> tuple[date, date]:
    """'2026-09' -> (first day of the month, first day of the next month); 422 for an impossible month."""
    try:
        first = date(int(month[:4]), int(month[5:7]), 1)
    except ValueError:
        raise HTTPException(422, "month must be a real month in YYYY-MM format")
    return first, (first.replace(day=28) + timedelta(days=4)).replace(day=1)


def days_between(start: date, end: date):
    """Every date from start up to (not including) end."""
    for i in range((end - start).days):
        yield start + timedelta(days=i)


def weekend_dates(start: date, end: date) -> list[str]:
    """The Saturdays and Sundays in [start, end) as YYYY-MM-DD strings (R7: weekends are not working days)."""
    return [d.isoformat() for d in days_between(start, end) if d.weekday() >= 5]


def working_days(first: date, next_first: date, joined_on: str) -> int:
    """R7: Monday-Friday in the month, counted from joined_on for someone who joined mid-month."""
    start = max(first, date.fromisoformat(joined_on))
    return sum(1 for d in days_between(start, next_first) if d.weekday() < 5)


 
# Models
 
class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=HHMM)
    shift_end: str = Field(default="18:30", pattern=HHMM)
    joined_on: str = Field(pattern=YMD)

    @model_validator(mode="after")
    def check_values(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        try:
            date.fromisoformat(self.joined_on)  # rejects 2026-02-30 and similar
        except ValueError:
            raise ValueError("joined_on is not a real calendar date")
        return self


class PunchInIn(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str = Field(min_length=1)
    punched_at: Optional[EpochMs] = None


class RegularizeIn(BaseModel):
    # A default of None is not validated, so a field can be omitted, while an explicit null is still a 422.
    # Derived fields (work_hours, late_minutes, ...) are not declared here, so a client cannot send them.
    status: Status = None
    punch_in: EpochMs = None
    punch_out: EpochMs = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)


def parse_date_param(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    try:
        # fromisoformat alone also accepts spellings like 20261102 or 2026-W45-1, so the shape is checked first.
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError(value)
        date.fromisoformat(value)
    except ValueError:
        raise HTTPException(422, f"{name} must be a real date in YYYY-MM-DD format")
    return value


 
# Endpoints
 
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_millis()
    try:
        db.employees.insert_one(doc)  # the unique index on emp_code decides; no check-then-insert
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return serialize_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {"department": department} if department else {}
    total = db.employees.count_documents(q)  # counts the filtered set (R10)
    cursor = (
        db.employees.find(q)
        .sort("emp_code", ASCENDING)
        .skip(page_skip(page, page_size))
        .limit(page_size)
    )
    return {"items": [serialize_employee(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc()
    d = attendance_date(ts, emp["shift_start"], emp["shift_end"])
    shift_start_dt, _ = shift_window(d, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": d,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(ts, shift_start_dt),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)  # unique (emp_code, date) index: exactly one request wins
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return serialize_attendance(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc()

    # The employee's most recent record whose punch_in is at or before the punch-out. Looking at punch_in
    # (not at the date) is what makes this work across midnight for overnight shifts.
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}},
        sort=[("punch_in", DESCENDING)],
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")

    punched_in = rec["punch_in"].replace(microsecond=0)
    if ts <= punched_in or ts - punched_in > timedelta(hours=24):
        raise HTTPException(422, "punched_at must be after punch_in and within 24 hours of it")

    _, shift_end_dt = shift_window(rec["date"], emp["shift_start"], emp["shift_end"])
    work_hours = compute_work_hours(punched_in, ts)
    # The filter "punch_out is null" makes the close atomic: of two simultaneous punch-outs only one matches.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": {
            "punch_out": ts,
            "work_hours": work_hours,
            "overtime_minutes": compute_overtime(ts, shift_end_dt),
            "half_day": compute_half_day(work_hours),
        }},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return serialize_attendance(updated)


HISTORY_FIELDS = ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day")


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):  # `date` = the record's attendance date
    parse_date_param(date, "date")
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if rec is None:
        raise HTTPException(404, "no attendance record for that date")

    # What is stored now, with missing legacy fields read as 0 / false (same defaults as the API output).
    old = {
        "status": rec["status"],
        "punch_in": rec["punch_in"].replace(microsecond=0) if rec.get("punch_in") else None,
        "punch_out": rec["punch_out"].replace(microsecond=0) if rec.get("punch_out") else None,
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": bool(rec.get("half_day", False)),
    }

    # Apply the request to get the final times, validating the rules of the contract.
    new_status = body.status or old["status"]
    if new_status in ("ABSENT", "LEAVE"):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(422, "ABSENT and LEAVE clear the punch times; do not supply punch_in or punch_out")
        new_in = new_out = None
    else:
        new_in = from_ms(body.punch_in) if body.punch_in is not None else old["punch_in"]
        new_out = from_ms(body.punch_out) if body.punch_out is not None else old["punch_out"]
        if new_in is None:
            raise HTTPException(422, "a presence status requires a punch_in")
        if body.punch_in is not None and attendance_date(new_in, emp["shift_start"], emp["shift_end"]) != date:
            raise HTTPException(422, "punch_in must stay on the record's attendance date")
        if new_out is not None and (new_out <= new_in or new_out - new_in > timedelta(hours=24)):
            raise HTTPException(422, "punch_out must be after punch_in and within 24 hours of it")

    # Recompute every derived field from the final times (R2 - R5).
    shift_start_dt, shift_end_dt = shift_window(date, emp["shift_start"], emp["shift_end"])
    new = {"status": new_status, "punch_in": new_in, "punch_out": new_out,
           "work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if new_in is not None:
        new["late_minutes"] = compute_late_minutes(new_in, shift_start_dt)
    if new_out is not None:
        new["work_hours"] = compute_work_hours(new_in, new_out)
        new["overtime_minutes"] = compute_overtime(new_out, shift_end_dt)
        new["half_day"] = compute_half_day(new["work_hours"])

    changes = {f: {"from": old[f], "to": new[f]} for f in HISTORY_FIELDS if old[f] != new[f]}
    if not changes:
        raise HTTPException(422, "the request changes nothing")

    # Optimistic concurrency: the update only applies if the record is still exactly what we read
    # (same status and punch times, and nobody appended to history). Otherwise 409 and nothing is lost.
    n = len(rec.get("history") or [])
    still_unchanged = {
        "_id": rec["_id"],
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        f"history.{n}": {"$exists": False},
    }
    entry = {"at": now_millis(), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    updated = db.attendance_logs.find_one_and_update(
        still_unchanged,
        {"$set": {f: new[f] for f in HISTORY_FIELDS}, "$push": {"history": entry}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "the record was modified by someone else; reload and retry")
    return serialize_attendance(updated)


ATTENDANCE_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]  # date desc, then emp_code asc


def attendance_query(emp_code: Optional[str], date_from: Optional[str], date_to: Optional[str],
                     status: Optional[str]) -> dict:
    """The filter of GET /attendance (also used by the explain endpoint, so both run the very same query)."""
    date_from = parse_date_param(date_from, "date_from")
    date_to = parse_date_param(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must not be after date_to")
    q: dict = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status
    return q


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = attendance_query(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(q)
    cursor = (
        db.attendance_logs.find(q)
        .sort(ATTENDANCE_SORT)  # sorted by the database, not in Python
        .skip(page_skip(page, page_size))
        .limit(page_size)
    )
    return {"items": [serialize_attendance(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


 
# Analytics
 
def present_credit(weekends: list[str]) -> dict:
    """Aggregation expression: how many present days one record is worth (R5 - R7).
    A presence status on Monday-Friday is 1 (0.5 for a half day); anything else is 0."""
    return {"$cond": [
        {"$and": [{"$in": ["$status", list(PRESENCE)]}, {"$eq": [{"$in": ["$date", weekends]}, False]}]},
        {"$cond": [{"$ifNull": ["$half_day", False]}, 0.5, 1]},
        0,
    ]}


def monthly_pipeline(emp_code: str, first: date, next_first: date) -> list[dict]:
    """One employee, one month. Reads the STORED derived values (late_minutes, overtime_minutes, half_day)."""
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first.isoformat(), "$lt": next_first.isoformat()}}},
        {"$group": {
            "_id": None,
            "present_days": {"$sum": present_credit(weekend_dates(first, next_first))},
            "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(pattern=MONTH_RE)):
    first, next_first = month_bounds(month)
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, "employee not found")
    rows = list(db.attendance_logs.aggregate(monthly_pipeline(emp_code, first, next_first)))
    agg = rows[0] if rows else {}
    present = float(agg.get("present_days", 0))
    days = working_days(first, next_first, emp["joined_on"])
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": days,
        "present_days": present,
        "leave_days": int(agg.get("leave_days", 0)),
        "late_count": int(agg.get("late_count", 0)),
        "total_late_minutes": int(agg.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(agg.get("total_overtime_minutes", 0)),
        "attendance_pct": round_half_up(Decimal(str(present)) * 100 / Decimal(days), 2) if days else None,
    }


def hundredths(field: str) -> dict:
    """work_hours as an exact whole number of hundredths (9.12 -> 912), so sums and averages carry no float error."""
    return {"$floor": {"$add": [{"$multiply": [field, 100]}, 0.5]}}


def department_activity_pipeline(first: date, next_first: date, department: Optional[str]) -> list[dict]:
    """Attendance records of the month, totalled per employee first (so the join to `employees` happens once per
    employee, not once per record), then grouped by the employee's department (R6, R9).
    Only employees that count towards the headcount (joined on or before the last day of the month) are included;
    logs whose employee does not exist are dropped by the $unwind. Every total is a sum, so totals of totals are exact."""
    last_day = (next_first - timedelta(days=1)).isoformat()
    employee_filter: dict = {"emp.joined_on": {"$lte": last_day}}
    if department:
        employee_filter["emp.department"] = department
    # "present-status AND has a numeric work_hours" is the population of the average (open records are excluded).
    counts_for_avg = {"$and": [
        {"$in": ["$status", list(PRESENCE)]},
        {"$ne": [{"$ifNull": ["$work_hours", None]}, None]},   # a missing field and null both become null
    ]}
    totals = ("present_days", "hours_records", "hours_in_hundredths", "late_count", "total_late_minutes",
              "leave_count", "on_duty_count")
    return [
        {"$match": {"date": {"$gte": first.isoformat(), "$lt": next_first.isoformat()}}},
        # Step 1: one row per employee (about as many rows as employees, instead of one per record).
        {"$group": {
            "_id": "$emp_code",
            "present_days": {"$sum": present_credit(weekend_dates(first, next_first))},
            "hours_records": {"$sum": {"$cond": [counts_for_avg, 1, 0]}},
            "hours_in_hundredths": {"$sum": {"$cond": [counts_for_avg, hundredths("$work_hours"), 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "leave_count": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
            "on_duty_count": {"$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}},
        }},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},
        {"$match": employee_filter},
        # Step 2: add the per-employee totals up into their department.
        {"$group": {"_id": "$emp.department", **{name: {"$sum": f"${name}"} for name in totals}}},
    ]


def department_headcount_pipeline(next_first: date, department: Optional[str]) -> list[dict]:
    """R9: employees who joined on or before the last day of the month, with or without any logs."""
    match: dict = {"joined_on": {"$lte": (next_first - timedelta(days=1)).isoformat()}}
    if department:
        match["department"] = department
    return [
        {"$match": match},
        {"$group": {"_id": "$department", "headcount": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]


@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(pattern=MONTH_RE), department: Optional[str] = None):
    first, next_first = month_bounds(month)
    activity = {r["_id"]: r for r in db.attendance_logs.aggregate(department_activity_pipeline(first, next_first, department))}
    items = []
    # The headcount query drives the output: a department with no logs still shows up, one with headcount 0 does not.
    for h in db.employees.aggregate(department_headcount_pipeline(next_first, department)):
        a = activity.get(h["_id"], {})
        n = int(a.get("hours_records", 0))
        avg = round_half_up(Decimal(int(a["hours_in_hundredths"])) / Decimal(100 * n), 2) if n else None
        items.append({
            "department": h["_id"],
            "headcount": h["headcount"],
            "present_days": float(a.get("present_days", 0)),
            "avg_work_hours": avg,
            "late_count": int(a.get("late_count", 0)),
            "total_late_minutes": int(a.get("total_late_minutes", 0)),
            "leave_count": int(a.get("leave_count", 0)),
            "on_duty_count": int(a.get("on_duty_count", 0)),
        })
    return {"month": month, "items": items}


def late_leaderboard_pipeline(first: date, next_first: date, limit: int, department: Optional[str]) -> list[dict]:
    """Employees ranked by total late minutes in the month, highest first (only totals > 0).
    $rank is standard competition ranking: ties share a rank and the next rank is skipped (1, 2, 2, 4).
    The limit is applied to the RANK, after ranking, so a tie at the cutoff returns more than `limit` rows."""
    pipeline: list[dict] = [
        {"$match": {"date": {"$gte": first.isoformat(), "$lt": next_first.isoformat()}}},
        {"$group": {
            "_id": "$emp_code",
            "total_late_minutes": {"$sum": {"$ifNull": ["$late_minutes", 0]}},
            "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
        }},
        {"$match": {"total_late_minutes": {"$gt": 0}}},
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},  # a log whose employee does not exist is dropped here
    ]
    if department:
        # Filter BEFORE ranking: with a department the ranking happens inside that department only.
        pipeline.append({"$match": {"emp.department": department}})
    pipeline += [
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},  # tied rows are ordered by emp_code
        {"$project": {
            "_id": 0, "rank": 1, "emp_code": "$_id", "name": "$emp.name", "department": "$emp.department",
            "total_late_minutes": 1, "late_count": 1,
        }},
    ]
    return pipeline


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(pattern=MONTH_RE),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    first, next_first = month_bounds(month)
    rows = db.attendance_logs.aggregate(late_leaderboard_pipeline(first, next_first, limit, department))
    items = [{
        "rank": int(r["rank"]),
        "emp_code": r["emp_code"],
        "name": r["name"],
        "department": r["department"],
        "total_late_minutes": int(r["total_late_minutes"]),
        "late_count": int(r["late_count"]),
    } for r in rows]
    return {"month": month, "items": items}


MAX_TREND_DAYS = 92
RATE_UNITS = 10_000  # rates are carried as whole numbers of 1/10000, which makes 4-decimal half-up rounding exact


def trend_pipeline(department: str, codes: list[str], first: date, last: date) -> list[dict]:
    """One row per calendar day from `first` to `last`, built entirely inside MongoDB.
    employees (one document for the department) -> calendar days via $range/$unwind -> per-day attendance via
    $lookup -> rates -> 7-day moving average via a window function."""
    days = (last - first).days + 1
    start = datetime.combine(first, time(0), tzinfo=UTC)
    return [
        # One document carrying the joined_on of every employee of the department (this is also the 404 guard's data).
        {"$match": {"department": department}},
        {"$group": {"_id": None, "joined": {"$push": "$joined_on"}}},
        # The calendar: day 0 .. day N-1 as dates, then one document per day (weekends and empty days included).
        {"$project": {"_id": 0, "joined": 1, "day": {"$map": {
            "input": {"$range": [0, days]}, "as": "i",
            "in": {"$add": [start, {"$multiply": ["$$i", 86400000]}]}}}}},
        {"$unwind": "$day"},
        {"$addFields": {
            "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$day"}},
            "is_working_day": {"$lte": [{"$isoDayOfWeek": "$day"}, 5]},
        }},
        # That day's attendance of the department's employees (equality on date uses the (date, emp_code) index).
        {"$lookup": {
            "from": "attendance_logs", "localField": "date", "foreignField": "date",
            "pipeline": [
                {"$match": {"emp_code": {"$in": codes}}},
                {"$group": {
                    "_id": None,
                    "present_count": {"$sum": present_credit([])},  # any weekday or weekend
                    "late_count": {"$sum": {"$cond": [{"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}, 1, 0]}},
                }},
            ],
            "as": "stats",
        }},
        {"$addFields": {
            # R9: employees who joined on or before that day (YYYY-MM-DD strings compare like dates).
            "headcount": {"$size": {"$filter": {"input": "$joined", "as": "j", "cond": {"$lte": ["$$j", "$date"]}}}},
            "present_count": {"$ifNull": [{"$arrayElemAt": ["$stats.present_count", 0]}, 0]},
            "late_count": {"$ifNull": [{"$arrayElemAt": ["$stats.late_count", 0]}, 0]},
        }},
        # attendance_rate in units of 1/10000, half-up: floor(present / headcount * 10000 + 0.5).
        # Only on working days with a headcount; a working day with no records is 0.
        {"$addFields": {"rate_units": {"$cond": [
            {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
            {"$floor": {"$add": [
                {"$divide": [{"$multiply": ["$present_count", 2 * RATE_UNITS]}, {"$multiply": ["$headcount", 2]}]},
                0.5]}},
            None,
        ]}}},
        # Mean of the non-null rates in the current row and up to 6 rows before it; the window never reaches before
        # the first requested day, so the first rows use a shorter window. $avg ignores nulls (all null -> null).
        {"$setWindowFields": {
            "sortBy": {"date": 1},
            "output": {"avg_units": {"$avg": "$rate_units", "window": {"documents": [-6, 0]}}},
        }},
        {"$sort": {"date": 1}},
        {"$project": {
            "_id": 0, "date": 1, "is_working_day": 1, "headcount": 1, "present_count": 1, "late_count": 1,
            "attendance_rate": {"$cond": [{"$eq": ["$rate_units", None]}, None, {"$divide": ["$rate_units", RATE_UNITS]}]},
            "moving_avg_7d": {"$cond": [
                {"$eq": ["$avg_units", None]}, None,
                {"$divide": [{"$floor": {"$add": ["$avg_units", 0.5]}}, RATE_UNITS]}]},
        }},
    ]


def prepare_trend(department: str, date_from: str, date_to: str) -> list[dict]:
    """Validate the request and build the trend pipeline (shared by the endpoint and the explain endpoint)."""
    first = date.fromisoformat(parse_date_param(date_from, "from"))
    last = date.fromisoformat(parse_date_param(date_to, "to"))
    if last < first:
        raise HTTPException(422, "to must not be before from")
    if (last - first).days + 1 > MAX_TREND_DAYS:
        raise HTTPException(422, f"the range must not exceed {MAX_TREND_DAYS} days")
    codes = db.employees.distinct("emp_code", {"department": department})
    if not codes:
        raise HTTPException(404, "unknown department")
    return trend_pipeline(department, codes, first, last)


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: str = Query(alias="from", pattern=YMD),
    date_to: str = Query(alias="to", pattern=YMD),
):
    rows = db.employees.aggregate(prepare_trend(department, date_from, date_to))
    return {"department": department, "items": [{
        "date": r["date"],
        "is_working_day": bool(r["is_working_day"]),
        "headcount": int(r["headcount"]),
        "present_count": float(r["present_count"]),
        "late_count": int(r["late_count"]),
        "attendance_rate": r["attendance_rate"],
        "moving_avg_7d": r["moving_avg_7d"],
    } for r in rows]}


 
# Admin: explain
 
ExplainTarget = Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard",
                        "department_trend"]


def require(**params) -> None:
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise HTTPException(422, "missing required parameter(s): " + ", ".join(missing))


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: ExplainTarget,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=MONTH_RE),
    department: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    from_: Optional[str] = Query(None, alias="from", pattern=YMD),
    to: Optional[str] = Query(None, alias="to", pattern=YMD),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """MongoDB's explain (executionStats) of the main query/pipeline of an endpoint, built by the same functions
    that the endpoint itself uses, so the plan shown is the plan that runs."""
    if endpoint == "attendance_list":
        collection = "attendance_logs"
        command = {
            "find": collection,
            "filter": attendance_query(emp_code, date_from, date_to, status),
            "sort": SON(ATTENDANCE_SORT),
            "skip": page_skip(page, page_size),
            "limit": page_size,
        }
    else:
        if endpoint == "department_trend":
            require(department=department, **{"from": from_, "to": to})
            collection, pipeline = "employees", prepare_trend(department, from_, to)
        else:
            if endpoint == "employee_monthly":
                require(emp_code=emp_code, month=month)
            else:
                require(month=month)
            first, next_first = month_bounds(month)
            collection = "attendance_logs"
            if endpoint == "employee_monthly":
                pipeline = monthly_pipeline(emp_code, first, next_first)
            elif endpoint == "department_summary":
                pipeline = department_activity_pipeline(first, next_first, department)
            else:
                pipeline = late_leaderboard_pipeline(first, next_first, limit, department)
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}
    raw = db.command({"explain": command, "verbosity": "executionStats"})
    # The explain document can hold BSON types (Int64, Timestamp, ...): convert to plain JSON.
    return {"endpoint": endpoint, "collection": collection, "explain": json.loads(json_util.dumps(raw))}