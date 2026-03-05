


from datetime import datetime
import os
from typing import Optional, Any, List

from dotenv import load_dotenv
load_dotenv()

import anthropic
import mariadb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str     # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class EmployeeCreate(BaseModel):
    name: str
    position: Optional[str] = None
    department: Optional[str] = None
    contact: Optional[str] = None


class PreventiveLogCreate(BaseModel):
    device_id: str
    device_type: str                        # AGV | OHT
    detected_at: Optional[str] = None      # YYYY-MM-DD HH:MM:SS
    description: Optional[str] = None
    predicted_state_before: Optional[int] = None
    predicted_state_after: Optional[int] = None
    result: Optional[str] = None           # 성공 | 실패
    technician: Optional[str] = None
    category: Optional[str] = None
    corrective_id: Optional[int] = None
    employee_id: Optional[int] = None


class CorrectiveLogCreate(BaseModel):
    device_id: str
    device_type: str                        # AGV | OHT
    description: Optional[str] = None
    technician: Optional[str] = None
    category: Optional[str] = None
    before_state: Optional[int] = None
    after_state: Optional[int] = None
    actioned_at: Optional[str] = None      # YYYY-MM-DD HH:MM:SS
    completed_at: Optional[str] = None     # NULL이면 진행중
    employee_id: Optional[int] = None

app = FastAPI(title="Device State Count API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    return mariadb.connect(
        user=os.environ.get("DB_USER", "appuser"),
        password=os.environ.get("DB_PASSWORD", ""),
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", "3306")),
        database=os.environ.get("DB_NAME", "appdb"),
    )


def _validate_inputs(date: str, device_type: str):
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="date는 YYYY-MM-DD 형식이어야 합니다.")

    dtype = device_type.lower()
    if dtype not in ("agv", "oht"):
        raise HTTPException(status_code=400, detail="device_type은 agv 또는 oht 이어야 합니다.")
    return dtype


def _sensor_table(dtype: str) -> str:
    return "agv_sensor_data" if dtype == "agv" else "oht_sensor_data"


ALARM_SENSOR_KEYS = [
    "PM10",
    "PM2_5",
    "CT1",
    "CT2",
    "CT3",
    "CT4",
    "NTC",
]

ALARM_ADVICE_RULES = [
    ("PM10", 80, "먼지농도", "점검"),
    ("PM2_5", 35, "먼지농도", "점검"),
    ("CT2", 90, "CT2", "점검"),
    ("CT1", 90, "CT1", "점검"),
    ("NTC", 40, "온도", "점검"),
    ("humidity", 70, "습도", "체크"),
]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_datetime(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _status_label(state: Optional[int]) -> str:
    if state is None:
        return "정상"
    try:
        state = int(state)
    except (TypeError, ValueError):
        return "정상"
    if state >= 2:
        return "고장"
    if state == 1:
        return "경고"
    return "정상"


def _build_sensor_advice(row: dict[str, Any]) -> tuple[str, str, float | None]:
    for key, threshold, label, verb in ALARM_ADVICE_RULES:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > threshold:
            advice = f"{key}({numeric:.1f}) → {label}{verb}하세요."
            return advice, label, numeric
    return "센서 이상값을 확인하세요.", "기타", None


def _fetch_recent_rows(cursor, table: str, log_type: str, limit: int):
    cursor.execute(
        f"""
        SELECT device_id, datetime, state, '{log_type}' AS type
        FROM {table}
        ORDER BY datetime DESC
        LIMIT ?
        """,
        (limit,),
    )
    return cursor.fetchall() or []


def _fetch_alarm_events(cursor, table: str, log_type: str, limit: int, date: str | None):
    """
    device_id + 날짜 단위로 alarm 이벤트를 그룹핑한다.
    지속 시간(duration_minutes) = MAX(datetime) - MIN(datetime).
    20분 미만 이벤트는 제외한다.
    """
    if date:
        date_filter = "AND DATE(datetime) = ?"
        params: tuple = (date, limit)
    else:
        date_filter = ""
        params = (limit,)

    cursor.execute(
        f"""
        SELECT
            device_id,
            MIN(datetime)                                          AS alarm_start,
            MAX(datetime)                                          AS alarm_end,
            TIMESTAMPDIFF(MINUTE, MIN(datetime), MAX(datetime))   AS duration_minutes,
            MAX(state)                                             AS state,
            AVG(PM10)                                              AS PM10,
            AVG(PM2_5)                                             AS PM2_5,
            AVG(CT1)                                               AS CT1,
            AVG(CT2)                                               AS CT2,
            AVG(CT3)                                               AS CT3,
            AVG(CT4)                                               AS CT4,
            AVG(NTC)                                               AS NTC,
            '{log_type}'                                           AS type
        FROM {table}
        WHERE state >= 2 {date_filter}
        GROUP BY device_id, DATE(datetime)
        HAVING duration_minutes >= 20
        ORDER BY alarm_start DESC
        LIMIT ?
        """,
        params,
    )
    return cursor.fetchall() or []


@app.get("/logs/recent")
def get_recent_logs(limit: int = Query(default=5, ge=1, le=50)):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    per_type_limit = max(1, limit)
    rows = []
    try:
        rows.extend(_fetch_recent_rows(cursor, "agv_sensor_data", "AGV", per_type_limit))
        rows.extend(_fetch_recent_rows(cursor, "oht_sensor_data", "OHT", per_type_limit))
    except Exception:
        rows = []
    finally:
        cursor.close()
        conn.close()

    logs = []
    for idx, row in enumerate(rows):
        dt = row.get("datetime")
   
        if dt is not None and hasattr(dt, "strftime"):
            dt_value = dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            dt_value = str(dt) if dt is not None else ""
        device = row.get("device_id") or ""
        log_type = row.get("type") or "AGV"
        logs.append({
            "id": f"{log_type}-{device}-{idx}",
            "datetime": dt_value,
            "equipment": device,
            "type": log_type,
            "status": _status_label(row.get("state")),
        })

    return {"logs": logs}


@app.get("/alarms/recent")
def get_recent_alarms(
    limit: int = Query(default=20, ge=1, le=100),
    equipment_type: str | None = Query(default=None),
    date: str | None = Query(default=None),
):
    """
    최근 알람 이벤트 목록.
    - equipment_type: AGV | OHT (미지정 시 둘 다)
    - date: YYYY-MM-DD 형식 (미지정 시 전체)
    - 지속시간 20분 이상인 이벤트만 반환
    """
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="date는 YYYY-MM-DD 형식이어야 합니다.")

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    rows = []
    dtype = equipment_type.upper() if equipment_type else None
    try:
        if not dtype or dtype == "AGV":
            rows.extend(_fetch_alarm_events(cursor, "agv_sensor_data", "AGV", limit, date))
        if not dtype or dtype == "OHT":
            rows.extend(_fetch_alarm_events(cursor, "oht_sensor_data", "OHT", limit, date))
    except Exception:
        rows = []
    finally:
        cursor.close()
        conn.close()

    rows = sorted(rows, key=lambda r: r.get("alarm_start") or datetime.min, reverse=True)
    normalized = []
    for idx, row in enumerate(rows[:limit]):
        advice, failure_type, _ = _build_sensor_advice(row)
        row_type = (row.get("type") or "AGV").upper()
        state = int(row.get("state") or 0)
        level = "위험" if state >= 3 else "경고"
        duration = int(row.get("duration_minutes") or 0)
        normalized.append({
            "id": f"{row_type}-{row.get('device_id', '')}-{idx}",
            "equipment": row.get("device_id") or "",
            "type": row_type,
            "datetime": _format_datetime(row.get("alarm_start")),
            "alarmEnd": _format_datetime(row.get("alarm_end")),
            "state": state,
            "durationMinutes": duration,
            "level": level,
            "message": advice,
            "failureType": failure_type,
            "actionAdvice": advice,
            "actionStatus": "대기",
            "status": _status_label(state),
            "sensors": {key: _safe_float(row.get(key)) for key in ALARM_SENSOR_KEYS},
        })
    return {"alarms": normalized}


@app.get("/debug/alarm-check")
def debug_alarm_check():
    """DB에 실제 데이터가 있는지 진단"""
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    result = {}
    for table in ("agv_sensor_data", "oht_sensor_data"):
        try:
            cursor.execute(f"SELECT COUNT(*) AS total FROM {table}")
            total = cursor.fetchone()["total"]
            cursor.execute(f"SELECT COUNT(*) AS cnt FROM {table} WHERE state >= 2")
            state_cnt = cursor.fetchone()["cnt"]
            cursor.execute(f"SELECT MIN(datetime) AS oldest, MAX(datetime) AS newest FROM {table}")
            times = cursor.fetchone()
            result[table] = {
                "total_rows": total,
                "state_gte2": state_cnt,
                "oldest": _format_datetime(times["oldest"]),
                "newest": _format_datetime(times["newest"]),
            }
        except Exception as e:
            result[table] = {"error": str(e)}
    cursor.close()
    conn.close()
    return result


@app.get("/")
def root():
    return {"message": "Device state count aggregation API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/devices")
def get_devices():
    """
    장비 목록 반환.
    agv_sensor_data(AGV), oht_sensor_data(OHT) 테이블에서 고유 device_id를 조회한다.
    devices 테이블이 있으면 우선 사용한다.
    """
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # devices 테이블이 있는지 먼저 시도
    try:
        cursor.execute("SELECT device_id, device_name, manufacturer FROM devices ORDER BY device_id")
        rows = cursor.fetchall()
        if rows:
            cursor.close()
            conn.close()
            return {"devices": rows}
    except Exception:
        pass  # devices 테이블 없으면 sensor 테이블에서 파생

    devices = []
    try:
        cursor.execute("SELECT DISTINCT device_id FROM agv_sensor_data ORDER BY device_id")
        for row in cursor.fetchall():
            did = row["device_id"]
            devices.append({
                "device_id": did,
                "device_name": str(did),
                "manufacturer": "AGV",
            })
    except Exception:
        pass

    try:
        cursor.execute("SELECT DISTINCT device_id FROM oht_sensor_data ORDER BY device_id")
        for row in cursor.fetchall():
            did = row["device_id"]
            devices.append({
                "device_id": did,
                "device_name": str(did),
                "manufacturer": "OHT",
            })
    except Exception:
        pass

    cursor.close()
    conn.close()
    return {"devices": devices}


def _fetch_history(cursor, table: str, device_id: str, limit: int):
    cursor.execute(
        f"""
        SELECT datetime, CT1, CT2, CT3, CT4,
               ir_temp_max, PM10, PM2_5, NTC, state
        FROM {table}
        WHERE device_id = ?
        ORDER BY datetime DESC
        LIMIT ?
        """,
        (device_id, limit),
    )
    rows = cursor.fetchall() or []
    result = []
    for row in rows:
        dt = row.get("datetime")
        if dt is not None and hasattr(dt, "strftime"):
            dt = dt.strftime("%Y-%m-%d %H:%M:%S")
        result.append({
            "datetime":    str(dt) if dt is not None else "",
            "CT1":         float(row.get("CT1") or 0),
            "CT2":         float(row.get("CT2") or 0),
            "CT3":         float(row.get("CT3") or 0),
            "CT4":         float(row.get("CT4") or 0),
            "ir_temp_max": float(row.get("ir_temp_max") or 0),
            "PM10":        float(row.get("PM10") or 0),
            "PM2_5":       float(row.get("PM2_5") or 0),
            "NTC":         float(row.get("NTC") or 0),
            "state":       int(row.get("state") or 0),
        })
    return result


@app.get("/agv/{device_id}/history")
def get_device_history(device_id: str, limit: int = Query(default=100, ge=1, le=10000)):
    """
    장비 센서 히스토리.
    AGV(agv_sensor_data) → OHT(oht_sensor_data) 순서로 조회한다.
    """
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    history = []
    for table in ("agv_sensor_data", "oht_sensor_data"):
        try:
            history = _fetch_history(cursor, table, device_id, limit)
            if history:
                break
        except Exception:
            pass

    cursor.close()
    conn.close()
    return {"device_id": device_id, "history": history}


def _fetch_stats(cursor, table: str, device_id: str):
    cursor.execute(
        f"""
        SELECT
            COUNT(*)                                    AS total_records,
            AVG(CT2)                                    AS avg_ct2,
            AVG(ir_temp_max)                            AS avg_temp,
            AVG(PM10)                                   AS avg_pm10,
            SUM(CASE WHEN state = 0 THEN 1 ELSE 0 END) AS state_0,
            SUM(CASE WHEN state = 1 THEN 1 ELSE 0 END) AS state_1,
            SUM(CASE WHEN state = 2 THEN 1 ELSE 0 END) AS state_2,
            SUM(CASE WHEN state = 3 THEN 1 ELSE 0 END) AS state_3
        FROM {table}
        WHERE device_id = ?
        """,
        (device_id,),
    )
    return cursor.fetchone()


@app.get("/agv/{device_id}/stats")
def get_device_stats(device_id: str):
    """
    장비 통계 (전체 기간).
    AGV(agv_sensor_data) → OHT(oht_sensor_data) 순서로 조회한다.
    """
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    row = None
    for table in ("agv_sensor_data", "oht_sensor_data"):
        try:
            row = _fetch_stats(cursor, table, device_id)
            if row and int(row.get("total_records") or 0) > 0:
                break
        except Exception:
            pass

    cursor.close()
    conn.close()

    if not row or int(row.get("total_records") or 0) == 0:
        raise HTTPException(status_code=404, detail=f"장비 '{device_id}'를 찾을 수 없습니다.")

    return {
        "device_id": device_id,
        "stats": {
            "total_records": int(row.get("total_records") or 0),
            "avg_ct2":       round(float(row.get("avg_ct2") or 0), 3),
            "avg_temp":      round(float(row.get("avg_temp") or 0), 3),
            "avg_pm10":      round(float(row.get("avg_pm10") or 0), 3),
            "state_0":       int(row.get("state_0") or 0),
            "state_1":       int(row.get("state_1") or 0),
            "state_2":       int(row.get("state_2") or 0),
            "state_3":       int(row.get("state_3") or 0),
        },
    }


@app.get("/preventive-logs")
def get_preventive_logs(
    limit: int = Query(default=50, ge=1, le=200),
    device_id: Optional[str] = Query(default=None),
    device_type: Optional[str] = Query(default=None),
):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    conditions = []
    params: list = []
    if device_id:
        conditions.append("device_id = ?")
        params.append(device_id)
    if device_type:
        conditions.append("device_type = ?")
        params.append(device_type.upper())
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    cursor.execute(f"""
        SELECT id, device_id, device_type, detected_at, description,
               predicted_state_before, predicted_state_after,
               result, technician, category, corrective_id, created_at
        FROM preventive_logs
        {where}
        ORDER BY created_at DESC
        LIMIT ?
    """, params)
    rows = cursor.fetchall() or []
    cursor.close()
    conn.close()
    for row in rows:
        row["detected_at"] = _format_datetime(row.get("detected_at"))
        row["created_at"]  = _format_datetime(row.get("created_at"))
    return {"preventive_logs": rows, "total": len(rows)}


@app.get("/employees")
def get_employees():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name, position, department, contact FROM employees ORDER BY name ASC")
    rows = cursor.fetchall() or []
    cursor.close()
    conn.close()
    return {"employees": rows}


@app.post("/employees")
def create_employee(body: EmployeeCreate):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO employees (name, position, department, contact) VALUES (?, ?, ?, ?)",
        (body.name, body.position, body.department, body.contact),
    )
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return {"id": new_id, "message": "직원이 등록되었습니다."}


@app.post("/preventive-logs")
def create_preventive_log(body: PreventiveLogCreate):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # employee_id로 담당자 정보 자동 채움
    technician = body.technician
    category = body.category
    if body.employee_id:
        cursor.execute("SELECT name, department FROM employees WHERE id = ?", (body.employee_id,))
        emp = cursor.fetchone()
        if emp:
            technician = emp["name"]
            category = emp["department"]

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO preventive_logs
            (device_id, device_type, detected_at, description,
             predicted_state_before, predicted_state_after,
             result, technician, category, corrective_id, employee_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        body.device_id,
        body.device_type.upper(),
        body.detected_at or None,
        body.description,
        body.predicted_state_before,
        body.predicted_state_after,
        body.result,
        technician,
        category,
        body.corrective_id,
        body.employee_id,
    ))
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return {"id": new_id, "message": "예방 이력이 등록되었습니다."}


@app.get("/corrective-logs")
def get_corrective_logs(
    limit: int = Query(default=50, ge=1, le=200),
    device_id: Optional[str] = Query(default=None),
    device_type: Optional[str] = Query(default=None),
):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    conditions = []
    params: list = []
    if device_id:
        conditions.append("device_id = ?")
        params.append(device_id)
    if device_type:
        conditions.append("device_type = ?")
        params.append(device_type.upper())
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)
    cursor.execute(f"""
        SELECT id, device_id, device_type, description,
               technician, category, before_state, after_state,
               actioned_at, completed_at, created_at
        FROM corrective_logs
        {where}
        ORDER BY created_at DESC
        LIMIT ?
    """, params)
    rows = cursor.fetchall() or []
    cursor.close()
    conn.close()
    for row in rows:
        row["actioned_at"]  = _format_datetime(row.get("actioned_at"))
        row["completed_at"] = _format_datetime(row.get("completed_at"))
        row["created_at"]   = _format_datetime(row.get("created_at"))
    return {"corrective_logs": rows, "total": len(rows)}


@app.post("/corrective-logs")
def create_corrective_log(body: CorrectiveLogCreate):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    # employee_id로 담당자 정보 자동 채움
    technician = body.technician
    category = body.category
    if body.employee_id:
        cursor.execute("SELECT name, department FROM employees WHERE id = ?", (body.employee_id,))
        emp = cursor.fetchone()
        if emp:
            technician = emp["name"]
            category = emp["department"]

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO corrective_logs
            (device_id, device_type, description, technician,
             category, before_state, after_state, actioned_at, completed_at, employee_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        body.device_id,
        body.device_type.upper(),
        body.description,
        technician,
        category,
        body.before_state,
        body.after_state,
        body.actioned_at or None,
        body.completed_at or None,
        body.employee_id,
    ))
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return {"id": new_id, "message": "조치 이력이 등록되었습니다."}


# ──────────────────────────────────────────────
# 금일 공정 변경점 (4M)
# ──────────────────────────────────────────────

VALID_CHANGE_TYPES = {"작업자변경", "자재변경", "방법변경", "설비기계변경"}


class FourMChangeCreate(BaseModel):
    change_type: str   # 작업자변경 | 자재변경 | 방법변경 | 설비기계변경
    content: str


@app.get("/four-m-changes")
def get_four_m_changes(date: Optional[str] = Query(default=None)):
    """금일(또는 지정일) 공정 변경점 목록. date 미지정 시 오늘 전체."""
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    if date:
        cursor.execute(
            "SELECT id, change_type, content, created_at FROM four_m_changes WHERE DATE(created_at) = ? ORDER BY created_at DESC",
            (date,),
        )
    else:
        cursor.execute(
            "SELECT id, change_type, content, created_at FROM four_m_changes WHERE DATE(created_at) = CURDATE() ORDER BY created_at DESC"
        )
    rows = cursor.fetchall() or []
    cursor.close()
    conn.close()
    for row in rows:
        row["created_at"] = _format_datetime(row.get("created_at"))
    return {"four_m_changes": rows, "total": len(rows)}


@app.post("/four-m-changes")
def create_four_m_change(body: FourMChangeCreate):
    if body.change_type not in VALID_CHANGE_TYPES:
        raise HTTPException(status_code=400, detail=f"change_type은 {VALID_CHANGE_TYPES} 중 하나여야 합니다.")
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="content는 비워둘 수 없습니다.")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO four_m_changes (change_type, content) VALUES (?, ?)",
        (body.change_type, body.content.strip()),
    )
    conn.commit()
    new_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return {"id": new_id, "message": "공정 변경점이 등록되었습니다."}


@app.delete("/four-m-changes/{item_id}")
def delete_four_m_change(item_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM four_m_changes WHERE id = ?", (item_id,))
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail="해당 항목을 찾을 수 없습니다.")
    return {"message": "삭제되었습니다."}


@app.get("/reports/daily")
def get_daily_state_counts(date: str, device_type: str = "agv", include_normal: bool = True):
    """
    일자/장비타입 기준 state 건수 집계.
    프론트 호환성을 위해 기존 경로(/reports/daily)를 유지한다.
    """
    dtype = _validate_inputs(date, device_type)
    table = _sensor_table(dtype)
    state_where = "" if include_normal else " AND state != 0"

    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        f"""
        SELECT
            device_id,
            COUNT(*) as total_records,
            SUM(CASE WHEN state = 0 THEN 1 ELSE 0 END) as state_0,
            SUM(CASE WHEN state = 1 THEN 1 ELSE 0 END) as state_1,
            SUM(CASE WHEN state = 2 THEN 1 ELSE 0 END) as state_2,
            SUM(CASE WHEN state = 3 THEN 1 ELSE 0 END) as state_3
        FROM {table}
        WHERE DATE(datetime) = ? {state_where}
        GROUP BY device_id
        ORDER BY device_id
    """,
        (date,),
    )

    by_device = cursor.fetchall() or []
    cursor.close()
    conn.close()

    for row in by_device:
        row["total_records"] = int(row.get("total_records") or 0)
        row["state_0"] = int(row.get("state_0") or 0)
        row["state_1"] = int(row.get("state_1") or 0)
        row["state_2"] = int(row.get("state_2") or 0)
        row["state_3"] = int(row.get("state_3") or 0)

    summary = {
        "total_records": sum(r["total_records"] for r in by_device),
        "state_0": sum(r["state_0"] for r in by_device),
        "state_1": sum(r["state_1"] for r in by_device),
        "state_2": sum(r["state_2"] for r in by_device),
        "state_3": sum(r["state_3"] for r in by_device),
    }

    return {
        "date": date,
        "device_type": dtype,
        "include_normal": include_normal,
        "summary": summary,
        "by_device": by_device,
    }


# ──────────────────────────────────────────────
# 조치 가이드 데이터
# ──────────────────────────────────────────────

AGV_GUIDE = {
    "PM": {
        "label": "미세먼지 (PM)",
        "state2": [
            {"action": "하우징 가스켓 보강", "detail": "하우징 커버 고무 가스켓 노후 확인 후 실링재 보강", "before": 85, "after": 45, "duration": "10분"},
            {"action": "에어건/접점부활제 사용", "detail": "PCB와 커넥터 사이 전도성 분진 정밀 클리닝", "before": 85, "after": 40, "duration": "15분"},
            {"action": "냉각 시스템 점검", "detail": "내부 환기팬 필터 교체 및 팬 날개 먼지 제거", "before": 85, "after": 48, "duration": "10분"},
        ],
        "state3": [
            {"action": "탄화 흔적 전수 조사", "detail": "가동 중단 후 케이블 피복·PCB 소자 탄화 여부 육안 확인", "before": None, "after": None, "duration": "5분"},
            {"action": "기구부 마찰 제거", "detail": "구동부(벨트/기어) 마찰 분진 점검 및 산업용 윤활제 재도포", "before": 85, "after": 50, "duration": "15분"},
            {"action": "비가역 부품 교체", "detail": "탄화된 PCB 또는 전원부 모듈 신품 교체", "before": 85, "after": 30, "duration": "40분"},
        ],
        "target": {"PM10": {"before": 85, "after": 40}, "PM2_5": {"before": 46.13, "after": 27.13}},
    },
    "CT": {
        "label": "전류 (CT)",
        "state2": [
            {"action": "이물질 제거 (CT3·CT4)", "detail": "구동 바퀴·기어 사이 비닐·끈·먼지 물리적 제거", "before": 79.87, "after": 49.91, "duration": "15분"},
            {"action": "윤활(구리스) 보충", "detail": "기어박스·베어링 산업용 구리스 주입으로 회전 저항 감소", "before": 55, "after": 45, "duration": "10분"},
            {"action": "단자 재체결 (CT1)", "detail": "진동으로 느슨해진 입력 배선 터미널 나사 재조임", "before": 32.26, "after": 1.86, "duration": "10분"},
        ],
        "state3": [
            {"action": "도통 시험 (CT2)", "detail": "전원 차단 후 멀티미터로 단락 여부 점검 및 탄화 소자 육안 확인", "before": 168.57, "after": 72.10, "duration": "20분"},
            {"action": "모터 절연 점검", "detail": "절연 저항계로 구동 모터 내부 권선 절연 파괴 여부 확인", "before": None, "after": None, "duration": "15분"},
            {"action": "비가역 부품 교체 (모터·PCB)", "detail": "단락·절연파괴 확인 시 모터 및 PCB 모듈 즉시 신품 교체", "before": 168.57, "after": 72.10, "duration": "50분"},
        ],
        "target": {"CT1": {"before": 32.26, "after": 1.86}, "CT2": {"before": 168.57, "after": 72.10}, "CT3": {"before": 79.87, "after": 49.91}, "CT4": {"before": 52.94, "after": 19.65}},
    },
    "ir_temp_max": {
        "label": "소자 온도 (ir_temp_max)",
        "state2": [
            {"action": "Hot Spot 특정", "detail": "열화상 이미지로 발열 소자(IC·MOSFET·커넥터 등) 정확히 특정", "before": None, "after": None, "duration": "5분"},
            {"action": "접촉 불량 조치", "detail": "커넥터·터미널 블록 나사 풀림 확인 후 재조임으로 국소 발열 해소", "before": 102.15, "after": 65, "duration": "10분"},
            {"action": "Thermal Grease 재도포", "detail": "경화된 구리스 제거 후 신규 도포로 열 전도 효율 복원", "before": 102.15, "after": 47.5, "duration": "25분"},
        ],
        "state3": [
            {"action": "즉시 전원 차단", "detail": "인접 소자 열 전이 방지를 위해 즉시 비상 정지 및 전원 분리", "before": 110, "after": None, "duration": "즉각적"},
            {"action": "모듈 교체", "detail": "물리적 변형·탄화 의심 PCB 보드·DC-DC 컨버터 통째 교체", "before": 110, "after": 35, "duration": "50분"},
        ],
        "target": {"ir_temp_max": {"before": 102.15, "after": 50.4}},
    },
    "NTC": {
        "label": "시스템 온도 (NTC)",
        "state2": [
            {"action": "냉각 팬 확인", "detail": "하우징 내부 팬 회전 속도·소음 확인, 베어링 노후 시 교체", "before": 65, "after": 50, "duration": "15분"},
            {"action": "공기 순환 필터 청소", "detail": "외부 흡입구 필터 먼지 막힘 확인 후 세척·교체", "before": 65, "after": 60, "duration": "10분"},
            {"action": "배선 정리 (Airflow 최적화)", "detail": "내부 배선 뭉치가 팬 기류를 막지 않도록 정리", "before": 65, "after": 60, "duration": "10분"},
        ],
        "state3": [
            {"action": "운행 중단·자연 냉각", "detail": "즉시 운행 중단 후 자연 냉각, 주변 발열원 제거", "before": 65, "after": 40, "duration": "20분"},
            {"action": "열원 연쇄 점검", "detail": "인접 부품·모터·전원부 중 추가 열원 여부 순차 점검", "before": None, "after": None, "duration": "5분"},
            {"action": "냉각 모듈 교체", "detail": "냉각 팬·방열판·히트싱크 등 냉각 계통 부품 신품 교체", "before": 65, "after": 26, "duration": "40분"},
        ],
        "target": {"NTC": {"before": 65, "after": 26}},
    },
}

OHT_GUIDE = {
    "NTC": {
        "label": "시스템 온도 (NTC)",
        "actions": [
            {"action": "쿨링 팬 RPM 최대 가동 및 공차주행", "detail": "쿨링 팬을 최대 RPM으로 가동하면서 공차 상태로 주행하여 자연 냉각 유도", "before": 65, "after": 40, "duration": "15분"},
        ],
    },
    "CT2": {
        "label": "전류 CT2",
        "actions": [
            {"action": "공차 주행", "detail": "부하를 제거한 공차 상태로 주행하여 전류 즉각 감소 유도", "before": 60, "after": 2, "duration": "즉각적"},
        ],
    },
    "ir_temp_max": {
        "label": "소자 온도 (ir_temp_max)",
        "actions": [
            {"action": "공차 주행", "detail": "부하를 제거한 공차 상태로 주행하여 소자 온도 신속 감소", "before": 110, "after": 40, "duration": "3분"},
        ],
    },
}


@app.get("/maintenance-guide")
def get_maintenance_guide(device_type: str = Query(default="AGV")):
    """AGV 또는 OHT 조치 가이드 반환"""
    dtype = device_type.upper()
    if dtype == "AGV":
        return {"device_type": "AGV", "guide": AGV_GUIDE}
    elif dtype == "OHT":
        return {"device_type": "OHT", "guide": OHT_GUIDE}
    else:
        raise HTTPException(status_code=400, detail="device_type은 AGV 또는 OHT 이어야 합니다.")


# ──────────────────────────────────────────────
# Claude 챗봇
# ──────────────────────────────────────────────

STATE_LABEL = {0: "정상", 1: "주의", 2: "경고", 3: "위험"}

SOP_TEXT = """
[전류 임계] 전류 임계 초과 시 표준 조치 절차
1. 해당 설비 즉시 감속 또는 정지 후 전원 확인
2. 전류계·접점 상태 점검 및 과부하 원인 기록
3. 부하 원인 제거 후 저부하로 재기동하여 확인
4. 동일 재발 시 부품(접점/케이블) 교체 검토

[온도 이상] 온도 이상 시 표준 조치 절차
1. 설비 가동 중단 및 환기/냉각 상태 확인
2. 온도 센서·냉각팬·필터 점검 및 이력 확인
3. 원인 제거 후 서서히 재기동, 온도 추이 모니터링
4. 재발 시 냉각 시스템 또는 부품 교체

[진동] 진동 이상 시 표준 조치 절차
1. 설비 정지 후 육안·청각 점검(이물, 풀림, 마모)
2. 베어링·커플링·볼트 체결 상태 점검 및 진동치 기록
3. 이상 부품 교체 또는 조정 후 무부하→유부하 순차 시운전
4. 조치 내용·재발 방지 대책을 조치 이력에 기록

[센서] 센서 이상 시 표준 조치 절차
1. 해당 센서 신호값·배선·접촉 상태 점검
2. 교정 가능 여부 확인 후 교정 또는 임시 우회 절차 적용
3. 교정/교체 후 기준치 대비 검증 및 이력 기록
4. 동일 유형 재발 시 설비/센서 교체 검토

[기타] 기타 고장 공통 절차
1. 설비 안전 정지 후 현상 및 원인 기록
2. 점검·조치 후 시운전
3. 조치 이력 및 재발 방지 대책 등록
""".strip()


def _build_db_context() -> str:
    """DB에서 최신 데이터를 가져와 컨텍스트 문자열로 반환"""
    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        # 최근 조치 이력 30건 (employees JOIN)
        cursor.execute("""
            SELECT c.device_id, c.device_type, c.description, c.technician,
                   c.category, c.before_state, c.after_state, c.actioned_at, c.completed_at,
                   e.name AS emp_name, e.position AS emp_position,
                   e.department AS emp_department, e.contact AS emp_contact
            FROM corrective_logs c
            LEFT JOIN employees e ON c.employee_id = e.id
            ORDER BY c.created_at DESC LIMIT 30
        """)
        corrective = cursor.fetchall() or []

        # 최근 예방 이력 20건 (employees JOIN)
        cursor.execute("""
            SELECT p.device_id, p.device_type, p.detected_at, p.description,
                   p.predicted_state_before, p.predicted_state_after, p.result, p.technician,
                   e.name AS emp_name, e.position AS emp_position,
                   e.department AS emp_department, e.contact AS emp_contact
            FROM preventive_logs p
            LEFT JOIN employees e ON p.employee_id = e.id
            ORDER BY p.created_at DESC LIMIT 20
        """)
        preventive = cursor.fetchall() or []

        # 직원 목록 전체
        cursor.execute("SELECT id, name, position, department, contact FROM employees ORDER BY name")
        employees = cursor.fetchall() or []

        # 최근 알람 (AGV + OHT 각 10건)
        alarms = []
        for tbl, dtype in [("agv_sensor_data", "AGV"), ("oht_sensor_data", "OHT")]:
            try:
                cursor.execute(f"""
                    SELECT device_id, MIN(datetime) AS alarm_start,
                           TIMESTAMPDIFF(MINUTE, MIN(datetime), MAX(datetime)) AS duration_minutes,
                           MAX(state) AS max_state
                    FROM {tbl} WHERE state >= 2
                    GROUP BY device_id, DATE(datetime)
                    HAVING duration_minutes >= 20
                    ORDER BY alarm_start DESC LIMIT 10
                """)
                rows = cursor.fetchall() or []
                for r in rows:
                    r["device_type"] = dtype
                alarms.extend(rows)
            except Exception:
                pass

        cursor.close()
        conn.close()

        def emp_info(r: dict) -> str:
            """조치/예방 이력 행에서 담당자 정보 문자열 생성"""
            name = r.get("emp_name") or r.get("technician") or "-"
            position = r.get("emp_position") or ""
            dept = r.get("emp_department") or r.get("category") or ""
            contact = r.get("emp_contact") or ""
            parts = [name]
            if position: parts.append(position)
            if dept: parts.append(dept)
            if contact: parts.append(contact)
            return " / ".join(parts)

        # 텍스트 조합
        corrective_text = "\n".join(
            f"- [{r['device_id']}({r['device_type']})] {r['actioned_at'] or '일시미상'} | "
            f"{r['description'] or '-'} | 담당:{emp_info(r)} | "
            f"{STATE_LABEL.get(r['before_state'], '-')}→{STATE_LABEL.get(r['after_state'], '-')}"
            for r in corrective
        ) or "없음"

        preventive_text = "\n".join(
            f"- [{r['device_id']}({r['device_type']})] {r['detected_at'] or '일시미상'} | "
            f"{r['description'] or '-'} | 결과:{r['result'] or '-'} | 담당:{emp_info(r)}"
            for r in preventive
        ) or "없음"

        employees_text = "\n".join(
            f"- {e['name']} | {e['position'] or '-'} | {e['department'] or '-'} | {e['contact'] or '-'}"
            for e in employees
        ) or "없음"

        alarm_text = "\n".join(
            f"- [{r['device_id']}({r['device_type']})] {r['alarm_start']} | "
            f"지속:{r['duration_minutes']}분 | 최대상태:{STATE_LABEL.get(r['max_state'], r['max_state'])}"
            for r in sorted(alarms, key=lambda x: x['alarm_start'], reverse=True)
        ) or "없음"

        return f"""## 직원 정보 (이름 | 직급 | 부서 | 연락처)
{employees_text}

## 최근 조치 이력 (담당자 정보 포함)
{corrective_text}

## 예방 이력 (담당자 정보 포함)
{preventive_text}

## 최근 알람 (20분 이상)
{alarm_text}"""

    except Exception as e:
        return f"## DB 조회 실패\n{str(e)}"


def _build_guide_text() -> str:
    """AGV/OHT 조치 가이드를 텍스트로 변환"""
    lines = ["## AGV 조치 가이드 (센서별 조치방법·목표수치·소요시간)"]
    for sensor, data in AGV_GUIDE.items():
        lines.append(f"\n### AGV {data['label']}")
        lines.append("▶ State 2 (경고) 조치:")
        for a in data["state2"]:
            before = f"{a['before']}" if a["before"] is not None else "-"
            after = f"{a['after']}" if a["after"] is not None else "-"
            lines.append(f"  - {a['action']}: {before}→{after}, {a['duration']}")
        lines.append("▶ State 3 (위험) 조치:")
        for a in data["state3"]:
            before = f"{a['before']}" if a["before"] is not None else "-"
            after = f"{a['after']}" if a["after"] is not None else "-"
            lines.append(f"  - {a['action']}: {before}→{after}, {a['duration']}")
        if data.get("target"):
            target_str = ", ".join(f"{k}: {v['before']}→{v['after']}" for k, v in data["target"].items())
            lines.append(f"  ✓ 개선 목표: {target_str}")

    lines.append("\n## OHT 조치 가이드 (센서별 조치방법·목표수치·소요시간)")
    for sensor, data in OHT_GUIDE.items():
        lines.append(f"\n### OHT {data['label']}")
        for a in data["actions"]:
            before = f"{a['before']}" if a["before"] is not None else "-"
            after = f"{a['after']}" if a["after"] is not None else "-"
            lines.append(f"  - {a['action']}: {before}→{after}, {a['duration']}")

    return "\n".join(lines)


def _build_system_prompt() -> str:
    db_context = _build_db_context()
    guide_text = _build_guide_text()
    return f"""너는 현장 작업자를 돕는 베테랑 선임이야. 말투는 친절하고 든든하게 해 줘.

## 역할
- 설비보전·고장·알람·조치 이력 등에 대해 물으면 아래 DB 데이터, 조치 가이드, SOP를 참고해서 답해 줘.
- 조치 방법을 물으면 반드시 가이드의 구체적인 수치(조치 전→후)와 소요시간을 포함해서 답해 줘.
- 가이드 수치와 실제 조치 수치 차이가 크면 "가이드 기준과 차이가 있으니 다시 한번 확인해보세요"라고 꼭 언급해 줘.
- 장비 ID, 날짜, 담당자 등 구체적인 값을 들어 답해 줘.
- DB에 없는 정보는 "DB에 해당 내용이 없습니다"라고 솔직하게 말해 줘.

{db_context}

{guide_text}

## 고장 유형별 SOP
{SOP_TEXT}

## 톤
- 선배가 후배에게 말해 주는 것처럼 친절하고 든든하게."""


@app.post("/chat")
def chat(body: ChatRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")

    client = anthropic.Anthropic(api_key=api_key)

    messages = [
        {"role": m.role, "content": m.content}
        for m in body.messages
        if m.role in ("user", "assistant")
    ]

    if not messages:
        raise HTTPException(status_code=400, detail="messages가 비어 있습니다.")

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        system=_build_system_prompt(),
        messages=messages,
    )

    text = response.content[0].text if response.content else ""
    return {"content": text}
