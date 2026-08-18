"""本地任务库(SQLite)。

存在的理由:视频类工作流端到端可能跑十几分钟,agent 不可能一直阻塞等待。
提交与取结果必须能拆成两次独立调用 —— 中间即使 agent 重启、会话结束,
凭 job_id 也能把结果捡回来。

同时记录**实际生效的参数**(含随机生成的 seed),否则「复现上次那张」无从谈起。
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path.home() / ".config" / "comfyagent" / "jobs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    endpoint_id TEXT NOT NULL,
    status      TEXT NOT NULL,
    params      TEXT NOT NULL,
    outputs     TEXT,
    error       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    return c


def record(job_id, workflow_id, endpoint_id, params, status="queued"):
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO jobs "
            "(job_id, workflow_id, endpoint_id, status, params, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (job_id, workflow_id, endpoint_id, status,
             json.dumps(params, ensure_ascii=False), now, now),
        )


def update(job_id, status, outputs=None, error=None):
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status=?, outputs=COALESCE(?, outputs), "
            "error=COALESCE(?, error), updated_at=? WHERE job_id=?",
            (status,
             json.dumps(outputs, ensure_ascii=False) if outputs is not None else None,
             error, time.time(), job_id),
        )


def get(job_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    return dict(row) if row else None


def recent(limit=20, status=None):
    q = "SELECT * FROM jobs"
    args = []
    if status:
        q += " WHERE status=?"
        args.append(status)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with _conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]
