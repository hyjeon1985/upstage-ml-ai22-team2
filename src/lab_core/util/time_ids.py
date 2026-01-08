import time


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id(prefix: str = "temp", *, mid_id: str | None = None) -> str:
    ts = time.strftime("%Y%m%d_%H%M%SZ", time.gmtime())

    return f"{prefix}_{mid_id}_{ts}" if mid_id else f"{prefix}_{ts}"
