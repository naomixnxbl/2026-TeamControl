"""Parsed onboard ball observation + packet decoder."""

from dataclasses import dataclass


@dataclass
class OnboardObservation:
    """A single onboard-camera ball reading.

    found       : True if the robot's camera sees an orange blob this frame.
    px, py      : Pixel centroid in the image (origin top-left).
    radius      : Enclosing-circle radius in pixels (rough distance proxy).
    bearing     : Horizontal bearing from camera optical axis in radians,
                  positive = right of center. Combine with robot heading
                  to get a world-frame bearing to the ball.
    confidence  : 0..1 roundness score; reject readings below ~0.3.
    robot_ts_ms : Timestamp from the robot's steady_clock (ms).
    recv_ts     : Wall-clock time.time() when the PC received the packet.
    robot_id    : Robot that produced the packet (if inferable).
    is_yellow   : Robot's team color (if inferable).
    """

    found: bool = False
    px: float = 0.0
    py: float = 0.0
    radius: float = 0.0
    bearing: float = 0.0
    confidence: float = 0.0
    robot_ts_ms: int = 0
    recv_ts: float = 0.0
    robot_id: int = -1
    is_yellow: bool = True


def _coerce(key, value):
    try:
        if key == "ball":
            return bool(int(value))
        if key in ("px", "py", "r", "radius", "bearing", "conf",
                   "confidence", "voltage"):
            return float(value)
        if key in ("ts_ms", "robot_id", "id"):
            return int(float(value))
        if key in ("yellow", "is_yellow"):
            return bool(int(value))
    except (ValueError, TypeError):
        return None
    return value


def parse_packet(payload):
    """Parse a RobotFramework telemetry packet.

    Expected format: `state=active,voltage=...,ball=1,px=160,py=120,...`
    Unknown keys are ignored. Returns an OnboardObservation with any
    missing fields left at their defaults.
    """
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8", errors="replace")
        except Exception:
            return None
    if not isinstance(payload, str) or "=" not in payload:
        return None

    kv = {}
    for token in payload.strip().split(","):
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        parsed = _coerce(k, v)
        if parsed is not None:
            kv[k] = parsed

    obs = OnboardObservation()
    if "ball" in kv:
        obs.found = bool(kv["ball"])
    obs.px = float(kv.get("px", 0.0))
    obs.py = float(kv.get("py", 0.0))
    obs.radius = float(kv.get("r", kv.get("radius", 0.0)))
    obs.bearing = float(kv.get("bearing", 0.0))
    obs.confidence = float(kv.get("conf", kv.get("confidence", 0.0)))
    obs.robot_ts_ms = int(kv.get("ts_ms", 0))
    if "robot_id" in kv:
        obs.robot_id = int(kv["robot_id"])
    elif "id" in kv:
        obs.robot_id = int(kv["id"])
    if "yellow" in kv:
        obs.is_yellow = bool(kv["yellow"])
    elif "is_yellow" in kv:
        obs.is_yellow = bool(kv["is_yellow"])
    return obs
