"""Receiver for per-robot onboard ball-detection telemetry.

Each robot's RobotFramework sends a UDP packet with its latest camera
observation (ball presence, pixel coords, bearing, confidence). This
module parses those packets and exposes the latest reading per robot
through `OnboardObservationStore`.

Public API:
  OnboardObservation  — parsed reading + receive timestamp
  OnboardReceiver     — background UDP listener thread
  OnboardObservationStore — thread-safe per-robot snapshot store
  parse_packet        — stateless key=value parser
"""

from .observation import OnboardObservation, parse_packet
from .store import OnboardObservationStore
from .receiver import OnboardReceiver

__all__ = [
    "OnboardObservation",
    "OnboardObservationStore",
    "OnboardReceiver",
    "parse_packet",
]
