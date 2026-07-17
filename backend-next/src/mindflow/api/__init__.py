"""MindFlow API layer.

Structured as:
  - errors.py: RFC 9457 ProblemDetail exception + FastAPI exception handlers
  - middleware/: auth, host validation, rate limiting, structured logging
  - routes/: REST endpoint definitions
  - websocket.py: WebSocket handler for real-time push
"""

from __future__ import annotations
