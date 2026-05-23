# ColonAI Security Policy

Last reviewed: 2026-05-23

ColonAI processes medical images and writes audit logs that may identify a
case (image SHA-256, timestamp, model verdict). It is **decision-support
software for screening review**, not a medical device. Treat the security
posture accordingly.

## Threat model

| Threat | Mitigation |
|---|---|
| Malicious model checkpoint (pickle RCE via `torch.load`) | `src.app.security.safe_torch_load` prefers `weights_only=True`. Only first-party checkpoints from this repo are loaded with `allow_unsafe=True`. **Never** point `CHECKPOINT` at a download from an untrusted source. |
| Decompression-bomb image upload | `PIL.Image.MAX_IMAGE_PIXELS = 100_000_000`; `LOAD_TRUNCATED_IMAGES = False`. Set at import time by `src.app.security`. |
| Oversize upload DOS | 10 MB hard cap at the validator (`MAX_UPLOAD_BYTES`); Streamlit-side cap also 10 MB. |
| Spoofed content type | MIME / extension allow-list + Pillow magic-byte check. |
| Network exposure | Both Streamlit and the API default to `127.0.0.1`. Binding to `0.0.0.0` without `COLONAI_API_KEY` is refused by the API. |
| Unauthenticated `/predict` access | `X-API-Key` header check when `COLONAI_API_KEY` env var is set. |
| Audit-log exfiltration via `/audit/today` | Endpoint requires the same API key. File is created with `chmod 0o600`. |
| Cross-origin abuse | FastAPI CORS allow-list defaults to localhost only; configurable via `COLONAI_CORS_ORIGINS`. Streamlit has CORS disabled in `.streamlit/config.toml`. |
| Traceback information leak | All caller-facing errors go through `safe_error()` — only a `request_id` is returned; the stack lives in the server log. |
| API docs exposure | `/docs` (Swagger) is disabled unless `COLONAI_EXPOSE_DOCS` is set. |
| Sensitive data in URL query strings | None of the endpoints accept PHI in query strings. |
| HIPAA/GDPR data retention | Out of scope for this project — operator is responsible for log rotation, encryption-at-rest, and access reviews. |

## Required environment variables for production

| Variable | Required if | Purpose |
|---|---|---|
| `COLONAI_API_KEY` | binding outside `127.0.0.1` | Shared secret for `/predict`, `/audit/today`. |
| `COLONAI_BIND` | exposing on LAN | Bind address. Default `127.0.0.1`. |
| `COLONAI_PORT` | optional | Port. Default 8081. |
| `COLONAI_CORS_ORIGINS` | optional | Comma-separated origin allow-list. |
| `COLONAI_EXPOSE_DOCS` | only on dev hosts | When set, enables `/docs` (Swagger). |
| `COLONAI_LOG_LEVEL` | optional | `DEBUG`, `INFO` (default), `WARNING`. |

## Operator runbook

1. Generate a 256-bit key: `openssl rand -hex 32`.
2. Export `COLONAI_API_KEY=...` in the systemd unit / launch script.
3. Run behind a reverse proxy (nginx / Caddy) that adds TLS. ColonAI does not terminate TLS itself.
4. Configure the proxy to enforce rate limits (e.g. `limit_req` in nginx). ColonAI does not rate-limit at the application layer.
5. Configure log rotation for `outputs/audit/audit_*.jsonl`.
6. Periodically review `outputs/audit/` files — every prediction is recorded with the image SHA-256, verdict, and confidence.

## Reporting

Report vulnerabilities privately to the project maintainer (see top of `README.md`). Do not file public GitHub issues for security bugs.
