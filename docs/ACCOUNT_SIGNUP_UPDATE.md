# ReliaGraph account screen update

## What changed

- The first page always displays both Sign in and Sign up tabs.
- A new workspace defaults to Sign up; an existing workspace defaults to Sign in. Both options remain visible either way.
- `POST /api/auth/signup` supports the first account and additional local accounts.
- Sign-up input uses the same username and minimum-password rules as initial setup, and throttles repeated signup attempts.
- The existing `POST /api/auth/setup` endpoint is retained for backward compatibility.
- Passwords continue to use salted PBKDF2-SHA256 hashes and sessions keep CSRF checks for state-changing authenticated operations.

## Workspace scope caveat

Accounts authenticate separately, but all accounts on this prototype share the same local document corpus, graph and dataset history. There is no per-user workspace isolation. The app is intended to run locally on `127.0.0.1`; do not expose it as a public multi-user service without adding tenant isolation, hardened deployment, HTTPS and an audited identity design.

## Validation

- Automated test suite: 100 passed.
- Web UI exposes both mode buttons in the served HTML.
- API test creates the first account using `/api/auth/signup`, ingests sample data, logs out, creates a second account, and verifies the second account can authenticate.
- JavaScript syntax check passed with Node.js.
