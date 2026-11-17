# Debugging a deployment — `seq policy doctor` and `seq policy logs`

When a deployment you drive starts misbehaving — some control requests are slower than expected, or
one just failed — two commands tell you **which segment of the request path** is at fault and let you
**pull the log record for one request by its id**. Both are read-only.

- `seq policy doctor <deployment>` — the platform walks the deployment's request path segment by segment (is it
  the deployment itself, its capacity, its last cold start, or a dead worker?).
- `seq policy logs <deployment>` — the deployment's structured logs, with the gateway `connect.*` sink and the
  worker `act.*` sink merged and time-ordered. Add `--rid r-…` to pull one request and join its worker
  phase with the client-reported timing, attributed to the stage that failed. `--build` shows the newest
  version's build instead: its phases and its whole log.

Authorization is enforced **server-side**: the gateway checks your `SEQUENCES_API_KEY` (hashed to an
identity, then owner-scoped) before returning any rows. You only ever see your own account's data; the CLI
never reads the platform's logs with credentials of its own, and does not filter client-side.

---

## The user's walkthrough: "some requests are slow, one failed"

**1. Localize with `seq policy doctor`.** The platform walks the deployment's request path and reports each
segment separately: `initial-connect` (which version serves on which card, how many workers run), `capacity`
(the slots taken and robots waiting, the GPU memory left), `cold-start` (the last start: how long loading the
policy took, or where it failed), and `worker-ping` (a running worker answers):

```bash
$ seq policy doctor pick_place_pi05_user
doctor pick_place_pi05_user: OK
  [OK ] initial-connect: v3 serves on L4; 1 worker(s) running
  [OK ] capacity: 1 of 4 slot(s) taken on 1 worker(s); GPU memory: 1 copy of 11.2 GiB on a 24 GiB card, 12.8 GiB left
  [OK ] cold-start: last start 42 min ago: loaded in 38.1 s, ready 96.4 s after it began
  [OK ] worker-ping: healthy in 12 ms
```

A dead worker is a `worker-ping` `FAIL`; a start that failed is a `cold-start` `FAIL` with the phase and reason; full slots or
little GPU memory left is a `capacity` `WARN`; each is a distinct segment. The doctor starts no worker: a
deployment scaled to zero reports `worker-ping` as skipped ("no worker running (scaled to zero): the next connect
starts one") and stays `OK`. Whether a slow request was slow on the network or in the policy's inference is that
request's to say: its client timing beside the worker's own phase, step 2.

**2. Pull the failed request with `seq policy logs --rid`.** The record is time-ordered by emission timestamp,
with per-hop timing, the worker phase (from the worker's own log line) joined to the client timing (from the
persisted `request_latency` row), and the failure attributed to a stage:

```bash
$ seq policy logs pick_place_pi05_user --rid r-worker01
rid r-worker01
  ── Sun 2026-10-11 · EDT ──
  14:05:09.149  [connect] connect.ok  deployment=pick_place_pi05_user direct_secret=***
  14:05:09.151  [worker ] act.start  (+2.0ms)
  14:05:09.210  [worker ] act.fail  (+59.0ms)  err=cuda_oom status=500
  client timing: {'total_ms': '61.0', ... 'outcome': 'error'}
  worker phase: act.fail
  FAILURE stage=worker: cuda_oom
```

A request that fails **before the worker emits any line** (a 402 `insufficient_credit`, rotated secret 401, warming
503, or a connect/network error) is still retrievable — it is attributed from the connect-side record instead:

```bash
$ seq policy logs pick_place_pi05_user --rid r-connect01
rid r-connect01
  ── Sun 2026-10-11 · EDT ──
  14:07:22.030  [connect] connect.warming  deployment=pick_place_pi05_user status=503 retry_after_s=25
  client timing: (none persisted for this rid)
  worker phase: (no worker-side record)
  FAILURE stage=connect: warming/503 before worker (connect.warming status=503)
```

If an id has no record on either sink and no persisted timing, `seq policy logs --rid` says so explicitly
(a `rid gap`) rather than silently returning empty — that means the request id was never threaded into
the connect record upstream, which is itself the finding.

---

## Secrets never appear in output

Platform secrets the CLI handles — `direct_secret`, lease tokens, the worker's identity token, and your
`SEQUENCES_API_KEY` — are redacted to `***` in every `seq policy logs` / `seq policy doctor` line, including
verbose output (see the `direct_secret=***` above). The consumer side never un-redacts rows that were
already redacted at ingest, and introduces no new leak path.
