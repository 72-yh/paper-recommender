# Fly.io Low-Cost Deployment Runbook

This runbook keeps the Paper Recommender MVP within a small monthly budget. It
is intentionally conservative: one public API app, one small Machine, one small
volume, and no managed services.

## Cost Guardrails

- Use exactly one `shared-cpu-1x` Machine with 1GB RAM.
- Keep `min_machines_running = 0` and `auto_stop_machines = "stop"` so the
  Machine can stop when idle.
- Use one 2GB volume named `paper_recommender_data`.
- Do not run `fly ips allocate-v4`; the shared IPv4 and Anycast IPv6 addresses
  are enough for this MVP.
- Do not enable metrics-based autoscaling.
- Do not create Managed Postgres, Redis, Tigris, or other managed services.
- Do not create extra regions, extra Machines, GPUs, Kubernetes, or support
  plans.
- Do not use a remote builder for this small deployment; use local Docker with
  `--local-only` to avoid builder resources.
- Check Fly Dashboard > Billing before and after every deploy session.

As of 2026-05-25, Fly's public pricing makes this a low-dollar deployment, but
it is not a hard billing cap. Treat the Fly dashboard's current-month usage as
the source of truth.

## Local Inputs

The Docker image contains only app code. The serving artifacts stay outside the
image and are uploaded to the Fly volume:

- `data/paper_recommender_1m.db`
- `data/vectors_1m_int8.npz`

Current local artifact sizes are about 551 MB total, so a 2GB volume is enough
for the 1M proof deployment. A larger backfill requires an explicit volume-size
review before upload.

## Safe Preparation

These commands should not create paid Fly resources:

```powershell
fly auth login
fly version
```

Install `flyctl` first if `fly` is not on `PATH`.

## Cost-Incurring Commands

Run these only after confirming the app name, region, and current Fly pricing.
Use `--ha=false`; plain `fly launch` defaults to creating spare Machines for
availability.

```powershell
fly apps create paper-recommender-72yh
fly volumes create paper_recommender_data --app paper-recommender-72yh --region sjc --size 2
fly deploy --app paper-recommender-72yh --ha=false --local-only
```

After deploy, confirm there is one Machine and one volume:

```powershell
fly status --app paper-recommender-72yh
fly machine list --app paper-recommender-72yh
fly volumes list --app paper-recommender-72yh
```

## Artifact Upload

Upload the artifacts into the mounted `/app/data` volume:

```powershell
fly sftp put data/paper_recommender_1m.db /app/data/paper_recommender_1m.db --app paper-recommender-72yh
fly sftp put data/vectors_1m_int8.npz /app/data/vectors_1m_int8.npz --app paper-recommender-72yh
```

If direct `put` does not work on the installed `flyctl`, use an interactive
session instead:

```powershell
fly sftp shell --app paper-recommender-72yh
```

Then run the same two `put` operations inside the SFTP shell.

Machine may auto-stop during long SFTP uploads. If an upload disconnects near
the end, verify the file size on the volume before retrying. During the 1M proof
deployment, the database upload completed, the vector upload needed a retry, and
a lightweight `/health` keepalive helped keep the Machine available during the
second upload.

## Verification

Use the deployment smoke test against the Fly URL:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_deployment.py `
  --base-url https://paper-recommender-72yh.fly.dev `
  --query-url https://arxiv.org/abs/0704.0004 `
  --expected-index-kind int8 `
  --min-indexed-papers 1000000
```

Then check billing again:

```powershell
fly dashboard --app paper-recommender-72yh
```

The first recommendation requests can load the index from the mounted volume.
For the 1M int8 proof, this is a 340MB file, so a cold request can take much
longer than a warm request. The app protects the first load with an index-load
lock, and the UI disables duplicate submits while a search is running.

## Cleanup

If the bill estimate looks wrong, stop before experimenting further. To remove
the deployment, delete the app and confirm no volumes or managed services remain
in the Fly dashboard:

```powershell
fly apps destroy paper-recommender-72yh
```
