"""
Tests for the compare API.

  GET /compare/projects  -> lists completed jobs (most recent first)
  GET /compare/{a}/{b}   -> returns both completed jobs, 404 if either missing
                            or not completed.

Jobs are seeded via the repository (same DATABASE_URL as the app), then read
back through the HTTP endpoints.
"""


def _seed_completed(repo, job_id):
    repo.create(job_id, f"/tmp/{job_id}.png", f"/tmp/out/{job_id}")
    repo.mark_completed(job_id, {"model_url": f"/outputs/{job_id}/model.ply"})


def test_projects_lists_completed_jobs(client, repo):
    _seed_completed(repo, "cmp_a")
    _seed_completed(repo, "cmp_b")

    resp = client.get("/compare/projects")
    assert resp.status_code == 200, resp.text
    ids = {p["id"] for p in resp.json()["projects"]}
    assert {"cmp_a", "cmp_b"} <= ids

    # Project shape includes model_url derived from result dict.
    by_id = {p["id"]: p for p in resp.json()["projects"]}
    assert by_id["cmp_a"]["model_url"] == "/outputs/cmp_a/model.ply"
    assert by_id["cmp_a"]["status"] == "completed"


def test_projects_excludes_non_completed(client, repo):
    _seed_completed(repo, "cmp_done")
    repo.create("cmp_running", "/tmp/r.png", "/tmp/out/r")
    repo.update("cmp_running", status="processing")

    ids = {p["id"] for p in client.get("/compare/projects").json()["projects"]}
    assert "cmp_done" in ids
    assert "cmp_running" not in ids


def test_compare_two_returns_both(client, repo):
    _seed_completed(repo, "cmp_x")
    _seed_completed(repo, "cmp_y")

    resp = client.get("/compare/cmp_x/cmp_y")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["a"]["id"] == "cmp_x"
    assert body["b"]["id"] == "cmp_y"
    assert body["a"]["model_url"] == "/outputs/cmp_x/model.ply"


def test_compare_unknown_id_returns_404(client, repo):
    _seed_completed(repo, "cmp_known")
    resp = client.get("/compare/cmp_known/cmp_missing")
    assert resp.status_code == 404


def test_compare_non_completed_returns_404(client, repo):
    _seed_completed(repo, "cmp_ok")
    repo.create("cmp_pending", "/tmp/p.png", "/tmp/out/p")
    repo.update("cmp_pending", status="processing")

    resp = client.get("/compare/cmp_ok/cmp_pending")
    assert resp.status_code == 404
