import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv/bin/python"
SCRIPT = REPO / "ml/inference_worker.py"
MODEL = REPO / "artifacts/model.keras"
DATASET = REPO / "data/raw/plantvillage"


def read_line(proc: subprocess.Popen, timeout: float = 60.0) -> str:
    """Read one line from the worker, failing instead of blocking forever."""
    result: list[str] = []
    reader = threading.Thread(target=lambda: result.append(proc.stdout.readline()), daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        proc.kill()
        pytest.fail(f"worker produced no answer within {timeout:.0f}s")
    return result[0] if result else ""


def first_dataset_image() -> Path | None:
    return next(iter(sorted(DATASET.rglob("*.JPG"))), None)


@pytest.fixture
def worker():
    if not (PYTHON.is_file() and MODEL.is_file()):
        pytest.skip("python environment or model artifact not built yet")
    proc = subprocess.Popen(
        [str(PYTHON), str(SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    line = read_line(proc)
    if not line:
        proc.kill()
        pytest.fail("worker produced no startup line")
    startup = json.loads(line)
    if startup.get("type") == "error":
        proc.kill()
        pytest.fail(f"worker failed to start: {startup}")
    assert startup["type"] == "ready"
    yield proc, startup
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def request(proc: subprocess.Popen, payload: dict) -> dict:
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = read_line(proc)
    assert line, "worker closed its output"
    return json.loads(line)


def test_ping(worker):
    proc, _ = worker
    assert request(proc, {"id": 1, "type": "ping"}) == {"id": 1, "type": "pong"}


def test_predict_returns_ranked_probabilities(worker):
    proc, ready = worker
    image = first_dataset_image()
    if image is None:
        pytest.skip("dataset not present")
    response = request(
        proc, {"id": 2, "type": "predict", "path": str(image), "top_k": 3}
    )
    assert response["type"] == "result"
    predictions = response["predictions"]
    assert len(predictions) == 3
    probabilities = [p["probability"] for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert all(p["class"] in ready["classes"] for p in predictions)
    assert all(p["label"] for p in predictions)


def test_several_requests_share_one_process(worker):
    proc, _ = worker
    for request_id in (10, 11, 12):
        response = request(proc, {"id": request_id, "type": "ping"})
        assert response == {"id": request_id, "type": "pong"}


def test_missing_file_is_a_structured_error(worker):
    proc, _ = worker
    response = request(
        proc, {"id": 3, "type": "predict", "path": "/nonexistent/leaf.jpg"}
    )
    assert response["type"] == "error"
    assert response["code"] == "file_not_found"
    assert response["id"] == 3


def test_paths_with_spaces_and_unicode(worker, tmp_path):
    proc, _ = worker
    image = first_dataset_image()
    if image is None:
        pytest.skip("dataset not present")
    awkward = tmp_path / "l eaf afbeelding 名前.png"
    shutil.copyfile(image, awkward)
    response = request(proc, {"id": 4, "type": "predict", "path": str(awkward)})
    assert response["type"] == "result", response


def test_malformed_request_gets_an_error(worker):
    proc, _ = worker
    proc.stdin.write("this is not json\n")
    proc.stdin.flush()
    response = json.loads(read_line(proc))
    assert response["type"] == "error"
    assert response["code"] == "bad_request"


def test_unknown_type_gets_an_error(worker):
    proc, _ = worker
    response = request(proc, {"id": 5, "type": "dance"})
    assert response["type"] == "error"
    assert response["code"] == "unknown_type"


def test_shutdown_says_goodbye(worker):
    proc, _ = worker
    response = request(proc, {"id": 6, "type": "shutdown"})
    assert response == {"id": 6, "type": "bye"}
    assert proc.wait(timeout=10) == 0
