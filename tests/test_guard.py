import threading

from classeval.guard import limit_gpu_memory, start_memory_watchdog


def test_watchdog_stops_the_process_when_a_limit_is_exceeded():
    fired = threading.Event()
    codes: list[int] = []

    def fake_exit(code: int) -> None:
        codes.append(code)
        fired.set()

    start_memory_watchdog(max_process_gb=0.0001, interval_s=0.01, exit_fn=fake_exit)
    assert fired.wait(timeout=5.0)
    assert codes == [3]


def test_watchdog_stops_when_system_memory_is_nearly_exhausted():
    fired = threading.Event()
    start_memory_watchdog(min_system_free_gb=10_000.0, interval_s=0.01, exit_fn=lambda code: fired.set())
    assert fired.wait(timeout=5.0)


def test_watchdog_leaves_a_healthy_process_alone():
    fired = threading.Event()
    start_memory_watchdog(interval_s=0.01, exit_fn=lambda code: fired.set())
    assert not fired.wait(timeout=0.3)


def test_gpu_memory_cap_is_requested_when_cuda_is_present(monkeypatch):
    import torch

    requested: list[float] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction: requested.append(fraction))
    limit_gpu_memory(0.9)
    assert requested == [0.9]


def test_gpu_memory_cap_does_nothing_without_cuda(monkeypatch):
    import torch

    requested: list[float] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction: requested.append(fraction))
    limit_gpu_memory(0.9)
    assert requested == []
