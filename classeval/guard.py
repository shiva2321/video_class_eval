"""Safety rails: a runaway stage must end with an error, never freeze the machine.

Windows lets the GPU driver spill CUDA allocations into system RAM and then into the pagefile; a stage that asks
for far more memory than the GPU has then thrashes the system disk and freezes the whole PC. These two guards
turn that into a quick, clean failure.
"""

import os
import sys
import threading
import time
from collections.abc import Callable

import psutil


def limit_gpu_memory(fraction: float = 0.85) -> None:
    """Cap this process's CUDA allocations: beyond the cap torch raises OutOfMemoryError instead of spilling."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def start_memory_watchdog(
    max_process_gb: float = 16.0,
    min_system_free_gb: float = 4.0,
    interval_s: float = 1.0,
    exit_fn: Callable[[int], None] = os._exit,
) -> threading.Thread:
    """Stop the process (exit code 3) when it grows past `max_process_gb` or the system's free RAM falls below
    `min_system_free_gb`. Prints numbers only."""
    proc = psutil.Process()

    def watch() -> None:
        while True:
            rss_gb = proc.memory_info().rss / 2**30
            free_gb = psutil.virtual_memory().available / 2**30
            if rss_gb > max_process_gb or free_gb < min_system_free_gb:
                print(
                    f"memory guard: stopping (process {rss_gb:.1f} GB, system free {free_gb:.1f} GB)",
                    file=sys.stderr,
                    flush=True,
                )
                exit_fn(3)
                return
            time.sleep(interval_s)

    thread = threading.Thread(target=watch, name="memory-watchdog", daemon=True)
    thread.start()
    return thread
