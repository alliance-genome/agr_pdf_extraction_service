from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest

from app.services.page_resolution import resolve_page_batches


def test_bounded_overlap_keeps_input_order_and_failure_receipts():
    started = [Event() for _ in range(6)]
    release = [Event() for _ in range(6)]
    lock = Lock()
    active = peak = 0
    completed = []

    def resolve(index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        started[index].set()
        assert release[index].wait(5)
        with lock:
            active -= 1
            completed.append(index)
        outcome = {1: 'timeout', 4: 'invalid'}.get(index, 'valid')
        return ({str(index): index + 1} if outcome == 'valid' else {}), {
            'index': index, 'outcome': outcome,
        }

    with ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(resolve_page_batches, list(range(6)), resolve,
                               concurrency=3, may_submit=lambda: True)
        try:
            assert all(event.wait(5) for event in started[:3])
            assert not any(event.is_set() for event in started[3:])
            for index, next_index in ((2, 3), (1, 4), (0, 5)):
                release[index].set()
                assert started[next_index].wait(5)
        finally:
            for event in release:
                event.set()
        choices, receipts = future.result(timeout=5)
    assert peak == 3
    assert active == 0
    assert completed[:3] == [2, 1, 0]
    assert choices == {'0': 1, '2': 3, '3': 4, '5': 6}
    assert [r['index'] for r in receipts] == list(range(6))
    assert [r['outcome'] for r in receipts] == [
        'valid', 'timeout', 'valid', 'valid', 'invalid', 'valid',
    ]


def test_budget_exhaustion_stops_admission_but_drains_active_requests():
    budget = Event()
    budget.set()
    both_started = Event()
    release = Event()
    calls = []

    def resolve(index):
        calls.append(index)
        if index == 1:
            budget.clear()
            both_started.set()
        assert both_started.wait(5)
        assert release.wait(5)
        return {str(index): index + 1}, {'index': index}

    with ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(resolve_page_batches, list(range(6)), resolve,
                               concurrency=2, may_submit=budget.is_set)
        try:
            assert both_started.wait(5)
            assert not future.done()
        finally:
            release.set()
        choices, receipts = future.result(timeout=5)
    assert sorted(calls) == [0, 1]
    assert choices == {'0': 1, '1': 2}
    assert receipts == [{'index': 0}, {'index': 1}]


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_unexpected_failure_drains_running_calls_before_propagation(error_type):
    second_started = Event()
    release = Event()
    failed = Event()
    drained = Event()
    calls = []

    def resolve(index):
        calls.append(index)
        if index == 0:
            assert second_started.wait(5)
            failed.set()
            raise error_type('stop job')
        second_started.set()
        assert release.wait(5)
        drained.set()
        return {}, {'index': index}

    with ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(resolve_page_batches, list(range(6)), resolve,
                               concurrency=2, may_submit=lambda: True)
        try:
            assert failed.wait(5)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(error_type, match='stop job'):
            future.result(timeout=5)
    assert drained.is_set()
    assert sorted(calls) == [0, 1]


def test_serial_mode_and_exhausted_budget():
    calls = []

    def resolve(index):
        calls.append(index)
        return {str(index): index + 1}, {'index': index}

    assert resolve_page_batches([0, 1], resolve, concurrency=1,
                                may_submit=lambda: False) == ({}, [])
    assert calls == []
    assert resolve_page_batches([0, 1], resolve, concurrency=1,
                                may_submit=lambda: True) == (
        {'0': 1, '1': 2}, [{'index': 0}, {'index': 1}],
    )
    assert calls == [0, 1]


@pytest.mark.parametrize('concurrency', [0, -1, 1.5, True])
def test_invalid_concurrency_is_rejected(concurrency):
    with pytest.raises(ValueError, match='positive integer'):
        resolve_page_batches([], None, concurrency=concurrency,
                             may_submit=lambda: True)


@pytest.mark.parametrize('value,expected', [(None, 4), ('1', 1), ('8', 8)])
def test_concurrency_environment_configuration(monkeypatch, value, expected):
    import runpy
    from pathlib import Path
    if value is None:
        monkeypatch.delenv('PAGE_PROVENANCE_LLM_CONCURRENCY', raising=False)
    else:
        monkeypatch.setenv('PAGE_PROVENANCE_LLM_CONCURRENCY', value)
    config = runpy.run_path(str(Path(__file__).parents[2] / 'config.py'))['Config']
    assert config.PAGE_PROVENANCE_LLM_CONCURRENCY == expected


@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'invalid'])
def test_invalid_concurrency_environment_fails_startup(monkeypatch, value):
    import runpy
    from pathlib import Path
    monkeypatch.setenv('PAGE_PROVENANCE_LLM_CONCURRENCY', value)
    with pytest.raises(ValueError):
        runpy.run_path(str(Path(__file__).parents[2] / 'config.py'))
