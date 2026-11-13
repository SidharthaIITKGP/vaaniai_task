from relay.scheduler_benchmark import simulate_fixed_window, simulate_token_bucket


def test_token_bucket_simulation_reduces_artificial_full_window_waits():
    calls = [
        ({"stage": "generation", "input_tokens": 2800, "output_tokens": 300,
          "latency_ms": 1000}, False),
        ({"stage": "verification", "input_tokens": 3100, "output_tokens": 150,
          "latency_ms": 1000}, True),
    ]
    old = simulate_fixed_window(calls, 6400, 0)
    new = simulate_token_bucket(calls, 6400, 0)
    assert old > 50
    assert new < old / 4
