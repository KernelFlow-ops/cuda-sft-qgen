"""Review waits only for embeddings in its own wave."""

from concurrent.futures import Future


def test_wait_prefetch_leaves_unrelated_request_running(fake_runtime):
    needed = Future()
    needed.set_result(None)
    unrelated = Future()
    fake_runtime._prefetch_futs = {"needed": needed, "unrelated": unrelated}

    fake_runtime.wait_prefetch(["needed"])

    assert "needed" not in fake_runtime._prefetch_futs
    assert fake_runtime._prefetch_futs["unrelated"] is unrelated
    unrelated.set_result(None)
