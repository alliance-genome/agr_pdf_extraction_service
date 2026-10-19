"""Bounded scheduling for independent page-provenance model requests."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def resolve_page_batches(batches, resolve, *, concurrency, may_submit):
    """Return choices and receipts in input order, never completion order.

    Only running-capacity work is submitted. Once the job budget is exhausted,
    stop submitting but collect already-started requests. On cancellation or an
    unexpected exception, cancel queued work and drain running calls before
    propagating it, so accounting cannot change after the caller finalizes.
    Each provider request retains its existing timeout/retry bounds.
    """
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("PAGE_PROVENANCE_LLM_CONCURRENCY must be a positive integer")
    if not batches:
        return {}, []

    results = {}
    pending = {}
    next_index = 0
    stopped = False
    pool = ThreadPoolExecutor(max_workers=min(concurrency, len(batches)))
    try:
        while pending or (next_index < len(batches) and not stopped):
            while not stopped and next_index < len(batches) and len(pending) < concurrency:
                if not may_submit():
                    stopped = True
                    break
                future = pool.submit(resolve, batches[next_index])
                pending[future] = next_index
                next_index += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Observe all failures in this completion set before replenishing.
            for future in done:
                index = pending.pop(future)
                results[index] = future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    choices = {}
    receipts = []
    for index in sorted(results):
        decisions, receipt = results[index]
        choices.update(decisions)
        receipts.append(receipt)
    return choices, receipts
