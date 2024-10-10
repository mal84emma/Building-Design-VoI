"""Helper functions for handling function calls."""

import time

def retry_wrapper(f, n_retrys=5, delay=1):
    """Wrapper function for retrying arg function n times.

    Args:
        f (function): Target function to retry.
        n_retrys (int, optional): Number of retries to attempt. Defaults to 5.
        delay (int, optional): Delay (seconds) between retries. Defaults to 1.
    """

    def retried_f(*args, **kwargs):
        for _ in range(n_retrys):
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                print(f'Error: {e}')
                time.sleep(delay)
        raise Exception(f'Function {f.__name__} failed after {n_retrys} retries.')

    return retried_f