import multiprocessing as mp
import pydeform


# NOTE: spawn a fresh interpreter for the subprocess, since running
# OpenMP code after a `fork()` syscall may cause a deadlock if OpenMP
# was also used before forking.
ctx = mp.get_context('spawn')


def _registration_worker(q, args, kwargs):
    R""" To be ran in a subprocess.

    Parameters
    ----------
    q: multiprocessing.Queue
        Queue to return the result.
    kwargs: dict
        Keyword arguments for the registration.
    """
    try:
        result = pydeform.register(*args, **kwargs)
    except BaseException as e:
        result = e
    q.put(result)


def register(*args, **kwargs):
    R""" Interruptible version of :func:`pydeform.register`.

    .. note::
        This function calls the registration routine in a subprocess
        in order to handle keyboard interrupts. This has a memory overhead,
        since a new instance of the intepreter is spawned and input objects
        are copied in the subprocess memory.

    .. seealso::
        :func:`pydeform.register`
    """

    # Run call in a subprocess, to handle keyboard interrupts
    q = ctx.Queue()
    p = ctx.Process(target=_registration_worker, args=[q, args, kwargs], daemon=True)
    p.start()
    try:
        result = q.get()
        if isinstance(result, BaseException):
            raise result
        p.join()
    except BaseException as e:
        p.terminate()
        p.join()
        raise e

    return result

