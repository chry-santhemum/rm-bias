from slist import Slist
import typing
from typing import Callable, TypeVar
import asyncio

A = TypeVar('A')
B = TypeVar('B')

async def par_map_async(
    self, func: Callable[[A], typing.Awaitable[B]], max_par: int | None = None, tqdm: bool = False, desc: str = ""
) -> Slist[B]:
    """Asynchronously apply a function to each element with optional parallelism limit.

    Parameters
    ----------
    func : Callable[[A], Awaitable[B]]
        Async function to apply to each element
    max_par : int | None, optional
        Maximum number of parallel operations, by default None
    tqdm : bool, optional
        Whether to show a progress bar, by default False

    Returns
    -------
    Slist[B]
        A new Slist with the transformed elements

    Examples
    --------
    >>> async def slow_double(x):
    ...     await asyncio.sleep(0.1)
    ...     return x * 2
    >>> await Slist([1, 2, 3]).par_map_async(slow_double, max_par=2)
    Slist([2, 4, 6])
    """
    if max_par is None:
        if tqdm:
            import tqdm as tqdm_module

            tqdm_counter = tqdm_module.tqdm(total=len(self), desc=desc)

            async def func_with_tqdm(item: A) -> B:
                result = await func(item)
                tqdm_counter.update(1)
                return result

            return Slist(await asyncio.gather(*[func_with_tqdm(item) for item in self]))
        else:
            # todo: clean up branching
            return Slist(await asyncio.gather(*[func(item) for item in self]))

    else:
        assert max_par > 0, "max_par must be greater than 0"
        sema = asyncio.Semaphore(max_par)
        if tqdm:
            import tqdm as tqdm_module

            tqdm_counter = tqdm_module.tqdm(total=len(self), desc=desc)

            async def func_with_semaphore(item: A) -> B:
                async with sema:
                    result = await func(item)
                    tqdm_counter.update(1)
                    return result

        else:

            async def func_with_semaphore(item: A) -> B:
                async with sema:
                    return await func(item)

        result = await self.par_map_async(func_with_semaphore)
        return result


def apply():
    Slist.par_map_async = par_map_async

apply()