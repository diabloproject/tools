from typing import Callable, Iterable, Protocol


class SupportsEq(Protocol):
    def __eq__(self, other: object) -> bool:
        ...


class ReduceOperator[T]:
    __opconf__ = {
        "streamable": False,
    }
    def __init__(self, f: Callable[[T, T], T], key_fn: Callable[[T], SupportsEq]) -> None:
        self._callable = f
        self._key_fn = key_fn

    def __call__(self, value: T) -> T:
        return value

    def __apply__(self, values: Iterable[T]) -> Iterable[T]:
        ...

def reduce[T](f: Callable[[T, T], T], key: Callable[[T], SupportsEq]) -> Callable[[T], T]:
    return ReduceOperator[T](f, key)
