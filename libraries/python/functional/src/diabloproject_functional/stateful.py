from typing import Callable


class StatefulOperator[S, R, T]:
    def __init__(self, f: Callable[[S, T], tuple[R, T]], state: T) -> None:
        self._callable = f
        self._state: T = state

    def __call__(self, value: S) -> R:
        result, self._state = self._callable(value, self._state)
        return result

def stateful[S, R, T](f: Callable[[S, T], tuple[R, T]], initial: T = None) -> Callable[[S], R]:
    return StatefulOperator[S, R, T](f, initial)
