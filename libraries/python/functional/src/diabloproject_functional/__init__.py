from typing import overload, Callable, Self
from functools import reduce


class FunctionalPipe[S, R]:
    def __init__(self, *args):
        self.args = args

    def __call__(self, value: S) -> R:
        return reduce(lambda x, y: y(x), self.args, value)  # type: ignore

    def apply(self, value: list[S]) -> list[R]:
        return [self(v) for v in value]

    @overload
    @classmethod
    def pipe_(
        cls,
        f1: Callable[[S], R], /, **kwargs
    ) -> Self:
        ...

    @overload
    @classmethod
    def pipe_[I](
        cls,
        f1: Callable[[S], I],
        f2: Callable[[I], R], /, **kwargs
    ) -> Self:
        ...

    @overload
    @classmethod
    def pipe_[I1, I2](
        cls,
        f1: Callable[[S], I1],
        f2: Callable[[I1], I2],
        f3: Callable[[I2], R], /, **kwargs
    ) -> Self:
        ...

    @overload
    @classmethod
    def pipe_[I1, I2, I3](
        cls,
        f1: Callable[[S], I1],
        f2: Callable[[I1], I2],
        f3: Callable[[I2], I3],
        f4: Callable[[I3], R], /, **kwargs
    ) -> Self:
        ...

    @overload
    @classmethod
    def pipe_[I1, I2, I3, I4](
        cls,
        f1: Callable[[S], I1],
        f2: Callable[[I1], I2],
        f3: Callable[[I2], I3],
        f4: Callable[[I3], I4],
        f5: Callable[[I4], R], /, **kwargs
    ) -> Self:
        ...

    @overload
    @classmethod
    def pipe_[I1, I2, I3, I4, I5](
        cls,
        f1: Callable[[S], I1],
        f2: Callable[[I1], I2],
        f3: Callable[[I2], I3],
        f4: Callable[[I3], I4],
        f5: Callable[[I4], I5],
        f6: Callable[[I5], R], /, **kwargs
    ) -> Self:
        ...

    @overload
    @classmethod
    def pipe_[I1, I2, I3, I4, I5, I6](
        cls,
        f1: Callable[[S], I1],
        f2: Callable[[I1], I2],
        f3: Callable[[I2], I3],
        f4: Callable[[I3], I4],
        f5: Callable[[I4], I5],
        f6: Callable[[I5], I6],
        f7: Callable[[I6], R], /, **kwargs
    ) -> Self:
        ...

    @overload
    @classmethod
    def pipe_[I1, I2, I3, I4, I5, I6, I7](
        cls,
        f1: Callable[[S], I1],
        f2: Callable[[I1], I2],
        f3: Callable[[I2], I3],
        f4: Callable[[I3], I4],
        f5: Callable[[I4], I5],
        f6: Callable[[I5], I6],
        f7: Callable[[I6], I7],
        f8: Callable[[I7], R], /, **kwargs
    ) -> Self:
        ...

    @classmethod
    def pipe_(cls, *args, **kwargs):
        return cls(*args)

class PipeTypeHelper:
    def __getitem__[S, R](self, value: tuple[type[S], type[R]]):
        src_ty, res_ty = value
        return FunctionalPipe[src_ty, res_ty].pipe_

pipe = PipeTypeHelper()

__all__ = ["pipe", "FunctionalPipe"]
