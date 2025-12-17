from typing import override


class RPath:
    segments: tuple[str, ...]

    def __init__(self, param) -> None:
        if isinstance(param, str):
            self.segments = tuple(param.split('/'))
        elif isinstance(param, list):
            for elem in param:
                if not isinstance(elem, str):
                    raise TypeError('Invalid type for RPath initialization: ')
            self.segments = tuple(param)
        elif isinstance(param, RPath):
            self.segments = param.segments
        else:
            raise TypeError('Invalid type for RPath initialization')

    @override
    def __str__(self) -> str:
        return '/'.join(self.segments)
