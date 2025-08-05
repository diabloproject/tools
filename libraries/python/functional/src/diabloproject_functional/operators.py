from typing import Callable
import pickle


dt = lambda x: x

class Operator:
    pass


def dfmapredc[S, I1, I2, R](
    array: list[S],
    c1: Callable[[S], Picklable],  # compile
    c2: Callable[[Picklable], S],  # decompile
    c3: Callable[[S], int],  # key
    c4: Callable[[S], I1],  # pre-map
    c5: Callable[[list[I1]], list[I2]],  # reduce
    c6: Callable[[I2], R],  # post-map
    c7: Callable[[R], Pickable],  # compile
    c8: Callable[[Pickable], R]  # decompile
) -> list[R]:
    array = list(map(c2, pickle.loads(pickle.dumps(list(map(c1, array))))))
    d: dict[int, list[S]] = {}
    keys_to_objects = list(map(lambda x: (c3(x), x), array))
    for key, item in keys_to_objects:
        if key not in d:
            d[key] = []
        d[key].append(item)
    after_premap = dict(map(lambda x: (x[0], list(map(c4, x[1]))), d.items()))
    after_reduce = dict(map(lambda x: (x[0], c5(x[1])), after_premap.items()))
    after_postmap = dict(map(lambda x: (x[0], list(map(c6, x[1]))), after_reduce.items()))
    # Merge
    picklable = dict(map(lambda x: (x[0], list(map(lambda x: pickle.dumps(c7(x)), x[1]))), after_postmap.items()))
    p = dict(map(lambda x: (x[0], list(map(lambda x: c8(pickle.loads(x)), x[1]))), picklable.items()))
    final = []
    for value in p.values():
        final.extend(value)
    return final


dfmapredc([1, 2, 3], dt, dt, hash, lambda x: x, lambda arr: arr, dt, dt, dt)
