from diabloproject_functional import pipe, effect, stateful,

numbers = set()

my = pipe[int, int](
    lambda x: x + 1,
    lambda x: str(x)
)

a = my(1)

# Render graph
p1 = pipe[int, int](
    lambda x: x + 1,
)

p2 = pipe[int, int](
    lambda x: x + 1,
    route(p1),
    lambda x: x ** 2,
    merge([p2], lambda x, p2_res: x + p2_res),
    lambda x: x
)
