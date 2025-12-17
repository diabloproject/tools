import asyncio

import ydb.aio

from data.tables.message import MessageTable


class DatabaseContext:
    def __init__(self, pool: ydb.aio.QuerySessionPool):
        self._pool = pool

    def __enter__(self):
        MessageTable.create(self._pool),
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass