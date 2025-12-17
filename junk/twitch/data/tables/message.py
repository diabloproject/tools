import textwrap

import ydb


class MessageTable:
    @classmethod
    def create(cls, pool: ydb.QuerySessionPool):
        pool.execute_with_retries(
            textwrap.dedent(
                """
                CREATE TABLE IF NOT EXISTS `message`
                (
                    `id`        Int64,
                    `sender_id` Int64,
                    `text`      Utf8,
                    PRIMARY KEY (id) 
                );
                """,
            ),
        )
