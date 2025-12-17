import asyncio
import ydb
import logging


from data.context import DatabaseContext

APP_ID = 'd17dx6p4htihr7jfyofxnug0bf632'
APP_SECRET = 'qzts6nthsqn33vw32k2zqgq8h7cdj7'
# TARGET_CHANNEL = 'lcolonq'
TARGET_CHANNEL = 'diabloproject'

async def main():
    endpoint = "grpc://localhost:2135"  # your ydb endpoint
    database = "/Root/database-minikube-sample"  # your ydb database

    with ydb.Driver(
        driver_config=ydb.DriverConfig(
            disable_discovery=True,
            endpoint=endpoint,
            database=database,
        )
    ) as driver:
        with ydb.QuerySessionPool(driver) as pool:
            with DatabaseContext(pool):
                result = pool.execute_with_retries("SELECT * FROM message;")
                print(result[0].rows)


if __name__ == "__main__":
    asyncio.run(main())