"""Create the S3/MinIO bucket for uploads if it does not exist yet.

Idempotent: safe to run on every stack start. Needs network access to S3_ENDPOINT,
so run it in a container attached to the MinIO network (e.g. the ``server`` service).

Usage:
    python app/scripts/ensure_bucket.py
"""

from __future__ import annotations

import asyncio
import logging

from aiobotocore.session import get_session
from botocore.exceptions import BotoCoreError, ClientError

from config import StorageSection, process_section
from infrastructure.storage.file_storage import s3_client_kwargs


logger = logging.getLogger("default")
ATTEMPTS = 10
RETRY_DELAY_SECONDS = 3
MISSING_BUCKET_CODES = frozenset({"404", "NoSuchBucket", "NotFound"})


async def ensure_bucket(storage: StorageSection) -> None:
    bucket = storage.s3_bucket
    async with get_session().create_client("s3", **s3_client_kwargs(storage)) as client:
        # MinIO may still be starting right after `docker compose up -d`.
        for attempt in range(1, ATTEMPTS + 1):
            try:
                await client.head_bucket(Bucket=bucket)
            except ClientError as error:
                if error.response.get("Error", {}).get("Code") not in MISSING_BUCKET_CODES:
                    raise

                await client.create_bucket(Bucket=bucket)
                logger.info("Bucket %s created", bucket)
                return
            except BotoCoreError:
                if attempt == ATTEMPTS:
                    raise

                logger.info("S3 endpoint is not reachable yet, retry %d/%d", attempt, ATTEMPTS)
                await asyncio.sleep(RETRY_DELAY_SECONDS)
                continue
            logger.info("Bucket %s already exists", bucket)
            return


def main() -> None:
    asyncio.run(ensure_bucket(process_section(StorageSection)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
