"""Where paused and finished assessment runs are kept, chosen by SCANNER_CHECKPOINTS:

  memory                  in the process (default; lost on restart)
  sqlite:<path>           a file, for a laptop or a self-hosted container
  dynamodb:<table>        DynamoDB, with anything over 350 KB offloaded to S3 (SCANNER_BUCKET). On Lambda,
                          which keeps nothing between requests, a run may wait hours for a person.
"""
import os
from functools import cache

TTL_DAYS = 30


@cache
def saver():
    spec = os.environ.get("SCANNER_CHECKPOINTS", "memory")
    kind, _, where = spec.partition(":")
    if kind == "dynamodb":
        from langgraph_checkpoint_aws import DynamoDBSaver
        return DynamoDBSaver(table_name=where, region_name=os.environ.get("AWS_REGION", "eu-north-1"),
                             ttl_seconds=TTL_DAYS * 86400, enable_checkpoint_compression=True,
                             s3_offload_config={"bucket_name": os.environ["SCANNER_BUCKET"], "key_prefix": "checkpoints"})
    if kind == "sqlite":
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver(sqlite3.connect(where, check_same_thread=False))
    if kind == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()
    raise ValueError(f"unknown SCANNER_CHECKPOINTS={spec!r}")
