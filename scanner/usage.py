"""Daily cap on LLM runs (assessments and questions), so a public demo can't run up a bill.

A counter per UTC day in DynamoDB (SCANNER_USAGE_TABLE), raised atomically and refused at the cap
(SCANNER_RUNS_PER_DAY). Without a table, as on a laptop, there is no cap.
"""
import os
import time
from datetime import datetime, timezone
from functools import cache


class OverCap(Exception):
    pass


@cache
def _table():
    name = os.environ.get("SCANNER_USAGE_TABLE")
    if not name:
        return None
    import boto3
    return boto3.resource("dynamodb", region_name=os.environ.get("AWS_REGION", "eu-north-1")).Table(name)


def take(kind):
    """Count one run of `kind` ("assess" or "ask"); raise OverCap if today's cap is reached."""
    table = _table()
    if table is None:
        return
    cap = int(os.environ.get("SCANNER_RUNS_PER_DAY", "200"))
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from botocore.exceptions import ClientError
    try:
        table.update_item(
            Key={"PK": f"runs#{day}"},
            UpdateExpression="ADD runs :one, #k :one SET #ttl = if_not_exists(#ttl, :ttl)",
            ConditionExpression="attribute_not_exists(runs) OR runs < :cap",
            ExpressionAttributeNames={"#k": kind, "#ttl": "ttl"},
            ExpressionAttributeValues={":one": 1, ":cap": cap, ":ttl": int(time.time()) + 90 * 86400})
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            raise OverCap(f"The demo's daily limit of {cap} runs is used up. Try again tomorrow (UTC).")
        raise
