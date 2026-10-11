"""Exact object-key extraction for S3 ingestion metadata."""

from domain.exceptions import ValidationError

S3_SOURCE_PREFIX = "s3://"


def object_key_from_source(source: str) -> str:
    """Remove only the scheme and bucket, preserving the object key verbatim."""
    if not source.startswith(S3_SOURCE_PREFIX):
        raise ValidationError("Ingestion source must be an S3 URI")
    bucket, separator, key = source[len(S3_SOURCE_PREFIX) :].partition("/")
    if not bucket or not separator or not key:
        raise ValidationError("Ingestion source must include a bucket and object key")
    return key
