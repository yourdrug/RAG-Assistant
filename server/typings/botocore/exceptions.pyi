from typing import Any

class BotoCoreError(Exception):
    def __init__(self, **kwargs: object) -> None: ...

class ClientError(Exception):
    response: dict[str, Any]
    operation_name: str
    def __init__(self, error_response: dict[str, Any], operation_name: str) -> None: ...
