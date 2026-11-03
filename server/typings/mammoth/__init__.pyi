from collections.abc import Callable
from typing import BinaryIO

def convert_to_html(file: BinaryIO, *, convert_image: Callable[[object], dict[str, object]]) -> object: ...
