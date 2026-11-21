"""Storage key prefixes the application writes its own objects under.

Uploaded documents (with their owners' access), preview files and upload backups
share the deployment's bucket with content imported from it; importers must skip them.
"""

from __future__ import annotations

from application.documents.preview_cache import PREVIEW_PREFIX
from application.documents.utils import UPLOADS_PREFIX
from application.ingestion.targets import UPLOAD_BACKUPS_PREFIX


APPLICATION_STORAGE_PREFIXES = (UPLOADS_PREFIX, PREVIEW_PREFIX, UPLOAD_BACKUPS_PREFIX)
