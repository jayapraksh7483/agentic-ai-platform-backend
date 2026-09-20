"""Read at most the permitted upload size into application memory."""
from fastapi import HTTPException


async def read_upload(file, limit):
    parts, size = [], 0
    while True:
        chunk = await file.read(min(65536, limit + 1 - size))
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, "Upload exceeds the permitted size")
        parts.append(chunk)
    if not size:
        raise HTTPException(400, "Uploaded file is empty")
    return b"".join(parts)
