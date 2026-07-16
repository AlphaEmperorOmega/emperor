from __future__ import annotations

import asyncio
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, BinaryIO

from fastapi import APIRouter, Depends, Request

from emperor_workbench.api._blocking import (
    BLOCKING_WORK_TIMEOUT_MESSAGE,
    DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS,
    named_blocking_work_limiter,
)
from emperor_workbench.api._dependencies import (
    get_run_history_service,
    get_workbench_settings,
)
from emperor_workbench.api._errors import ApiError
from emperor_workbench.api._mutations import (
    HttpOperationPolicy,
    declare_http_operation,
    run_mutation_io,
)
from emperor_workbench.api.v1.run_history._archive_upload import (
    parse_multipart_log_archive_upload,
)
from emperor_workbench.api.v1.run_history._contracts import (
    LogArchiveImportResponse,
)
from emperor_workbench.api.v1.run_history._mapping import (
    log_archive_import_to_payload,
)
from emperor_workbench.run_history import RunHistoryService
from emperor_workbench.settings import WorkbenchApiSettings

router = APIRouter()
LOG_ARCHIVE_UPLOAD_MEMORY_SPOOL_SIZE = 1024 * 1024
LOG_ARCHIVE_UPLOAD_LIMITER_NAME = "log-archive-upload"


def _upload_too_large_error(limit: int) -> ApiError:
    return ApiError(
        f"Log archive upload exceeds the {limit} byte limit.",
        status_code=413,
    )


async def _read_upload_body_with_limit(
    request: Request,
    *,
    max_upload_size: int | None,
) -> BinaryIO:
    body = tempfile.SpooledTemporaryFile(
        max_size=LOG_ARCHIVE_UPLOAD_MEMORY_SPOOL_SIZE,
        mode="w+b",
    )
    total_size = 0
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="workbench-upload-spool",
    )

    async def run_file_call(callable_object, *args):
        future = executor.submit(callable_object, *args)
        while not future.done():
            await asyncio.sleep(0)
        return future.result()

    try:
        async for chunk in request.stream():
            total_size += len(chunk)
            if max_upload_size is not None and total_size > max_upload_size:
                raise _upload_too_large_error(max_upload_size)
            await run_file_call(body.write, chunk)
        await run_file_call(body.seek, 0)
        return body
    except BaseException:
        await run_file_call(body.close)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


@router.post(
    "/import",
    response_model=LogArchiveImportResponse,
    summary="Import log archive",
    response_description="Extracted log archive import summary.",
)
@declare_http_operation(HttpOperationPolicy.LOG_IMPORT)
async def import_log_archive(
    request: Request,
    service: Annotated[RunHistoryService, Depends(get_run_history_service)],
    settings: Annotated[WorkbenchApiSettings, Depends(get_workbench_settings)],
) -> LogArchiveImportResponse:
    max_upload_size = settings.effective_max_upload_size
    max_extracted_size = settings.effective_max_log_archive_extracted_size
    max_member_count = settings.max_log_archive_member_count
    max_path_bytes = settings.max_log_archive_path_bytes
    upload_concurrency = settings.log_archive_upload_concurrency
    content_type = request.headers.get("content-type", "")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            upload_size = int(content_length)
        except ValueError:
            upload_size = 0
        if max_upload_size is not None and upload_size > max_upload_size:
            raise _upload_too_large_error(max_upload_size)

    upload_limiter = named_blocking_work_limiter(
        f"{LOG_ARCHIVE_UPLOAD_LIMITER_NAME}:{upload_concurrency}",
        upload_concurrency,
    )
    try:
        await asyncio.wait_for(
            upload_limiter.acquire(),
            DEFAULT_BLOCKING_WORK_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        raise ApiError(
            BLOCKING_WORK_TIMEOUT_MESSAGE,
            status_code=503,
        ) from exc

    limiter_handed_to_worker = False
    try:
        body = await _read_upload_body_with_limit(
            request,
            max_upload_size=max_upload_size,
        )

        def parse_and_extract_archive():
            try:
                upload = parse_multipart_log_archive_upload(
                    content_type=content_type,
                    body=body,
                    max_upload_size=max_upload_size,
                )
                return service.import_archive(
                    archive=upload.content,
                    filename=upload.filename,
                    max_upload_size=max_upload_size,
                    max_extracted_size=max_extracted_size,
                    max_member_count=max_member_count,
                    max_path_bytes=max_path_bytes,
                )
            finally:
                body.close()

        limiter_handed_to_worker = True
        result = await run_mutation_io(
            parse_and_extract_archive,
            limiter=upload_limiter,
            limiter_already_acquired=True,
        )
        return LogArchiveImportResponse.model_validate(
            log_archive_import_to_payload(result)
        )
    finally:
        if not limiter_handed_to_worker:
            upload_limiter.release()


__all__ = ["router"]
