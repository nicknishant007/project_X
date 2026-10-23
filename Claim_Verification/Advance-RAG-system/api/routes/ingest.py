import traceback

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse

from api.services.ingest_service import IngestService

router = APIRouter()


# ============================================================
# Upload Single Document
# ============================================================

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    # Validate extension
    if not IngestService.validate_extension(file.filename):

        raise HTTPException(

            status_code=400,

            detail={

                "message": "Unsupported file type.",

                "allowed_extensions":
                    IngestService.get_allowed_extensions()

            }

        )

    try:

        content = await file.read()

        if len(content) == 0:

            raise HTTPException(

                status_code=400,

                detail="Uploaded file is empty."

            )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to read uploaded file: {e}"

        )

    try:

        file_path = IngestService.save_uploaded_file(

            filename=file.filename,

            content=content,

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to save uploaded file: {e}"

        )

    try:

        result = IngestService.process_single_file(file_path)

        return JSONResponse(

            status_code=200,

            content={

                "status":
                    "skipped"
                    if result["skipped"]
                    else "success",

                "file":
                    result["file"],

                "chunks_created":
                    result["chunks_created"],

                "skipped":
                    result["skipped"],

                "message":
                    result["message"]

            }

        )

    except Exception as e:

        print(traceback.format_exc())

        raise HTTPException(

            status_code=500,

            detail=f"Ingestion failed: {e}"

        )


# ============================================================
# Run Complete Pipeline
# ============================================================

@router.post("/run")
async def run_pipeline():

    try:

        result = IngestService.run_full_pipeline()

        return JSONResponse(

            content={

                "status": "success",

                "total_chunks_created":
                    result["total_chunks_created"],

                "message":
                    result["message"]

            }

        )

    except Exception as e:

        print(traceback.format_exc())

        raise HTTPException(

            status_code=500,

            detail=f"Pipeline execution failed: {e}"

        )