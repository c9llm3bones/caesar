import io
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import model as M

app = FastAPI(title="Protein Autoencoder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_SIZE = 5 * 1024 * 1024  # 5 MB


def _save_upload(upload: UploadFile) -> str:
    data = upload.file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "File too large (max 5 MB)")
    suffix = os.path.splitext(upload.filename or "")[-1] or ".pdb"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


@app.post("/api/reconstruct")
async def reconstruct(file: UploadFile = File(...)):
    path = _save_upload(file)
    try:
        data, L, orig_pdb = M.load_pdb(path)
    except Exception as e:
        raise HTTPException(422, f"Cannot parse PDB: {e}")
    finally:
        os.unlink(path)

    metrics, orig_pdb, recon_pdb, _ = M.reconstruct(data, L, orig_pdb)
    return JSONResponse({
        "metrics": metrics,
        "pdb_original": orig_pdb,
        "pdb_reconstructed": recon_pdb,
    })


@app.post("/api/interpolate")
async def interpolate(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
):
    path_a = _save_upload(file_a)
    path_b = _save_upload(file_b)
    try:
        data1, L1, _ = M.load_pdb(path_a)
        data2, L2, _ = M.load_pdb(path_b)
    except Exception as e:
        raise HTTPException(422, f"Cannot parse PDB: {e}")
    finally:
        os.unlink(path_a)
        os.unlink(path_b)

    alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
    pdbs = M.interpolate(data1, L1, data2, L2, alphas=alphas)
    return JSONResponse({
        "structures": [{"alpha": a, "pdb": p} for a, p in zip(alphas, pdbs)]
    })


@app.post("/api/encode")
async def encode(file: UploadFile = File(...)):
    path = _save_upload(file)
    try:
        data, L, _ = M.load_pdb(path)
    except Exception as e:
        raise HTTPException(422, f"Cannot parse PDB: {e}")
    finally:
        os.unlink(path)

    latent = M.encode(data, L)  # [L, 320]
    buf = io.BytesIO()
    np.savez_compressed(buf, latent=latent)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=latent.npz"},
    )


@app.post("/api/decode")
async def decode(file: UploadFile = File(...)):
    data = file.file.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(413, "File too large (max 5 MB)")
    try:
        npz = np.load(io.BytesIO(data))
        latent = npz["latent"].astype(np.float32)
    except Exception as e:
        raise HTTPException(422, f"Cannot read .npz: {e}")

    if latent.ndim != 2 or latent.shape[1] != 320:
        raise HTTPException(422, "latent must have shape [L, 320]")

    pdb_str = M.decode(latent)
    return Response(
        content=pdb_str.encode(),
        media_type="chemical/x-pdb",
        headers={"Content-Disposition": "attachment; filename=decoded.pdb"},
    )


STATIC_DIR = Path(__file__).resolve().parent / "static" / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
