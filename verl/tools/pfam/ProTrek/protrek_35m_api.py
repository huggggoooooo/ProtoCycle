# protrek_score_35m_api.py
import os
import json
import tempfile
import subprocess
import logging
import shutil
import asyncio
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="ProTrek 35M + ESM Constrain API (script wrapper)")

# --------------------------
#       ProTrek 部分
# --------------------------

class SequenceItem(BaseModel):
    accession: Optional[str] = None
    sequence: str

class ProtrekScoreRequest(BaseModel):
    text: str
    sequences: List[SequenceItem]
    topk: int = 0

# 并发控制：最多同时跑多少个 ProTrek 任务
MAX_PROTREK_CONCURRENCY = int(os.getenv("PROTREK_MAX_CONCURRENCY", "2"))
protrek_semaphore = asyncio.Semaphore(MAX_PROTREK_CONCURRENCY)


def _run_protrek_script(
    text: str,
    sequences: List[Dict[str, Optional[str]]],
    topk: int,
) -> List[Dict[str, Optional[str]]]:
    """
    在本机用 protrek_rank_seqs.py 脚本执行打分。
    使用临时 json 文件作为 --json / --out，结束后删除。
    """
    tmp_dir = tempfile.mkdtemp(prefix="protrek_api_")
    in_path = os.path.join(tmp_dir, "input.json")
    out_path = os.path.join(tmp_dir, "output.json")

    try:
        # 写入输入 json
        with open(in_path, "w", encoding="utf-8") as f_in:
            json.dump(sequences, f_in, ensure_ascii=False, indent=2)

        script_path = (
            "/path/to/ProtoCycle/"
            "verl/tools/pfam/ProTrek/caculate_similarity_text_seq_35M.py"
        )
        topk_val = topk if topk and topk > 0 else 0

        cmd = [
            "python",
            script_path,
            "--text", text,
            "--json", in_path,
            "--topk", str(topk_val),
            "--out", out_path,
            "--device", "cuda",   # 如果想在 CPU 上跑就改成 "cpu"
        ]

        logger.info("Running ProTrek script: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            logger.error(
                "ProTrek script failed (exit=%s)\nSTDOUT:\n%s\nSTDERR:\n%s",
                proc.returncode, proc.stdout, proc.stderr,
            )
            raise RuntimeError(f"ProTrek script failed with exit code {proc.returncode}")

        if not os.path.exists(out_path):
            raise RuntimeError(f"ProTrek script did not create output file: {out_path}")

        with open(out_path, "r", encoding="utf-8") as f_out:
            ranked_raw = json.load(f_out)

        # 与本地解析逻辑保持一致
        if isinstance(ranked_raw, dict):
            if "results" in ranked_raw and isinstance(ranked_raw["results"], list):
                ranked_entries = ranked_raw["results"]
            else:
                ranked_entries = [ranked_raw]
        elif isinstance(ranked_raw, list):
            ranked_entries = ranked_raw
        else:
            ranked_entries = []

        return ranked_entries

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to cleanup tmp_dir %s: %s", tmp_dir, e)


@app.post("/protrek_score_35m", response_class=JSONResponse)
async def protrek_score_35m(query: ProtrekScoreRequest):
    """
    输入：
    {
      "text": "...",
      "sequences": [{"accession": "...", "sequence": "..."}, ...],
      "topk": 10
    }

    输出：
    {
      "status": "success",
      "results": [
        {"sequence": "...", "accession": "...", "score": ...},
        ...
      ]
    }
    """
    if not query.sequences:
        raise HTTPException(status_code=400, detail="sequences is empty")

    text = query.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    seq_items: List[Dict[str, Optional[str]]] = []
    for it in query.sequences:
        seq_items.append({
            "accession": it.accession,
            "sequence": it.sequence,
        })

    async with protrek_semaphore:
        loop = asyncio.get_event_loop()
        try:
            ranked = await loop.run_in_executor(
                None, _run_protrek_script, text, seq_items, query.topk
            )
        except Exception as e:
            logger.exception("protrek_score_35m script wrapper failed")
            if os.getenv("DEBUG", "0") == "1":
                raise HTTPException(status_code=500, detail=str(e))
            raise HTTPException(status_code=500, detail="ProTrek scoring failed")

    return {
        "status": "success",
        "results": ranked,
    }


# --------------------------
#       ESM constrain 部分
# --------------------------

# esm_constrain.py 的路径（可以用环境变量覆盖）
ESM_CONSTRAIN_SCRIPT = os.getenv(
    "ESM_CONSTRAIN_SCRIPT",
    "/path/to/ProtoCycle/verl/tools/pfam/esm/esm_constrain.py",
)

DEFAULT_ESM_MODEL_DIR = (
    "/path/to/ProtoCycle/pfam/esm/"
    "models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"
)

# 并发控制：最多同时跑多少个 ESM 任务
MAX_ESM_CONCURRENCY = int(os.getenv("ESM_MAX_CONCURRENCY", "2"))
esm_semaphore = asyncio.Semaphore(MAX_ESM_CONCURRENCY)


class EsmConstrainRequest(BaseModel):
    # 就是 constraints.json 的内容（包含 sequence、motif、regex_allow 等）
    constraints: Dict[str, Any]
    model_dir: Optional[str] = None   # 可选覆盖 --model_dir


class EsmConstrainResponse(BaseModel):
    status: str
    result: Dict[str, Any]              # 原来 args.out 写出的 JSON
    updated_constraints: Dict[str, Any] # 写回后的完整 constraints（带 sequence_inpaint）


def _run_esm_constrain_script(
    constraints: Dict[str, Any],
    model_dir: Optional[str],
) -> (Dict[str, Any], Dict[str, Any]):
    """
    在服务器本地跑 esm_constrain.py：

    - 写一个临时 constraints.json
    - 调用: python esm_constrain.py --json constraints.json --out result.json [--model_dir xxx]
    - 读回 result.json（sequence / original_sequence / debug_info）
    - 再读回 constraints.json（此时已更新 sequence_inpaint）
    - 最后删除临时目录
    """
    tmp_dir = tempfile.mkdtemp(prefix="esm_constrain_api_")
    in_path = os.path.join(tmp_dir, "constraints.json")
    out_path = os.path.join(tmp_dir, "result.json")

    try:
        with open(in_path, "w", encoding="utf-8") as f_in:
            json.dump(constraints, f_in, ensure_ascii=False, indent=2)

        cmd = ["python", ESM_CONSTRAIN_SCRIPT, "--json", in_path, "--out", out_path]
        model_dir_effective = model_dir or DEFAULT_ESM_MODEL_DIR
        if model_dir_effective:
            cmd.extend(["--model_dir", model_dir_effective])

        logger.info("Running esm_constrain.py: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(
                "esm_constrain.py failed (exit=%s)\nSTDOUT:\n%s\nSTDERR:\n%s",
                proc.returncode, proc.stdout, proc.stderr,
            )
            raise RuntimeError(f"esm_constrain.py failed with exit code {proc.returncode}")

        if not os.path.exists(out_path):
            raise RuntimeError(f"esm_constrain.py did not create output file: {out_path}")

        # 读 result.json
        with open(out_path, "r", encoding="utf-8") as f_res:
            result_payload = json.load(f_res)

        # 读回更新后的 constraints.json
        with open(in_path, "r", encoding="utf-8") as f_cfg:
            updated_constraints = json.load(f_cfg)

        return result_payload, updated_constraints

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to cleanup tmp_dir %s: %s", tmp_dir, e)


@app.post("/esm_constrain", response_model=EsmConstrainResponse)
async def esm_constrain_endpoint(query: EsmConstrainRequest):
    """
    输入:
    {
      "constraints": { ... 原 constraints.json 的内容 ... },
      "model_dir": "/path/to/esm/checkpoint"   // 可选
    }

    输出:
    {
      "status": "success",
      "result": { "sequence": "...", "original_sequence": "...", "debug_info": {...} },
      "updated_constraints": { ... 带 sequence_inpaint 的完整 cfg ... }
    }
    """
    async with esm_semaphore:
        loop = asyncio.get_event_loop()
        try:
            result, updated_constraints = await loop.run_in_executor(
                None,
                _run_esm_constrain_script,
                query.constraints,
                query.model_dir,
            )
        except Exception as e:
            logger.exception("esm_constrain_endpoint failed")
            if os.getenv("DEBUG", "0") == "1":
                raise HTTPException(status_code=500, detail=str(e))
            raise HTTPException(status_code=500, detail="ESM constrain failed")

    return EsmConstrainResponse(
        status="success",
        result=result,
        updated_constraints=updated_constraints,
    )


# --------------------------
#          启动
# --------------------------
if __name__ == "__main__":
    port = int(os.getenv("PROTREK_PORT", "8863"))
    uvicorn.run(app, host="0.0.0.0", port=port)
