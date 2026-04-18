import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op
from verl.tools.pfam.pipline_new import AgentRuntime

# ===================== helpers: 文本清洗 / requirement 抽取 =====================

REQ_SENTINEL = "The following text is the design requirement you must satisfy for this conversation."
_NUMBERED_LINE_RE = re.compile(r"\d+\.")  # 匹配 "1." / "2." 等

# 形如 ", 'role': 'user' ...]" 这种尾巴垃圾
_JUNK_TAIL_RE = re.compile(
    r"""
    (                       # capture 从这里开始到行尾/串尾
      ,\s*['"]role['"]      # , 'role' 或 ,"role"
      \s*:\s*
      ['"][^'"]*['"]        # 'user' / "user" 等
      .*                    # 后面啥都砍
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _slugify(s: str, maxlen: int = 64) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    if len(s) > maxlen:
        s = s[:maxlen]
    return s or "unknown"


def _hash8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def _cleanup_requirement_trailing_junk(s: str) -> str:
    """
    对已经抽到的编号块做一次“尾巴清理”：
    - 把像 ", 'role': 'user'}]" 这种 JSON/日志残渣删掉
    - 再顺手去掉末尾多余的 ] } 逗号空格
    """
    if not isinstance(s, str) or not s:
        return s

    # 有些时候是 "\\n" 字面量而不是真实换行，这里一并修一下
    if "\\n" in s and "\n" not in s:
        s = s.replace("\\n", "\n")

    s = _JUNK_TAIL_RE.sub("", s)
    s = s.rstrip("]} ,")
    return s.strip()


def _normalize_newlines(text: str) -> str:
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    return text


def _extract_numbered_block_from_start(text: str) -> str:
    """
    已知 text 是 sentinel 后的部分：
    从第一行出现 1. / 2. 的地方开始，
    一直到“最后一行仍是编号行”的地方结束。
    """
    if not text:
        return ""

    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")

    lines = text.splitlines()
    start = None
    end = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if _NUMBERED_LINE_RE.search(stripped):
            if start is None:
                start = i
            end = i

    if start is None:
        return ""
    if end is None:
        end = len(lines) - 1

    return "\n".join(lines[start : end + 1]).strip()


def _extract_numbered_block_from_end(text: str) -> str:
    """
    在全文尾部从后往前找带编号的行，
    找到最后一段 1. / 2. ... 的块。
    """
    if not text:
        return ""

    text = _normalize_newlines(text)
    lines = text.splitlines()
    start = None
    end = None
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if _NUMBERED_LINE_RE.match(stripped) or _NUMBERED_LINE_RE.search(lines[i]):
            if start is None:
                start = i
            end = i

    if start is None:
        return ""
    if end is None:
        end = len(lines) - 1

    return "\n".join(lines[start : end + 1]).strip()


def extract_requirement_from_messages(raw_prompt: List[Dict[str, Any]]) -> str:
    """
    从 verl 传下来的 raw_prompt（一个 messages list）里抽 requirement。

    逻辑：
    1) 找最后一个 user 的 content，当成全量 prompt 文本 text。
    2) 先找 sentinel:
       "The following text is the design requirement you must satisfy for this conversation."
       如果存在，就截取它后面的部分，再在这段里找编号块作为 requirement。
    3) 如果没有 sentinel，就在整个 text 的“末尾”找编号块。
    4) 都失败，就退回：
       - 有 sentinel：返回 sentinel 之后的整段
       - 没 sentinel：返回全文（兜底）
    """
    if not isinstance(raw_prompt, list):
        return ""

    user_text = ""
    for m in reversed(raw_prompt):
        if isinstance(m, dict) and m.get("role") == "user":
            user_text = m.get("content") or ""
            break

    if not isinstance(user_text, str) or not user_text.strip():
        return ""

    text = user_text.strip()

    lower = text.lower()
    sentinel_lower = REQ_SENTINEL.lower()
    idx = lower.rfind(sentinel_lower)
    tail_after_sentinel: Optional[str] = None

    # 2) sentinel 优先
    if idx != -1:
        tail_after_sentinel = text[idx + len(REQ_SENTINEL) :].strip()
        numbered = _extract_numbered_block_from_start(tail_after_sentinel)
        if numbered:
            numbered = _cleanup_requirement_trailing_junk(numbered)
            return numbered

    # 3) 全文尾部找编号块
    numbered_global = _extract_numbered_block_from_end(text)
    if numbered_global:
        return numbered_global

    # 4) 兜底
    if tail_after_sentinel is not None:
        return tail_after_sentinel
    return text


# 在 protein_tools.py 顶部附近
def cleanup_session_tmp(session_key: str, *, silent: bool = True):
    """
    强制删除某个 session_key 对应的 tmp 目录，并从 sessions 里移除。
    不依赖 refcnt/cleanup_on_release。
    """
    global _GLOBAL_SESSION_MANAGER
    mgr = _GLOBAL_SESSION_MANAGER
    if mgr is None:
        return

    # 尝试从已知 session 里拿 work_dir
    sess = mgr.sessions.pop(session_key, None)
    if sess is not None:
        work_dir = sess.work_dir
    else:
        # fallback：根据 session_key 推导目录名
        work_dir = mgr._stable_dir_for(session_key)

    if isinstance(work_dir, str) and os.path.isdir(work_dir):
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
            if not silent:
                print(f"[ProteinDesignSessionManager] cleaned work_dir={work_dir}")
        except Exception as e:
            if not silent:
                print(f"[ProteinDesignSessionManager][WARN] failed to cleanup {work_dir}: {e}")


# ===================== session manager =====================

@dataclass
class _Session:
    task_id: str
    work_dir: str
    runtime: AgentRuntime
    reward: float = 0.0
    last_obs: str = ""
    refcnt: int = 0
    meta: dict = field(default_factory=dict)


class ProteinDesignSessionManager:
    """
    同一 episode（对话） -> 同一个确定性目录:
      <base_tmp_root>/episode_<slug>-<hash8>

    所有工具调用共享同一个 AgentRuntime（基于 session_key）。
    """

    def __init__(
        self,
        base_tmp_root: str,
        protrek_env_python: str,
        default_topk: int,
        deterministic_session_dir: bool = True,
        cleanup_on_release: bool = False,
    ):
        self.base_tmp_root = base_tmp_root
        os.makedirs(self.base_tmp_root, exist_ok=True)

        self.protrek_env_python = protrek_env_python
        self.default_topk = default_topk
        self.sessions: Dict[str, _Session] = {}
        self.deterministic_session_dir = deterministic_session_dir
        self.cleanup_on_release = cleanup_on_release

    def _stable_dir_for(self, session_key: str) -> str:
        slug = _slugify(session_key)
        h8 = _hash8(session_key)
        return os.path.join(self.base_tmp_root, f"episode_{slug}-{h8}")

    def create_or_get_session(self, session_key: str, requirement_text: Optional[str] = None) -> str:
        """
        requirement_text:
          - 第一次建 session 时，用来初始化 AgentRuntime.requirement_text
          - 如果 session 已存在但 runtime 里还没有 requirement_text，而这次传入了，则补上
        """
        if session_key in self.sessions:
            sess = self.sessions[session_key]
            sess.refcnt += 1

            if requirement_text:
                rt = sess.runtime
                if not getattr(rt, "requirement_text", None):
                    rt.requirement_text = requirement_text
                sess.meta.setdefault("requirement_text", requirement_text)

            return session_key

        work_dir = self._stable_dir_for(session_key)
        os.makedirs(work_dir, exist_ok=True)

        runtime = AgentRuntime(
            work_dir=work_dir,
            requirement_text=requirement_text,
            protrek_env_python=self.protrek_env_python,
        )
        runtime.default_topk = self.default_topk

        self.sessions[session_key] = _Session(
            task_id=session_key,
            work_dir=work_dir,
            runtime=runtime,
            refcnt=1,
            meta={"requirement_text": requirement_text} if requirement_text else {},
        )


        return session_key

    def get_session(self, session_key: str) -> _Session:
        return self.sessions[session_key]

    def store_reward(self, session_key: str, value: float):
        self.sessions[session_key].reward = value

    def add_obs(self, session_key: str, obs: str):
        self.sessions[session_key].last_obs = obs

    def calc_reward(self, session_key: str) -> float:
        return float(self.sessions[session_key].reward)

    def release_session(self, session_key: str):
        """
        工具实例释放时只减少 refcnt。
        真正的 session 生命周期由：
        - cleanup_session_tmp(session_key)
        或
        - cleanup_on_release=True 且 refcnt 归零
        来控制。
        """
        sess = self.sessions.get(session_key)
        if not sess:
            return

        # 避免负数
        sess.refcnt = max(sess.refcnt - 1, 0)

        # 默认 cleanup_on_release=False：什么都不做，保留 session + runtime
        if not self.cleanup_on_release or sess.refcnt > 0:
            return

        # 只有 cleanup_on_release=True 且 refcnt==0 时，才真的删 session + 目录
        work_dir = getattr(sess, "work_dir", None)
        self.sessions.pop(session_key, None)

        if isinstance(work_dir, str) and os.path.isdir(work_dir):
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
                print(f"[ProteinDesignSessionManager] cleaned work_dir={work_dir}")
            except Exception as e:
                print(f"[ProteinDesignSessionManager][WARN] failed to cleanup {work_dir}: {e}")


# 全局共享一个 session manager，确保所有 ProteinDesignTool 实例共用同一套 sessions
_GLOBAL_SESSION_MANAGER: Optional[ProteinDesignSessionManager] = None


# ===================== ProteinDesignTool =====================

class ProteinDesignTool(BaseTool):
    """
    - 同一 episode_id / conversation_id -> 同一个 session_key
    - 所有 ProteinDesignTool 实例共用 _GLOBAL_SESSION_MANAGER
    - 同一 session_key 下，所有工具调用共用同一个 AgentRuntime 和 tmp_dir
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

        # 标记：此工具愿意额外吃 raw_prompt
        self.accepts_raw_prompt = True

        global _GLOBAL_SESSION_MANAGER

        maybe_manager = config.get("session_manager")
        if maybe_manager is not None:
            self.session_manager: ProteinDesignSessionManager = maybe_manager
        else:
            if _GLOBAL_SESSION_MANAGER is None:
                _GLOBAL_SESSION_MANAGER = ProteinDesignSessionManager(
                    base_tmp_root=config.get("base_tmp_root", "/tmp/protein_design_tasks"),
                    protrek_env_python=config.get(
                        "protrek_env_python",
                        "/path/to/miniconda3/envs/protrek/bin/python",
                    ),
                    default_topk=config.get("default_topk", 5),
                    deterministic_session_dir=True,
                    cleanup_on_release=config.get("cleanup_on_release", False), 
                )
            self.session_manager = _GLOBAL_SESSION_MANAGER

        # 上游可指定一个字段作为会话键（例如 "trajectory_id"），目前用不到可以保留
        self.force_session_key_key: Optional[str] = config.get("force_session_key_key")

        # 默认强制扁平：不再进入 round 子目录
        self.force_flat_dir: bool = config.get("force_flat_dir", True)

        # instance_id -> session_key
        self._instance_session_key: Dict[str, str] = {}

        # 把 config 存一下，后面抽 requirement 时也用得上
        self._config = config

    # -------- requirement 抽取辅助 --------
    def _extract_requirement_text(self, **kwargs) -> Optional[str]:
        """
        尝试从 kwargs 或 config 里捞出当前 episode 的 requirement 文本。
        可以和 raw_prompt 抽取互补。
        """
        # 1) 优先从 kwargs
        for k in (
            "requirement",
            "requirement_text",
            "design_requirement",
            "task_description",
            "instruction",
            "prompt",
        ):
            v = kwargs.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # 2) 其次从 config（有些场景可能直接写死在 config 里）
        cfg_req = (
            self._config.get("requirement_text")
            or self._config.get("requirement")
            or self._config.get("design_requirement")
        )
        if isinstance(cfg_req, str) and cfg_req.strip():
            return cfg_req.strip()

        return None

    def _ensure_requirement_for_session(
        self,
        sess: _Session,
        raw_prompt: Optional[List[Dict[str, Any]]],
    ) -> None:
        """
        只在本 episode 第一次调用时，从 raw_prompt 里抽 requirement，一次性写进：
        - sess.meta["requirement_text"]
        - sess.runtime.requirement_text
        """
        if sess.meta.get("requirement_text") is not None:
            return
        if not raw_prompt:
            return

        requirement = extract_requirement_from_messages(raw_prompt)
        if not requirement:
            return

        sess.meta["requirement_text"] = requirement
        runtime = sess.runtime
        if hasattr(runtime, "requirement_text"):
            setattr(runtime, "requirement_text", requirement)

    # -------- 统一解析“对话级”会话键 --------
    def _resolve_session_key(self, instance_id: str, **kwargs) -> str:
        """
        会话级 session_key 解析顺序：

        1) 明确传入的 episode_id / conversation_id / trajectory_id / session_id
        2) requirement_text_for_session_key / requirement_text：用 requirement 的 hash 做稳定 key
        3) 兜底：退回 instance_id（旧行为）
        """
        # 1) 上游显式传的 ID
        for k in ("episode_id", "conversation_id", "trajectory_id", "session_id"):
            v = kwargs.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # 2) 用 requirement 文本生成稳定 hash 作为 session_key
        req = (
            kwargs.get("requirement_text_for_session_key")
            or kwargs.get("requirement_text")
        )
        if isinstance(req, str) and req.strip():
            return f"req-{_hash8(req.strip())}"

        # 3) 兜底：和以前一样，用 instance_id
        return instance_id

    # ==================== BaseTool 接口 ====================

    async def create(
        self,
        instance_id: Optional[str] = None,
        create_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[str, ToolResponse]:
        ...
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = dict(create_kwargs or {})

        # 兼容旧用法
        if "episode_id" in kwargs and "episode_id" not in create_kwargs:
            create_kwargs["episode_id"] = kwargs["episode_id"]
        if "conversation_id" in kwargs and "conversation_id" not in create_kwargs:
            create_kwargs["conversation_id"] = kwargs["conversation_id"]
        if "raw_prompt" in kwargs and "raw_prompt" not in create_kwargs:
            create_kwargs["raw_prompt"] = kwargs["raw_prompt"]

        # ===== 抽 requirement =====
        raw_prompt = create_kwargs.get("raw_prompt")
        requirement_text = None
        if raw_prompt:
            requirement_text = extract_requirement_from_messages(raw_prompt)
        if not requirement_text:
            requirement_text = self._extract_requirement_text(**create_kwargs)

        # 把 requirement_text 一起喂给 _resolve_session_key，方便它用 hash 做稳定 key
        resolve_kwargs = dict(create_kwargs)
        if requirement_text:
            resolve_kwargs.setdefault("requirement_text_for_session_key", requirement_text)

        # 用 episode_id / conversation_id / requirement_hash / instance_id 解析 session_key
        session_key = self._resolve_session_key(instance_id, **resolve_kwargs)
        print(f"[DEBUG] ProteinDesignTool.create session_key={session_key}")

        task_id = self.session_manager.create_or_get_session(
            session_key,
            requirement_text=requirement_text,
        )
        self._instance_session_key[instance_id] = task_id

        work_dir = self.session_manager.get_session(task_id).work_dir
        return instance_id, ToolResponse(
            text=f"[ProteinDesignTool] Session {task_id} ready at {work_dir}"
        )


    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> Tuple[ToolResponse, float, dict]:
        session_key = self._instance_session_key[instance_id]
        sess = self.session_manager.get_session(session_key)
        runtime = sess.runtime

        raw_prompt = kwargs.get("raw_prompt")
        if isinstance(raw_prompt, list):
            self._ensure_requirement_for_session(sess, raw_prompt)

        # 强制扁平：所有调用都在 episode 根目录
        work_dir_for_this_call = sess.work_dir

        original_runtime_dir = runtime.work_dir
        if work_dir_for_this_call != original_runtime_dir:
            runtime.work_dir = work_dir_for_this_call
        

        try:
            tool_call = {
                "id": f"call_{uuid4()}",
                "type": "function",
                "function": {
                    "name": self.name,
                    "arguments": parameters or {},
                },
            }

            result = runtime.run_tool_call(tool_call)

            obs_text = result.get("content", "")
            extra = result.get("extra", {})
            ok = result.get("ok", True)

            self.session_manager.add_obs(session_key, obs_text)

            # reward 暂时保留原逻辑；如果你以后想用 get_score 的分数做 reward，可以在这里改
            step_reward = 0.0
            if self.name == "similarity_score":
                ranked = extra.get("ranked", [])
                if ranked and isinstance(ranked, list):
                    top_score = ranked[0].get("score")
                    if isinstance(top_score, (int, float)):
                        step_reward = float(top_score)
                        self.session_manager.store_reward(session_key, step_reward)

            metrics = {"ok": ok, **{f"extra_{k}": v for k, v in extra.items()}}
            return ToolResponse(text=obs_text), step_reward, metrics

        finally:
            # 保守恢复：避免 runtime 停留在子目录
            if runtime.work_dir != sess.work_dir:
                runtime.work_dir = sess.work_dir

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        session_key = self._instance_session_key[instance_id]
        return self.session_manager.calc_reward(session_key)

    async def release(self, instance_id: str, **kwargs) -> None:
        session_key = self._instance_session_key.pop(instance_id, None)
        if session_key is None:
            return
        self.session_manager.release_session(session_key)
