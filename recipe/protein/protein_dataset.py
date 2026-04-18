# my_datasets.py
# -*- coding: utf-8 -*-

import logging, json, re
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        return data_item

def _loads_json_maybe(s):
    if isinstance(s, str):
        s = s.strip()
        if s == "":
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return s

# 识别“工具观测”文本的保险正则
_TOOL_OBS_RE = re.compile(r"^\s*Tool call ['\w\-]+.*executed successfully", re.I)

def _restore_tool_calls_and_fix_roles(messages):
    """
    - 若 assistant.message 里的 tool_calls 是字符串，则 json.loads -> list/dict
    - 若上一条是工具调用（assistant 含 tool_calls），下一条被写成 user 且像工具观测文本，则改成 role='tool'
    - 若 message content 为 None，则转换为空字符串
    返回 Python 原生 list[dict]
    """
    # messages 可能是 JSON 字符串 / list / ndarray(object)
    msgs = _loads_json_maybe(messages) if isinstance(messages, str) else messages
    if isinstance(msgs, np.ndarray):
        msgs = msgs.tolist()
    if not isinstance(msgs, list):
        return messages  # 保底：维持原状

    # 0) 处理 None content -> 转换为空字符串
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("content") is None:
            m["content"] = ""
        if m.get("name") is None:
            m["name"] = ""
        if m.get("tool_call_id") is None:
            m["tool_call_id"] = ""

    # 1) 还原 tool_calls
    for m in msgs:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        tc = m.get("tool_calls", None)
        if isinstance(tc, str):
            obj = _loads_json_maybe(tc)
            # 允许存成 dict（单调用）或 list
            if isinstance(obj, dict):
                obj = [obj]
            m["tool_calls"] = obj

    # 2) 修正 role=tool（仅在显然是工具观测时）
    prev_was_tool_call = False
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        tcs = m.get("tool_calls")
        if isinstance(tcs, list) and tcs:  # 这是一个工具调用回合
            prev_was_tool_call = True
            continue
        # 下一条如果像工具观测却被标成 user，就修正为 tool
        if prev_was_tool_call and role not in ("assistant", "tool"):
            content = m.get("content") or ""
            if isinstance(content, str) and _TOOL_OBS_RE.match(content):
                m["role"] = "tool"
        prev_was_tool_call = False

    return msgs

def _restore_tools_field(val):
    """
    将 parquet 里的 tools 列恢复为 Python 对象：
    - 如果是 JSON 字符串 -> json.loads -> list/dict
    - 如果是 list/dict/ndarray -> 转成原生 list/dict
    - 其它或解析失败 -> 返回 None（表示不提供 tools）
    """
    # 处理 pandas 的 NaN
    try:
        import pandas as _pd
        if isinstance(val, float) and _pd.isna(val):
            return None
    except Exception:
        pass

    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return convert_nested_value_to_list_recursive(val)
    if isinstance(val, np.ndarray):
        return convert_nested_value_to_list_recursive(val.tolist())
    if isinstance(val, str):
        obj = _loads_json_maybe(val)
        if isinstance(obj, (list, dict)):
            return convert_nested_value_to_list_recursive(obj)
        return None
    return None


class ParquetJSONAdapter(Dataset):
    """
    与官方 MultiTurnSFTDataset 接口保持一致：
    dataset_cls(parquet_files, tokenizer, config)

    补丁点：
    - 读 parquet 后，先把 messages[*].tool_calls 从字符串还原为对象
    - 必要时把“工具观测”轮的 role 从 user 改为 tool
    - 其余：分词、mask、拼接逻辑完全照原版
    """

    def __init__(self, parquet_files: str | list[str], tokenizer, config=None):
        config = config or {}
        self.truncation = config.get("truncation", "error")
        self.max_length = config.get("max_length", 1024)
        multiturn_config = config.get("multiturn", {})
        self.messages_key = multiturn_config.get("messages_key", "messages")
        self.tools_key = multiturn_config.get("tools_key", "tools")
        self.enable_thinking_key = multiturn_config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy, pandas
            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        # 读 messages（逐条做还原与修正）
        raw_messages = self.dataframe[self.messages_key].apply(series_to_item).tolist()
        self.messages = [ _restore_tool_calls_and_fix_roles(m) for m in raw_messages ]

        # 读 tools（字符串 -> JSON -> Python 对象）
        if self.tools_key in self.dataframe.columns:
            raw_tools = self.dataframe[self.tools_key].tolist()
            self.tools = [_restore_tools_field(t) for t in raw_tools]
        else:
            self.tools = None


        # 读 enable_thinking（可选）
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

    def __len__(self):
        return len(self.messages)

    # ===== 以下 tokenize / 拼接 / 打 mask 逻辑与原版一致 =====

    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        enable_thinking: Optional[bool] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        if start_idx > 0:
            prev_applied_text = self.tokenizer.apply_chat_template(
                messages[:start_idx],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                tools=tools,
                **self.apply_chat_template_kwargs,
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self.tokenizer.apply_chat_template(
                    messages[:start_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                    tools=tools,
                    **self.apply_chat_template_kwargs,
                )
        else:
            prev_applied_text = ""

        cur_applied_text = self.tokenizer.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            tools=tools,
            **self.apply_chat_template_kwargs,
        )

        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text):]
            generation_prompt_tokens = self.tokenizer.encode(generation_prompt_text, add_special_tokens=False)
            _message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text_w_generation_prompt):],
                add_special_tokens=False,
            )
            message_tokens = generation_prompt_tokens + _message_tokens
            loss_mask = [0] * len(generation_prompt_tokens) + [1] * (len(message_tokens) - len(generation_prompt_tokens))
        else:
            message_tokens = self.tokenizer.encode(
                cur_applied_text[len(prev_applied_text):],
                add_special_tokens=False,
            )
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)
        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full_tokens_list = full_tokens.tolist()
        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logging.warning(
                f"Token mismatch detected! Full tokenization length: {len(full_tokens_list)}, "
                f"Concatenated tokens length: {len(concat_tokens)}. Using concatenated version."
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )
        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        messages = self.messages[item]
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        def sanitize_messages(msgs):
            if not isinstance(msgs, list):
                return msgs
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                if m.get("content") is None:
                    m["content"] = ""
                if m.get("name") is None:
                    m["name"] = ""
                if m.get("tool_call_id") is None:
                    m["tool_call_id"] = ""
                # Keep absent tool_calls absent; Llama-3.1 template rejects empty tool_calls.
                if "tool_calls" in m and m.get("tool_calls") is None:
                    m.pop("tool_calls", None)
            return msgs

        messages = sanitize_messages(messages)

        try:
            full_tokens = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                **self.apply_chat_template_kwargs,
            )
        except Exception as e:
            logging.error(
                f"Error applying chat template: {e}\nMessages: {messages}\nTools: {tools}\nEnable thinking: {enable_thinking}"
            )
            raise

        concat_tokens, concat_loss_mask, concat_attention_mask = [], [], []
        i = 0
        while i < len(messages):
            cur_messages = messages[i]
            if cur_messages["role"] == "assistant":
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, is_assistant=True, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            elif cur_messages["role"] == "tool":
                st, ed = i, i + 1
                while ed < len(messages) and messages[ed]["role"] == "tool":
                    ed += 1
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, st, ed, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i = ed
            elif cur_messages["role"] in ["user", "system"]:
                if cur_messages["role"] == "system" and i != 0:
                    raise ValueError("System message should be the first message")
                tokens, loss_mask, attention_mask = self._process_message_tokens(
                    messages, i, i + 1, enable_thinking=enable_thinking, tools=tools
                )
                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)
                i += 1
            else:
                raise ValueError(f"Unknown role: {cur_messages['role']}")

        input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
            full_tokens[0], concat_tokens, concat_loss_mask, concat_attention_mask
        )

        seq_len = input_ids.shape[0]
        if seq_len < self.max_length:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = torch.cat((input_ids, torch.full((self.max_length - seq_len,), pad_id, dtype=input_ids.dtype)))
            attention_mask = torch.cat((attention_mask, torch.zeros((self.max_length - seq_len,), dtype=attention_mask.dtype)))
            loss_mask = torch.cat((loss_mask, torch.zeros((self.max_length - seq_len,), dtype=loss_mask.dtype)))
        elif seq_len > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif self.truncation == "error":
                raise ValueError(f"{seq_len=} is larger than {self.max_length=}")
            else:
                raise ValueError(f"Unknown truncation method {self.truncation}")

        position_ids = torch.arange(len(input_ids), dtype=torch.long)
        position_ids = position_ids * attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
