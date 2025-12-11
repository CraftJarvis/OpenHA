import threading
import time
import uuid
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Any, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import torch
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from codetiming import Timer
from verl.utils.device import get_torch_device
from contextlib import contextmanager
from verl import DataProto
import random
import copy
from copy import deepcopy

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last

@dataclass
class _Req:
    env_id: int
    obs: Dict[str, Any]          # 单个 env 的 obs（与 preprocess_batch 的 obs 约定一致）
    gen_idx: int                 # 从 gen_batch 里抽取哪一条提示（通常=env_id）
    done_event: threading.Event  # 等待推理结果
    result: Any = None           # 推理结果（batch_output 的单项）

def select_env_obs(obs_batch: Dict[str, Any], env_id: int) -> Dict[str, Any]:
    """
    从向量化 obs（batch 版）中抽取单 env 的 obs 视图，交给 preprocess 使用。
    这里假设 obs_batch 的每个 key 的第 0 维是 env 维。
    """
    out = {}
    for k, v in obs_batch.items():
        if isinstance(v, list):
            out[k] = v[env_id]
        elif isinstance(v, np.ndarray):
            out[k] = v[env_id]
        else:
            # anchor 之类的复杂对象，通常是 list/tuple；这里也按 list 取
            try:
                out[k] = v[env_id]
            except Exception:
                out[k] = v  # 如果不是按 env 切的对象，直接透传
    return out


def dataproto_equal(dp1: DataProto, dp2: DataProto, verbose=True) -> bool:
    """
    Deep compare two DataProto objects for full equivalence.
    Returns True if exactly equal, False otherwise.
    """
    # 1. Compare meta_info
    if dp1.meta_info != dp2.meta_info:
        if verbose:
            print("❌ meta_info not equal")
        return False

    # 2. Compare batch (TensorDict)
    if (dp1.batch is None) ^ (dp2.batch is None):
        if verbose:
            print("❌ One batch is None, the other is not")
        return False

    if dp1.batch is not None:
        keys1 = set(dp1.batch.keys())
        keys2 = set(dp2.batch.keys())
        if keys1 != keys2:
            if verbose:
                print("❌ batch keys not equal:", keys1, keys2)
            return False
        for key in keys1:
            t1 = dp1.batch[key]
            t2 = dp2.batch[key]
            if not torch.equal(t1, t2):
                if verbose:
                    print(f"❌ batch tensor {key} not equal")
                return False

    # 3. Compare non_tensor_batch
    keys1 = set(dp1.non_tensor_batch.keys())
    keys2 = set(dp2.non_tensor_batch.keys())
    if keys1 != keys2:
        if verbose:
            print("❌ non_tensor_batch keys not equal:", keys1, keys2)
        return False
    for key in keys1:
        a1 = dp1.non_tensor_batch[key]
        a2 = dp2.non_tensor_batch[key]
        if not np.array_equal(a1, a2):
            if verbose:
                print(f"❌ non_tensor_batch array {key} not equal")
            return False

    if verbose:
        print("✅ DataProto objects are fully equal")

    return True


class InferenceBatcher:
    def __init__(self, actor_rollout_wg, preprocess_fn, tokenizer, base_gen_batch,
                 max_batch_size=8, max_wait=0.015, worker_group_target=-1):
        self.time_raw = {}
        self.actor = actor_rollout_wg
        self.preprocess_fn = preprocess_fn
        self.tokenizer = tokenizer
        self.base_gen_batch = base_gen_batch
        self.max_bs = max_batch_size
        self.max_wait = max_wait

        self.q: Queue[_Req] = Queue()
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

        self.reserved_basedata = {
        }
        self.first_copy = None
        self.worker_group_target = worker_group_target
    def stop(self):
        self._stop = True
        self._th.join(timeout=1.0)

    def submit(self, env_id: int, obs: Dict[str, Any], gen_idx: int, meta: Dict[str, Any]) -> _Req:
        ev = threading.Event()
        req = _Req(env_id=env_id, obs=obs, gen_idx=gen_idx, done_event=ev)
        self.q.put(req)
        return req

    def _slice_dataproto(self, dp, idx_list: List[int]):
        sub = dp.empty_like() if hasattr(dp, "empty_like") else dp.__class__()
        sub.meta_info = dict(dp.meta_info) if hasattr(dp, "meta_info") else {}
        sub.batch = {k: v[idx_list] if isinstance(v, (np.ndarray, torch.Tensor)) else v for k, v in dp.batch.items()}
        sub.non_tensor_batch = {k: [v[i] for i in idx_list] if isinstance(v, list) else np.array(v)[idx_list]
                                for k, v in dp.non_tensor_batch.items()}
        return sub


    def _combine_obs_list(self, obs_list):
        if not obs_list:
            return {}
        keys = set().union(*(o.keys() for o in obs_list))
        out = {}
        for k in keys:
            out[k] = [o.get(k) for o in obs_list]   # 只加一层 env 维
        return out

    def _loop(self):
        import ray
        try:
            while not self._stop:
                batch_reqs: List[_Req] = []

                try:
                    first = self.q.get(timeout=self.max_wait)
                    batch_reqs.append(first)
                except Empty:
                    continue

                start = time.time()
                while len(batch_reqs) < self.max_bs and (time.time() - start) < self.max_wait:
                    try:
                        batch_reqs.append(self.q.get_nowait())
                    except Empty:
                        break

                if len(batch_reqs) < 1:
                    assert 1==0

                print(f"[InferenceBatcher] Processed batch of size {len(batch_reqs)}, timing: {self.time_raw}")
                idx_list = [r.gen_idx for r in batch_reqs]
                if len(idx_list) in self.reserved_basedata:
                    sub_gen_batch = self.reserved_basedata[len(idx_list)]
                else:
                    sub_gen_batch = self._slice_dataproto(self.base_gen_batch, idx_list)
                    self.reserved_basedata[len(idx_list)] = sub_gen_batch
                obs_batch = self._combine_obs_list([r.obs for r in batch_reqs])
                batch_input = self.preprocess_fn(gen_batch=sub_gen_batch, obs=obs_batch)
                # pad_size = (self.max_bs - len(batch_reqs) % self.max_bs) % self.max_bs
                # if pad_size > 0:
                #     batch_input.padding(pad_size)
                
                batch_output_ref = self.actor._workers[self.worker_group_target%len(self.actor._workers)].actor_rollout_generate_sequences.remote(batch_input)
                batch_output = ray.get(batch_output_ref)

                print(f"INFERENCE... batchsize:{len(self.actor._workers)} worker_id:{self.worker_group_target%len(self.actor._workers)} Done")
                # batch_output = self._slice_dataproto(batch_output, list(range(len(batch_reqs))))
                for i, req in enumerate(batch_reqs):
                    req.result = self._slice_dataproto(batch_output, [i])
                    req.done_event.set()
                
        except Exception as e:
            breakpoint()
    
    # def _loop(self):
    #     pending_tasks = []   # [(ref, batch_reqs), ...]

    #     while not self._stop:
    #         batch_reqs: List[_Req] = []

    #         try:
    #             first = self.q.get(timeout=self.max_wait)
    #             batch_reqs.append(first)
    #         except Empty:
    #             continue

    #         start = time.time()
    #         while len(batch_reqs) < self.max_bs and (time.time() - start) < self.max_wait:
    #             try:
    #                 batch_reqs.append(self.q.get_nowait())
    #             except Empty:
    #                 break

    #         if len(batch_reqs) < 1:
    #             assert 1==0

    #         print(f"[InferenceBatcher] Processed batch of size {len(batch_reqs)}, timing: {self.time_raw}")
    #         idx_list = [r.gen_idx for r in batch_reqs]
    #         if len(idx_list) in self.reserved_basedata:
    #             sub_gen_batch = self.reserved_basedata[len(idx_list)]
    #         else:
    #             sub_gen_batch = self._slice_dataproto(self.base_gen_batch, idx_list)
    #             self.reserved_basedata[len(idx_list)] = sub_gen_batch
    #         obs_batch = self._combine_obs_list([r.obs for r in batch_reqs])
    #         batch_input = self.preprocess_fn(gen_batch=sub_gen_batch, obs=obs_batch)
    #         pad_size = (self.max_bs - len(batch_reqs) % self.max_bs) % self.max_bs
    #         if pad_size > 0:
    #             batch_input.padding(pad_size)

    #         # 3. 发起异步推理（不等待）
    #         ref = self.actor.generate_sequences(batch_input)   # 返回的是 ObjectRef
    #         pending_tasks.append((ref, batch_reqs))

    #         # 4. 检查是否有结果返回（非阻塞 ray.wait）
    #         ready, pending = ray.wait([r for r, _ in pending_tasks], num_returns=1, timeout=0)
    #         if ready:
    #             finished_ref = ready[0]
    #             # 找到对应batch
    #             for i, (r, batch_reqs) in enumerate(pending_tasks):
    #                 if r == finished_ref:
    #                     result = ray.get(finished_ref)
    #                     # 处理返回结果
    #                     batch_output = self._slice_dataproto(result, range(len(batch_reqs)))
    #                     for i, req in enumerate(batch_reqs):
    #                         req.result = self._slice_dataproto(batch_output, [i])
    #                         req.done_event.set()
    #                     del pending_tasks[i]
    #                     break

