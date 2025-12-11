import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from tqdm import tqdm
from agent_system.multi_turn_rollout.inference_batcher import InferenceBatcher, _Req, select_env_obs
import time
import random
import ray

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None and type(obs_text) == str:
            obs_content += obs_text
        elif type(obs_text) == list:
            obs_content = np.array(obs_text)
        else:
            print(f"Warning: No text observation found!")

        if "content" in obs_content[0]:
            chat = (obs_content)#np.array?
        else:
            chat = np.array([{
                "content": obs_content,
                "role": "user",
            }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            #add_generation_prompt=True,
            tokenize=False
        )

        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')

            #hkc: can image be list??
            if type(obs_image) == list:
                row_dict['multi_modal_data'] = {'image': [process_image(img) for img in obs_image]}
            else:   
                row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            
            if len(row_dict['multi_modal_data']['image']) == 0:
                breakpoint()
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        seq_lengths = attention_mask.sum(dim=1)  # shape: [batch]
        #print("真实每个样本的长度:", seq_lengths.tolist())
        #print("input_ids's shape:", input_ids.shape)

        if is_multi_modal:

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(raw_prompt, add_special_tokens=False),
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format

        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )

        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        # Initial observations from the environment

        obs, infos = envs.reset()
        # Initialize trajectory collection
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        

        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop

        actor_rollout_wg.enter_sharding_manager()
        for _step in tqdm(range(self.config.env.max_steps)):
            import time
            print("Step:", _step)
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            batch_input.meta_info = gen_batch.meta_info

            batch_output = actor_rollout_wg.generate_sequences(batch_input)

            batch.non_tensor_batch["uid"] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            breakpoint()
            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=False)
            next_obs, rewards, dones, infos = envs.step(text_actions)

            # maintain length_per_rewards for each environment
            length_per_rewards = (_step+1.0)*rewards
            # 初始化 action 分布
            action_dist = np.zeros((batch_size, 4), dtype=np.int32)  #Motion+Grounding,  Motion, Grounding, Action
            for i, ta in enumerate(text_actions):
                thought = ta["thought"]
                # 统计出现次数（只扫一次）
                motion_count = thought.count("Motion:")
                grounding_count = thought.count("Grounding:")
                action_count = thought.count("Action:")
                if motion_count > 1 or grounding_count > 1 or action_count > 1:
                    rewards[i] -= self.config.env.format_penalty  # penalty for multiple actions
                    continue

                # 分类计数（只用布尔值判断，不再重复扫字符串）
                has_motion = motion_count > 0
                has_grounding = grounding_count > 0
                has_action = action_count > 0

                if has_motion and has_grounding:
                    action_dist[i][0] = 1
                elif has_motion:
                    action_dist[i][1] = 1
                elif has_grounding:
                    action_dist[i][2] = 1
                elif has_action:
                    action_dist[i][3] = 1
                else:
                    rewards[i] -= 0.1  # penalty for no action



            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            # Create reward tensor, only assign rewards for active environments
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            batch.non_tensor_batch["length_per_rewards"] = length_per_rewards
            batch.non_tensor_batch["action_dist"] = action_dist
            #batch.non_tensor_batch["length_per_rewards"] = length_per_rewards.copy()
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        actor_rollout_wg.exit_sharding_manager()
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid
    

    def async_dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: "EnvironmentManagerBase",   # 需支持 reset_one / step_one
        ):
        """
        异步 + 动态批量（含异步 reset）版本，产物字段与 vanilla_multi_turn_loop 对齐：
        - 每个 env 独立 reset，不互相等待；
        - reset 完成的 env 立刻进入推理与 step；
        - 推理端用 InferenceBatcher 合并成大 batch 调 vLLM；
        - total_batch_list[i][t] 中附带：
            uid, traj_uid, rewards, active_masks, length_per_rewards, action_dist, is_action_valid, meta_info 等。
        返回：
            total_batch_list, episode_rewards, episode_lengths, success, traj_uid
        """

        import uuid, time
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        import numpy as np

        # ========= 工具函数 =========
        def _decode_action(tokenizer, resp):
            """更鲁棒地从推理结果中取出文本动作；兼容 str / list[int] / dict / tensor."""
            # 1) 直接字符串
            if isinstance(resp, str):
                return resp
            # 2) vLLM/OpenAI 风格对象
            if isinstance(resp, dict):
                # 常见字段尝试
                for key in ("text", "output_text", "response_text", "generated_text"):
                    if isinstance(resp.get(key), str):
                        return resp[key]
                # 递归处理 list 容器字段
                for k in ("responses", "choices", "outputs"):
                    if k in resp and isinstance(resp[k], list) and resp[k]:
                        return _decode_action(tokenizer, resp[k][0])
            # 3) token id 列表 / ndarray
            if isinstance(resp, (list, np.ndarray)):
                if len(resp) > 0 and isinstance(resp[0], (int, np.integer)):
                    return tokenizer.decode(resp, skip_special_tokens=False)
                if len(resp) > 0 and isinstance(resp[0], (list, np.ndarray)):
                    return tokenizer.decode(resp[0], skip_special_tokens=False)
            # 4) tensor 类型（处理 [256] 的 tensor）
            if isinstance(resp, torch.Tensor):
                resp = resp.detach().cpu().numpy()  # 转为 numpy 数组
                if len(resp.shape) > 1 and resp.shape[0] > 1:
                    breakpoint()
                resp = resp.flatten()
                return tokenizer.decode(resp, skip_special_tokens=False)

            raise ValueError(f"Unsupported response format: {type(resp)}")


        def _extract_thought(text_or_dict):
            """
            同步版本在 text_actions[i]['thought'] 上统计 Motion/Grounding/Action。
            这里兼容两种情况：
            - 若是 dict 且含 'thought' 则直接返回；
            - 若是 str，则直接用整段文本（用 .count() 统计关键片段）。
            """
            if isinstance(text_or_dict, dict) and "thought" in text_or_dict:
                return str(text_or_dict["thought"])
            return str(text_or_dict)

        def ppl_mask_frame(batches, mask_ratio=0.2):
            """
            给视频的每帧（batches[i]）计算 PPL，并标记 ppl-低 的帧。
            
            Args:
                batches: list of frame dicts, len = #frames (e.g., 13)
                mask_ratio: 比如 0.2 → mask ppl 最低的前 20% 的帧
            
            Returns:
                batches (in-place updated), 其中每个 batch 增加 key:
                    "ppl_mask_frame": bool
            """

            # 1. 计算每帧的 ppl
            frame_ppls = []
            for i, frame in enumerate(batches):
                lp = frame["rollout_log_probs"]  # shape [256]、
                valid = (lp != -1)
                ppl = torch.exp(-lp[valid].mean())
                frame_ppls.append((i, ppl.item()))

            # 2. 根据 ppl 排序（从小到大）
            frame_ppls.sort(key=lambda x: x[1])  # (frame_idx, ppl)

            # 3. 选前若干帧作为 mask
            num_frames = len(batches)
            num_mask = max(1, int(num_frames * mask_ratio))

            mask_frames = set([idx for idx, _ in frame_ppls[:num_mask]])

            # 4. 写入标记
            for i in range(num_frames):
                batches[i]["ppl_mask_frame"] = (i in mask_frames)

            return batches

        # ========= 初始化 =========
        # 从 env 侧初始 obs：异步版本逐一 reset_one，因此这里只确定 env 个数
        if hasattr(envs, "num_envs"):
            num_envs = envs.num_envs * self.config.env.rollout.n if getattr(self.config.env.rollout, "n", 0) > 0 else envs.num_envs
        else:
            num_envs = len(gen_batch.batch["input_ids"]) * self.config.env.rollout.n if getattr(self.config.env.rollout, "n", 0) > 0 else envs.num_envs

        # —— 根据同步版的做法，必要时对 gen_batch 做 repeat（与 obs 数量/分组一致）
        # 异步初始时未拿到 obs（逐个 reset），以 num_envs 对齐
        if getattr(self.config.env.rollout, "n", 0) > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == num_envs, \
            f"gen_batch size {len(gen_batch.batch)} does not match env size {num_envs}"

        batch_size = num_envs
        # —— uid / traj_uid（与同步版一致）
        if getattr(self.config.env.rollout, "n", 0) > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)

        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

        # —— 统计容器（与同步版一致）
        total_batch_list: list[list[dict]] = [[] for _ in range(batch_size)]
        total_infos: list[list[dict]] = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        is_done = np.zeros(batch_size, dtype=bool)

        # ========= 推理批处理器 =========
        # batcher = InferenceBatcher(
        #     actor_rollout_wg=actor_rollout_wg,
        #     preprocess_fn=self.preprocess_batch,
        #     tokenizer=self.tokenizer,
        #     base_gen_batch=gen_batch,
        #     max_batch_size=getattr(self.config.env, "max_infer_batch_size", 8),
        #     max_wait=getattr(self.config.env, "max_infer_wait_s", 0.015),
        # )
        num_batchers = len(actor_rollout_wg.workers)  # 每个 GPU 一个
        batchers = [
            InferenceBatcher(
                actor_rollout_wg=actor_rollout_wg,
                preprocess_fn=self.preprocess_batch,
                tokenizer=self.tokenizer,
                base_gen_batch=gen_batch,
                max_batch_size=getattr(self.config.env, "max_infer_batch_size", 8),
                max_wait=getattr(self.config.env, "max_infer_wait_s", 0.015),
                worker_group_target=i  # 关键：调度给指定 worker
            )
            for i in range(num_batchers)
        ]

        # ========= 异步 reset =========
        executor = ThreadPoolExecutor(max_workers=min(batch_size, getattr(self.config.env, "reset_max_workers", 8)))
        reset_futures = {}  # env_id -> Future

        def _reset_job(eid: int):
            ret = envs.reset_one(eid)  # -> (obs_i, info_i)
            return ret

        for env_id in range(batch_size):
            reset_futures[env_id] = executor.submit(_reset_job, env_id)

        # ========= 状态表 =========
        # pending: env_id -> req（等待推理完成）
        pending: dict[int, _Req] = {}
        
        
        actor_rollout_wg.enter_sharding_manager()

        # 用 tqdm 总量不预设，实时增长避免误导
        pbar = tqdm(desc="async rollout", unit="step", total=0)
        total_batch_list_flat_num = 0
        breakflagtime = -999
        # ========= 主循环 =========
        while True:
            any_event = False

            # 1) 处理“reset 完成”的环境：立刻提交首条推理请求（enter）
            finished_resets = []
            for env_id, fut in list(reset_futures.items()):
                
                if fut.done():
                    print("Resetting_Done: ", env_id)
                    try:
                        obs_i, info_i = fut.result()
                    except Exception:
                        print(f"Exception during env {env_id} reset; retrying...")
                        time.sleep(random.randint(1,10))
                        reset_futures[env_id] = executor.submit(envs.reset_one, env_id)
                        continue


                    # 提交到 batcher：附加 meta（对齐同步版）
                    req = batchers[env_id%num_batchers].submit(
                        env_id=env_id,
                        obs=obs_i,
                        gen_idx=env_id,
                        meta={
                            "uid": uid_batch[env_id],
                            "traj_uid": traj_uid[env_id],
                            "meta_info": gen_batch.meta_info,
                        },
                        # 可选：pop keys 与同步版一致的裁剪由 preprocess_fn 内部完成
                    )
                    pending[env_id] = req
                    finished_resets.append(env_id)
                    any_event = True
            
            for env_id in finished_resets:
                reset_futures.pop(env_id, None)

            # 2) 处理“推理完成”的环境：做 step_one，并按需提交下一步
            finished_envs = []
            for env_id, req in list(pending.items()):
                if req.done_event.is_set():
                    finished_envs.append(env_id)

            for env_id in finished_envs:
                req = pending.pop(env_id)

                try:
                    out = req.result  # 期望为 dict
                    if out is None:
                        raise RuntimeError("Inference result is None")

                    # —— 动作解码（与同步版统一用 tokenizer.decode → 文本 / 或结构体）
                    try:
                        resp = out.batch['responses']
                        text_action = _decode_action(self.tokenizer, resp)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        breakpoint()        
                    # —— 环境单步
                    next_obs, reward, done, info = envs.step_one(env_id, text_action)

                    # —— 与同步版一致的统计/惩罚/分布
                    # 统计 active_masks：仅当前 env（1 个）为 True, 只用到了env_id这一项
                    active_masks = np.zeros(batch_size, dtype=bool)
                    active_masks[env_id] = not is_done[env_id]

                    # 计算 action_dist 与格式惩罚
                    action_dist = np.zeros((batch_size, 4), dtype=np.int32)  # [Motion+Ground, Motion, Ground, Action]
                    thought_text = _extract_thought(text_action)  # 支持 str 或 dict
                    motion_count = thought_text.count("Motion:")
                    grounding_count = thought_text.count("Grounding:")
                    action_count = thought_text.count("Action:")

                    # 多重片段惩罚（仅对当前 env）
                    if motion_count > 1 or grounding_count > 1 or action_count > 1:
                        try:
                            reward = float(reward) - float(getattr(self.config.env, "format_penalty", 0.0))
                        except Exception:
                            pass

                    has_motion = motion_count > 0
                    has_grounding = grounding_count > 0
                    has_action = action_count > 0
                    if has_motion and has_grounding:
                        action_dist[env_id, 0] = 1
                    elif has_motion:
                        action_dist[env_id, 1] = 1
                    elif has_grounding:
                        action_dist[env_id, 2] = 1
                    elif has_action:
                        action_dist[env_id, 3] = 1
                    else:
                        # 无任何片段：轻惩罚
                        try:
                            reward = float(reward) - 0.1
                        except Exception:
                            pass

                    # squeeze 一下 env 返回
                    if getattr(reward, "shape", None) is not None and len(getattr(reward, "shape")) == 2:
                        reward = np.squeeze(reward, 1)
                    if getattr(done, "shape", None) is not None and len(getattr(done, "shape")) == 2:
                        done = np.squeeze(done, 1)

                    # numpy 标量化
                    reward_float = float(np.squeeze(reward)) if np.ndim(reward) else float(reward)
                    done_bool = bool(np.squeeze(done)) if np.ndim(done) else bool(done)

                    # 维护 episode 累积
                    if not is_done[env_id]:
                        episode_rewards[env_id] += reward_float
                        episode_lengths[env_id] += 1
                        pbar.total = pbar.n + 1  # 动态增量
                        pbar.update(1)

                    # length_per_rewards：与同步版一致，用“已计步数+1 * reward”
                    # 注意异步里每个 env 的步数不同，用该 env 的 episode_lengths 计算
                    length_per_rewards = (float(episode_lengths[env_id])) * reward_float

                    # is_action_valid：从 info 抽取
                    if isinstance(info, dict) and "is_action_valid" in info:
                        is_action_valid = info["is_action_valid"]
                    else:
                        is_action_valid = True

                    def _to_singleton_field(val):
                        return np.array([val], dtype=object)

                    ntb = out.non_tensor_batch
                    ntb["uid"] = _to_singleton_field(uid_batch[env_id])
                    ntb["traj_uid"] = _to_singleton_field(traj_uid[env_id])
                    ntb["rewards"] = _to_singleton_field(reward_float)
                    ntb["active_masks"] = _to_singleton_field(active_masks[env_id])
                    ntb["length_per_rewards"] = _to_singleton_field(length_per_rewards)
                    ntb["action_dist"] = _to_singleton_field(action_dist[env_id].copy())
                    ntb["is_action_valid"] = _to_singleton_field(is_action_valid)


                    batch_list = to_list_of_dict(out)
                    total_batch_list[env_id].append(batch_list[0])
                    total_infos[env_id].append(info)

                    if done_bool or episode_lengths[env_id] >= self.config.env.max_steps:
                        is_done[env_id] = True
                    else:
                        # 继续推理下一步
                        req2 = batchers[env_id%num_batchers].submit(
                            env_id=env_id,
                            obs=next_obs,
                            gen_idx=env_id,
                            meta={
                                "uid": uid_batch[env_id],
                                "traj_uid": traj_uid[env_id],
                                "meta_info": gen_batch.meta_info,
                            },
                        )
                        pending[env_id] = req2

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    is_done[env_id] = True
                    total_infos[env_id].append({"error": str(e), "stage": "inference/step"})


            finished_ratio = sum(is_done) / batch_size


            total_batch_list_flat = [item for sublist in total_batch_list for item in sublist]
            assert len(total_batch_list_flat) >= total_batch_list_flat_num, "total_batch_list_flat_num should not decrease"
            
            if len(total_batch_list_flat) > total_batch_list_flat_num + 40: #
                print("finished_ratio:", finished_ratio, "pending:", len(pending), "reset_futures:", len(reset_futures), "total_batch_list_flat_num:", total_batch_list_flat_num)
                breakflagtime = time.time()
                total_batch_list_flat_num = len(total_batch_list_flat)

            #if time.time() - breakflagtime > getattr(self.config.env, "rollout_done_wait", 10) and breakflagtime > 0 and finished_ratio > 0.8: #10s没增长40条, 0.5完成就跳出
            if (time.time() - breakflagtime)*finished_ratio > getattr(self.config.env, "rollout_done_wait", 20) and breakflagtime > 0 and finished_ratio > 0.5:
                is_done[:] = True
                print("Rollout breaked due to env finished ratio.")
                break

            # 3) 退出条件：所有 env 都 done，且无 pending，无未处理的 reset
            if is_done.all() and (not pending) and (not reset_futures):
                print("All environments are done.")
                break

            if not any_event:
                time.sleep(0.001)

        # ========= 收尾 =========

        if self.config.trainer.ppl_mask_frame_ratio:
            for sublist in total_batch_list:
                ppl_mask_frame(sublist, self.config.trainer.ppl_mask_frame_ratio)

        if pbar.n < pbar.total:
            pbar.total = pbar.n
            pbar.refresh()
            
        pbar.close()
        for batcher in batchers:
            batcher.stop()

        # try:
        #     assert total_batch_list[12][14]["uid"] == total_batch_list[15][2]["uid"]
        #     assert total_infos[12][14]['task_name'] == total_infos[15][2]['task_name']
        #     assert total_batch_list[12][14]['uid'] != total_batch_list[11][2]['uid']
        #     assert total_infos[12][14]['task_name'] != total_infos[11][2]['task_name']
        # except:
        #     breakpoint()
            
        envs.envs.shuffle_tasks()
        executor.shutdown(wait=True, cancel_futures=True)
        actor_rollout_wg.exit_sharding_manager()

        # ========= success 计算（与同步版保持一致的输入） =========
        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        print("rollout_finished, env_nums:", batch_size, "max_steps:", self.config.env.max_steps)
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, finished_ratio






    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        #len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and 
        while try_count < max_try_count:
            
            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, finished_ratio = self.async_dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            
            if self.config.algorithm.filter_groups.enable:
                batch_list, episode_rewards, episode_lengths, success, traj_uid = filter_group_data(batch_list=batch_list,
                                                                                                    episode_rewards=episode_rewards, 
                                                                                                    episode_lengths=episode_lengths, 
                                                                                                    success=success, 
                                                                                                    traj_uid=traj_uid, 
                                                                                                    config=self.config,
                                                                                                    last_try=(try_count == max_try_count),
                                                                                                    )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, finished_ratio

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Initial observations from the environment
        print("1111")
        if (self.config.algorithm.dynamic_rollouts or self.config.algorithm.filter_groups.enable) and is_train:
            print('2222')
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, finished_ratio = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            print('3333')
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        print("4444")
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )

        gen_batch_output.meta_info['avg_length'] = np.mean(total_episode_lengths)
        gen_batch_output.meta_info['traj_level_batch_size'] = len(total_batch_list)
        gen_batch_output.meta_info['step_level_batch_size'] = gen_batch_output.batch['input_ids'].shape[0]
        
        return gen_batch_output, finished_ratio