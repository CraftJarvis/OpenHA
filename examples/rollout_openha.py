import argparse 
import ray
from tqdm import tqdm
import random 
import os 
import json 
from datetime import datetime
import logging
import torch
import uuid 
import time
from openagents.utils.render import render_video
from openagents.utils.file_op import save_render_videos, clip_action_info
from openagents.envs.tasks.task_manager import choose_available_task
from openagents.envs import DEFAULT_MAXIMUM_BUFFER_SIZE
from openagents.envs.env import env_init
from openagents.agents.openha import OpenHA

# Configure logging format and level
logging.basicConfig(level=logging.INFO)
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s')

def instruction_from_task(task:str) -> str:
    """Convert a task string into a natural instruction for the agent."""
    if ':' in task:
        obj_name = task.split(":")[-1]
        return f"Pinpoint the {obj_name} and track it."
    else:
        return task

def rollout_wrapper(args):
    """Wrapper for rollout function with error handling (useful for Ray parallel execution)."""
    try:
        rollout(args)
    except Exception as e:
        print(e)
    return args.task

# @ray.remote(num_gpus=0.5)
def rollout(args):
    """Main rollout loop for running an OpenHA agent in the Minecraft environment."""
    
    # Task setup and environment configuration
    tasks = args.task.split(",")
    task = random.choice([p.strip() for p in tasks])
    task_config = choose_available_task(task, difficulty=args.difficulty)
    task_name = task_config['task_name']
    model_id = args.model_id + "-" + args.output_mode
    max_buffer_size = DEFAULT_MAXIMUM_BUFFER_SIZE if args.num_rollouts > 10 else 1

    # Save directory setup
    rollout_path = os.path.join(
        args.record_path, 
        model_id, 
        task_name,
        datetime.now().strftime('%y-%m-%d-%H-%M-%S')+'-'+task_name.replace(":", "_")+'-'+str(uuid.uuid4()).replace('-', '')[:8])
    os.makedirs(rollout_path, exist_ok=True)
    raw_action_file_path = os.path.join(rollout_path, "raw_action.jsonl")
    
    # Determine segment type based on task name
    if "mine_block" in task_name:
        segment_type = "Mine"
    elif "kill_entity" in task_name:
        segment_type = "Hunt"
    elif "craft_item" in task_name:
        segment_type = "Craft"
    else:
        segment_type = "Explore"

    # Agent initialization with configuration
    model_url = f"http://{random.choice(args.model_ips.split(','))}:{random.choice(args.model_ports.split(','))}/v1"
    agent = OpenHA(
        output_mode = args.output_mode,
        raw_action_type = args.raw_action_type,
        vlm_client_mode = args.vlm_client_mode,
        model_path = args.model_path, 
        grounding_policy_path = args.grounding_policy_path,
        motion_policy_path = args.motion_policy_path,
        sam_path = args.sam_path,
        segment_type = segment_type,
        model_id = args.model_id, 
        model_url = model_url, 
        maximum_history_length=args.maximum_history_length, 
        action_chunk_len=args.action_chunk_len,
        instruction_type=args.instruction_type,
        LLM_backbone=args.LLM_backbone,
        VLM_backbone=args.VLM_backbone,
        tokenizer_path=args.tokenizer_path,
        
        system_message=args.system_message,
        system_message_tag = args.system_message_tag,
        enforce_format= args.enforce_format,
        enforce_prefix = args.enforce_prefix,
        temperature=args.temperature,
        top_k = args.top_k,
        top_p = args.top_p,
        grounding_inference_interval = args.grounding_inference_interval,
        motion_inference_interval = args.motion_inference_interval,
    )
    
    # Sleep randomly to avoid contention when launching multiple rollouts in parallel
    time.sleep(random.random())
    
    # Initialize environment
    env = env_init(task_config, rollout_path, args,)
    noop_action = env.noop_action()
    obs, reward, terminated, truncated, info = env.step(noop_action)
    
    # Reset the agent with task description
    agent.reset(instruction=task_config['task_description'], task_name = task_name, )
    
    # Rollout loop variables
    success_state = False
    reward = 0
    start_time = time.time()
    run_frame_idx = 0
    
    # Buffer for raw actions to save periodically
    processed_raw_actions = [{"raw_action": ""}]
    
    # Main environment-agent interaction loop
    for run_frame_idx in tqdm(range(args.max_steps_num)):
        # Query agent for next action
        action = agent.get_action(obs=obs, info=info,verbose=args.verbose)
        if agent.action_type == 'agent':
            action = env.agent_action_to_env_action(action)
        if action == None:
            action = env.action_space.sample()
        elif action == 'no_op':
            action = env.action_space.no_op()

        # Save action details for logging
        processed_raw_action = {"points": agent._points, "raw_action": agent._response, "action_type": agent._policy_type }
        processed_raw_actions.append(processed_raw_action)
        
        # Write actions to file when buffer is full
        if len(processed_raw_actions) % max_buffer_size == 0:
            with open(raw_action_file_path, 'a', encoding='utf-8') as f:
                for processed_raw_action in processed_raw_actions:
                    f.write(json.dumps(processed_raw_action, ensure_ascii=False) + '\n')
                    processed_raw_actions = []
            
        # Step environment with chosen action
        obs, reward, terminated, truncated, info = env.step(action)
        if reward > 0:
            success_state = True
            break
        
    # Report FPS
    print(f"FPS: {run_frame_idx/(time.time()-start_time)}")
    
    # Pause after success (to render final states more clearly)
    end_pause = 1
    if success_state:
        end_pause = random.randint(5, 20)
    for _ in range(end_pause): 
        time.sleep(0.1)
        processed_raw_actions.append({"raw_action":""})
        obs, reward, terminated, truncated, info = env.step(noop_action)
    
    # Save remaining actions
    with open(raw_action_file_path, 'a', encoding='utf-8') as f:
        for processed_raw_action in processed_raw_actions:
            f.write(json.dumps(processed_raw_action, ensure_ascii=False) + '\n')
    
    # Close environment
    env.close()
    
    # Write success/failure metadata
    if success_state:
        with open(os.path.join(rollout_path, 'success.json'), 'w') as f:
            json.dump({"success": True, "frames": run_frame_idx,"args": vars(args)}, f, indent=2)
    else:
        with open(os.path.join(rollout_path, 'loss.json'), 'w') as f:
            json.dump({"frames": run_frame_idx,"args": vars(args)}, f, indent=2 ,ensure_ascii=False)

    # Clip actions and render video
    clip_action_info(rollout_path, assert_video_action_num_same=True)
    renderred_frames = render_video(
        rollout_path, 
        enable_task_name=True, enable_rawaction=True, enable_thought=False, enable_envaction=True, enable_point_cot=False,
        raw_action_max_length = 5,
    )
    print(f"saving render video in {rollout_path} with {len(renderred_frames)} frames")
    save_render_videos(rollout_path, renderred_frames, fps=args.fps)
    return rollout_path

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_mode', type=str, default="greedy", choices=["greedy", "grounding", "motion", "text_action"],)
    parser.add_argument('--vlm_client_mode', type=str, default="online", choices=["online", "openai", "anthropic", "vllm", "lmdeploy", "hf", ],)
    parser.add_argument("--system_message_tag", type=str, default="text_action")
    parser.add_argument("--system_message", type=str, default = None)
    parser.add_argument("--model_ips", type=str, default="localhost")
    parser.add_argument("--model_ports", type=str, default="11000")
    parser.add_argument('--raw_action_type', type=str,default="text", choices=["reserved", "text"],)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="CraftJarvis/minecraft-openha-qwen2vl-7b-2509")
    parser.add_argument("--sam_path", type=str, default="facebook/sam2-hiera-base-plus")
    parser.add_argument("--grounding_policy_path", type=str, default="CraftJarvis/MineStudio_ROCKET-1.12w_EMA")
    parser.add_argument("--motion_policy_path", type=str, default="CraftJarvis/Minecraft-Motion_policy-2509")
    parser.add_argument("--grounding_inference_interval", type=int, default=4)
    parser.add_argument("--motion_inference_interval", type=int, default = 4)
    parser.add_argument('--action-chunk-len',type=int, default = 1)
    parser.add_argument('--maximum_history_length',type=int, default = 15)
    
    parser.add_argument('--LLM_backbone', type=str, default="")
    parser.add_argument('--VLM_backbone', type=str, default="")
    parser.add_argument('--tokenizer_path', type=str, default="")
    
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--record_path", type=str, required=True)
    
    parser.add_argument("--max_steps_num", type=int, default=400)
    parser.add_argument("--num_rollouts", type=int, default=1)
    
    parser.add_argument("--task", type=str, default="kill_entity:sheep") 
    parser.add_argument('--difficulty', type=str,default="zero")
    
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--fps',type=int,default=20)
    
    parser.add_argument("--enforce_format", type=bool, default = False)
    parser.add_argument("--enforce_prefix", type=str, default = "")
    parser.add_argument('--instruction-type',type=str, default='final')
    parser.add_argument('--temperature','-t',type=float, default=0.8)
    parser.add_argument('--top_p',type=float, default=0.99)
    parser.add_argument('--top_k',type=int, default=-1)
    parser.add_argument('--gpu_per_rollout',type=float, default=0.5)
    args = parser.parse_args()
    
    # Multi-rollout mode with Ray
    if args.num_rollouts > 1:
        args.num_rollouts = args.num_rollouts
        num_gpus = torch.cuda.device_count()
        logging.info(f"Available GPUs: {num_gpus}")

        # Initialize Ray with available GPUs
        ray.init(num_gpus=num_gpus, log_to_driver=True)
        # ray.init(address='auto')

        # Launch rollout tasks in parallel
        futures = [ray.remote(num_gpus=args.gpu_per_rollout)(rollout_wrapper).remote(args) for _ in range(args.num_rollouts)]
        print(f"Rollout futures: {len(futures)}")

        # Gather results
        results = ray.get(futures)
        for result in results:
            logging.info(result)

        # Shutdown Ray
        ray.shutdown()
    else:
        # Single rollout mode
        rollout(args)