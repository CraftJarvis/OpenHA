import ray

# Initialize Ray on the head node
ray.init(
    address="auto",  # Automatically connect to the Ray cluster
    runtime_env={
        "RAY_IGNORE_VERSION_MISMATCH": True,
        "env_vars": {
            "RAY_DEBUG": "legacy",
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1"
        }
    },
    num_cpus=None,  # Auto-detect number of CPUs
)
