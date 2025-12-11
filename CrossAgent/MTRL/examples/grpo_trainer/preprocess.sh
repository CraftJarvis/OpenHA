
ray stop --force
export RAY_TMPDIR='/DATA/hkc/tmp/ray'
export RAY_DEBUG='legacy'
#export MINESTUDIO_DIR='~/tmp/minestudio'
#!/bin/bash

ps -u hkc -o pid=,comm= \
  | grep -E 'java|Xvfb|xvfb-run' \
  | awk '{print $1}' \
  | grep -v '^970734$' \
  | xargs -r kill -9


# 删除并重建 ~/tmp/ray 目录
# rm -rf ~/tmp/ray
# mkdir -p ~/tmp/ray

# 1. 提高文件描述符限制
ulimit -n 1048576

# 2. 环境变量优化（影响 raylet/dashboard）
export RAY_EVENT_LOGGING_ENABLED=0
export RAY_PROFILING_ENABLED=0
export RAY_worker_logs_max_files=2
export RAY_worker_logs_max_file_size=1048576
export RAY_task_events_max_num=1000

# 3. 启动 Ray head 节点--address=172.16.0.14:6379
ray start --head --resources='{"minecraft_env": 16}' --object-spilling-directory /DATA/hkc/tmp/ray --temp-dir /DATA/hkc/tmp/ray