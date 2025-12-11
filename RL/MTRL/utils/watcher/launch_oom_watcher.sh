python watch_oom.py --proc-regex "vllm|ray::" \
  --interval 5 \
  --log oom_watch.csv \
  --proc-mem-thresh-mb 300000 \
  --sys-mem-thresh 92 \
  --gpu-mem-thresh-pct 99 \
  --disk-path /public \
  --disk-thresh 92 \
  --capture-dmesg
