import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from swanlab import OpenApi
import random
# Run name 到 exp_id 的映射
RUN_NAME_TO_EXP_ID = {
    "text_coa": ["zwgm2fna12u8lfflb7cb6", "nnh56yvjyctrloe2jkmv1"],
    "grounding_coa": ["iclfplwxumx439yoejop0"],
    "motion_coa": ["cu0mc8ylirbljj7e6qug7"],
    "grounding_no_his_coa": ["07r6mu9gdl6hc35vwkvu6"],
    "mix_coa_251103": ["dn30agnx74tx11nuzrvzr"],
    "mix_coa": ["b36n7jivh8mt2hizo48pm"]
}

RUN_NAME_TO_LABLE = {
    "text_coa": "TextHA",
    "grounding_coa": "GroundingHA",
    "motion_coa": "MotionHA",
    "grounding_no_his_coa": "GroundingHA",
    "mix_coa_251103": "CrossAgent",
    "mix_coa": "CrossAgent(w/o SSRL stage)"
}

def fetch_merged_metrics(exp_ids, key: str):
    """
    输入多个 exp_id，输出合并后的 metric：
    - 同一个 step，如重复出现，用后者覆盖前者
    - 最终返回 [(step, value), ...] 按 step 排序
    """
    api = OpenApi()
    merged = {}  # step -> value

    for exp_id in exp_ids:
        resp = api.get_metrics(exp_id=exp_id, keys=[key])
        df = resp.data

        if key not in df.columns:
            raise KeyError(f"Metric key '{key}' does not exist in exp {exp_id}.")

        # 逐条写入，后出现的覆盖前面
        for step, row in df.iterrows():
            if row[key]<0:
                row[key] = last_row_key
            last_row_key = row[key]
            merged[int(step)] = float(row[key])

    # 返回按 step 排序的列表
    return sorted(merged.items(), key=lambda x: x[0])

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from swanlab import OpenApi
import random

# 你之前的函数 fetch_merged_metrics 和 RUN_NAME_TO_EXP_ID 保持不变

def plot_metrics_with_smooth_and_std(run_names, key, smooth=False, window_length=39, polyorder=3):
    """
    输入多个 run_name，获取每个 run 的 metric 并画出折线图，同时显示标准差。
    - run_names: 包含多个 run name 的列表
    - key: metric 键（例如 "episode/reward/mean"）
    - smooth: 是否进行平滑处理
    - window_length: Savitzky-Golay滤波器的窗口大小
    - polyorder: Savitzky-Golay滤波器的多项式阶数
    """
    plt.figure(figsize=(10, 6))

    for run_name in run_names:
        # 获取 run_name 对应的 exp_id
        exp_ids = RUN_NAME_TO_EXP_ID.get(run_name)
        if exp_ids is None:
            print(f"Run name '{run_name}' is not valid!")
            continue

        # 获取合并的metrics
        res = fetch_merged_metrics(exp_ids=exp_ids, key=key)
        
        # 提取 step 和 value
        steps, values = zip(*res)

        if smooth:
            # 使用 Savitzky-Golay 滤波器平滑数据
            smoothed_values = savgol_filter(values, window_length=window_length, polyorder=polyorder)
            
            # 计算每个窗口的标准差
            std_values = []
            half_window = window_length // 2  # 窗口的一半，用来计算标准差

            for i in range(len(values)):
                start = max(i - half_window, 0)
                end = min(i + half_window + 1, len(values))
                window_values = values[start:end]
                std_values.append(np.std(window_values))  # 计算窗口内的标准差

            # 绘制平滑曲线
            plt.plot(steps[:40], smoothed_values[:40], linestyle='-', label=f'{RUN_NAME_TO_LABLE[run_name]}')

            # 绘制标准差区域（阴影）
            plt.fill_between(steps[:40], np.array(smoothed_values)[:40] - np.array(std_values)[:40], np.array(smoothed_values)[:40] + np.array(std_values)[:40], alpha=0.3)


    plt.title(f'Ablation on Single-Turn RL Stage', fontsize=20)
    plt.xlabel('Training Step', fontsize=18)
    plt.ylabel('Average Success Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    
    plt.grid(True)
    plt.legend(fontsize=14)

    # 保存图像
    output_path = 'output/merged_run_comparison_with_std.png'
    plt.savefig(output_path)
    plt.show()

    print(f"Plot saved to {output_path}")

# 调用函数，比较多个 run 并绘制标准差区域
run_names = ["mix_coa", 'mix_coa_251103']


key = "episode/reward/mean"  # 选择你需要比较的metric键
plot_metrics_with_smooth_and_std(run_names=run_names, key=key, smooth=True, window_length=7, polyorder=3)
