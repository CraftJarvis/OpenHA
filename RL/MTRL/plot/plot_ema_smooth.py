import matplotlib.pyplot as plt
import numpy as np
from swanlab import OpenApi

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

        last_row_key = 0.0  # 防止第一次就遇到 <0
        for step, row in df.iterrows():
            v = row[key]
            if v < 0:
                v = last_row_key
            else:
                last_row_key = v
            merged[int(step)] = float(v)

    return sorted(merged.items(), key=lambda x: x[0])

def plot_metrics_with_ema_and_std(run_names, key, smooth=False, alpha=0.1, max_steps=80):
    """
    输入多个 run_name，获取每个 run 的 metric 并画出折线图。
    - 原始曲线：同色浅色、无 label
    - 平滑曲线：使用 EMA，深色、有 label
    - 在平滑曲线的最高点画一条横线，并标注该 reward 值
    """
    plt.figure(figsize=(10, 6))

    # 按顺序从 matplotlib 的颜色循环里取颜色
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, run_name in enumerate(run_names):
        exp_ids = RUN_NAME_TO_EXP_ID.get(run_name)
        if exp_ids is None:
            print(f"Run name '{run_name}' is not valid!")
            continue

        res = fetch_merged_metrics(exp_ids=exp_ids, key=key)
        if not res:
            continue

        steps, values = zip(*res)
        steps = np.array(steps)
        values = np.array(values, dtype=float)

        # 只画前 max_steps 个点（跟你原来 :40 的逻辑一致）
        steps_plot = steps[:max_steps]
        values_plot = values[:max_steps]

        # 这条 run 的主色
        color = color_cycle[idx % len(color_cycle)]

        # 1. 原始曲线：同色 + 浅一点 + 无 label
        plt.plot(
            steps_plot,
            values_plot,
            linestyle='-',
            color=color,
            alpha=0.3,   # 透明度调浅
        )

        if smooth:
            # 2. EMA 平滑
            ema_values = [values[0]]
            for i in range(1, len(values)):
                ema_values.append(alpha * values[i] + (1 - alpha) * ema_values[-1])
            ema_values = np.array(ema_values)
            ema_plot = ema_values[:max_steps]

            # 主曲线：深色 + 有 label
            plt.plot(
                steps_plot,
                ema_plot,
                linestyle='-',
                color=color,
                label=f'{RUN_NAME_TO_LABLE[run_name]}'
            )

            # 3. 找到平滑曲线最高点，画横线并标注 reward
            y_max = float(ema_plot.max())
            # 横线范围：从这条曲线的第一个 step 到最后一个 step
            plt.hlines(
                y_max,
                xmin=steps_plot[0],
                xmax=steps_plot[-1]+3,
                colors=color,
                linestyles=':',
                alpha=0.8
            )
            # 在横线右端标注 reward 值
            plt.text(
                steps_plot[-1]+3,
                y_max,
                f'{y_max:.3f}',
                color=color,
                fontsize=14,
                va='bottom',
                ha='right'
            )

    plt.title('Ablation on Single-Turn RL Stage', fontsize=20)
    plt.xlabel('Training Step', fontsize=18)
    plt.ylabel('Average Success Rate', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    #plt.grid(True)
    plt.legend(fontsize=14)

    output_path =f'output/ablation_coa.png'
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {output_path}")


# 调用函数，比较多个 run 并绘制原始曲线和EMA平滑曲线
run_names = ["mix_coa_251103", "mix_coa"]
key = "episode/reward/mean"  # 选择你需要比较的metric键

#for run_name in RUN_NAME_TO_EXP_ID.keys():
plot_metrics_with_ema_and_std(run_names=run_names, key=key, smooth=True, alpha=0.2)
