import json

with open("OpenHA/rl/data_processor/task_suc_rate.json", "r", encoding="utf-8") as f:
    data = json.load(f)

data = data["mc-mix-coa-qwen2-vl-7b-250906_120"]
# 取出内部数据
inner_data = data

# 需要处理的前缀
prefixes = ["mine_block", "craft_item", "kill_entity"]

task_list = []

for prefix in prefixes:
    # 筛选出该前缀的任务
    sub_tasks = {
        k: v for k, v in inner_data.items()
        if k.startswith(prefix)
    }
    # 按成功率排序并取前10
    top10 = sorted(sub_tasks.items(), key=lambda x: x[1]["task_suc_nums"], reverse=True)[:10]
    for task, stats in top10:
        task_list.append(task)

# 保存到 JSON 文件
with open("task_list.json", "w", encoding="utf-8") as f:
    json.dump({"task_list": task_list}, f, ensure_ascii=False, indent=4)

print("已保存到 task_list.json")
