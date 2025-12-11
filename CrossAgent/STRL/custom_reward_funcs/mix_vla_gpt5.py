import math
import torch
import ray
import re
import math
from openagents.agents.utils.action_mapping import TextActionTokenizer
import torch
from minestudio.simulator.entry import MinecraftSim
import random
import numpy as np

def parse_action(action_str):
    if "move_to" in action_str:
        # 提取 move_to 的坐标
        coords = re.findall(r'move_to\(([^)]+)\)', action_str)
        if coords:
            x, y = map(float, coords[0].split(','))
            return {'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 'forward': 0, 'back': 0, 'left': 0, 'right': 0, 'sprint': 0, 'sneak': 0, 'use': 0, 'drop': 0, 'attack': 0, 'jump': 0, 'inventory': 0, 'camera': np.array([x, y])}
    elif "click" in action_str:
        return {'hotbar.1': 0, 'hotbar.2': 0, 'hotbar.3': 0, 'hotbar.4': 0, 'hotbar.5': 0, 'hotbar.6': 0, 'hotbar.7': 0, 'hotbar.8': 0, 'hotbar.9': 0, 'forward': 0, 'back': 0, 'left': 0, 'right': 0, 'sprint': 0, 'sneak': 0, 'use': 0, 'drop': 0, 'attack': 1, 'jump': 0, 'inventory': 0, 'camera': np.array([0.0, 0.0])}


tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = TextActionTokenizer()
    return tokenizer

# # --- Camera score: 归一化到 [0,1]，含漏报惩罚 ---
# def camera_score_norm(
#     gt: torch.Tensor,     # (..., 2)
#     pred: torch.Tensor,   # (..., 2)
#     scales=(3.0, 4.5),
#     deadzone=0.05,        # 原始量纲
#     tau=0.15,             # 归一化 L2 的“成功半径”
#     k=10.0,               # logistic 坡度
#     fn_lambda=0.5         # 漏报惩罚权重
# ) -> torch.Tensor:
#     scales = torch.as_tensor(scales, dtype=torch.float32, device=gt.device)
#     e = (pred - gt) / scales
#     err = torch.norm(e, p=2, dim=-1)     # 归一化 L2

#     # 映射到 [0,1]：误差小 -> 分高
#     # 让 err=tau 时得分约 0.5，再随 k 调整陡峭度
#     score = torch.sigmoid((tau - err) * k)

#     # 漏报惩罚（gt 有明显动作，但 pred 落在死区）
#     gt_big   = (gt.abs()  >= deadzone)    # (..., 2) bool
#     pred_sml = (pred.abs() <= deadzone)   # (..., 2)
#     fn_pen   = (gt_big & pred_sml).to(torch.float32).sum(dim=-1) * fn_lambda

#     score = (score - fn_pen).clamp(0.0, 1.0)
#     return score  # tensor in [0,1]

# # --- 按钮分：只看被激活的键集合，Jaccard ---
# def button_score_jaccard(ga: dict, sa: dict) -> float:
#     keys = [k for k in ga.keys() if k != "camera" and not k.startswith("cursor")]
#     Sg = {k for k in keys if int(ga.get(k, 0)) == 1}
#     Sp = {k for k in keys if int(sa.get(k, 0)) == 1}
#     if not Sg and not Sp:
#         return 1.0
#     inter = len(Sg & Sp)
#     union = len(Sg | Sp)
#     return inter / union

# def safe_len_penalty(solution_action: dict, max_penalty=0.05) -> float:
#     """按动作键数量做轻微惩罚，最多扣 max_penalty。"""
#     n_keys_on = sum(int(v) == 1 for k, v in solution_action.items() if k not in ("camera", "cursor"))
#     # 例如每开启 10 个键扣 0.01，上限 0.05
#     return min(max_penalty, 0.01 * (n_keys_on / 10.0))







# ------- 辅助：带权 F1（按钮） -------
def button_score_weighted_f1(ga: dict, sa: dict, class_weights: dict[str, float]):
    keys = [k for k in ga.keys() if k != "camera" and not k.startswith("cursor")]
    # 逐键 0/1
    y = torch.tensor([int(ga.get(k, 0)) for k in keys], dtype=torch.float32)
    p = torch.tensor([int(sa.get(k, 0)) for k in keys], dtype=torch.float32)
    w = torch.tensor([class_weights.get(k, 1.0) for k in keys], dtype=torch.float32)

    tp = (p * y) * w
    fp = (p * (1 - y)) * w
    fn = ((1 - p) * y) * w

    tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(f1.clamp(0., 1.))

# ------- 相机：方向+幅值（平滑） -------
def camera_score_dir_mag(gt: torch.Tensor, pred: torch.Tensor,
                         deadzone=0.05, sigma_mag=0.25, w_dir=0.7):
    # gt / pred: shape (2,)  ; 先处理 deadzone
    if gt.abs().max().item() <= deadzone and pred.abs().max().item() <= deadzone:
        return 1.0  # 都几乎没动
    # 方向
    gnorm = gt.norm() + 1e-8
    pnorm = pred.norm() + 1e-8
    cos = torch.dot(gt/gnorm, pred/pnorm).clamp(-1, 1)   # [-1,1]
    s_dir = 0.5 * (cos + 1.0)                             # [0,1]

    # 幅值（高斯核）
    dmag = (pnorm - gnorm).abs()
    s_mag = torch.exp(- (dmag * dmag) / (2 * sigma_mag * sigma_mag))

    return float((w_dir * s_dir + (1 - w_dir) * s_mag).clamp(0., 1.))

# ------- 非法组合罚分 -------
def illegal_combo_penalty(sa: dict):
    k = 0.0
    if sa.get('forward',0)==1 and sa.get('back',0)==1: k += 0.2
    if sa.get('left',0)==1    and sa.get('right',0)==1: k += 0.2
    if sa.get('sprint',0)==1  and sa.get('sneak',0)==1: k += 0.2
    return k

def noop_penalty(ga: dict, sa: dict):
    has_gt = any(int(v)==1 for k,v in ga.items() if k!='camera')
    has_pred = any(int(v)==1 for k,v in sa.items() if k!='camera')
    return 0.15 if (has_gt and not has_pred) else 0.0

# ------- 稀有动作 bonus（可选，不依赖 GT） -------
def rarity_bonus(sa: dict, hist: dict[str,int] | None, beta=0.02):
    if not hist: return 0.0
    bonus = 0.0
    for k, v in sa.items():
        if k=='camera' or v!=1: continue
        c = max(0, int(hist.get(k, 0)))
        bonus += beta / math.sqrt(c + 1.0)
    # 上限，免得过大
    return min(0.1, bonus)

def first_action_tail_len(s: str) -> int:
    """返回字符串里第一个 'Action:' 之后的字符长度；没找到则返回 len。"""
    idx = s.find("Action:")
    if idx == -1:
        return len(s)
    return len(s) - (idx + len("Action:"))

def raw_action_length_penalty(data_source, solution_str):
    """按动作字符串长度做轻微惩罚，最多扣 0.1。"""
    data_source_len = first_action_tail_len(data_source)
    solution_len = first_action_tail_len(solution_str)
    len_diff = abs(solution_len - data_source_len)
    # 例如每多/少 50 字符扣 0.01，上限 0.2
    return min(0.2, 0.005 * len_diff)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from openagents.agents.utils.action_mapping import TextActionTokenizer
    tokenizer = TextActionTokenizer()

    length_penalty = raw_action_length_penalty(ground_truth, solution_str)

    try:
        ga  = tokenizer.decode(ground_truth)[0]
        sa  = tokenizer.decode(solution_str)[0]
    except Exception as e:
        return {"score": 0.0, "button_score": 0.0, "camera_score": 0.0, "penalty": 0.0}

    # 1) 按钮：加权 F1 （给稀有键更大权重）
    # 频次可离线统计；没有统计时用静态先验
    class_w = {
        "attack": 3.0, "use": 2.0,
        "forward": 1.0, "back": 1.0, "left": 1.0, "right": 1.0,
        "jump": 2.0, "sneak": 1.0, "sprint": 1.0, "inventory": 1.0,
        "drop": 1.0
    }
    btn = button_score_weighted_f1(ga, sa, class_w)

    # 2) 相机：方向+幅值，保留 deadzone
    cam = 0.0
    if "camera" in ga and "camera" in sa:
        gt   = torch.tensor(ga["camera"], dtype=torch.float32)
        pred = torch.tensor(sa["camera"], dtype=torch.float32)
        cam  = camera_score_dir_mag(gt, pred, deadzone=0.05, sigma_mag=0.25, w_dir=0.8)

    # 3) 组合惩罚
    pen_illegal = illegal_combo_penalty(sa)
    pen_noop    = noop_penalty(ga, sa)
    # 可保留你的轻量“多开键”惩罚，但幅度小一点
    n_on = sum(int(v)==1 for k,v in sa.items() if k!='camera')
    pen_len = min(0.10, 0.01 * n_on)

    # 4) 稀有动作 bonus（可选）
    hist = (extra_info or {}).get("rollout_action_hist", None)  # 由外部维护
    bonus = rarity_bonus(sa, hist, beta=0.02)

    # 5) 融合（按钮主导）
    w_btn, w_cam = 0.75, 0.25
    raw = w_btn*btn + w_cam*cam - (pen_illegal + pen_noop + pen_len) + bonus - length_penalty
    # 不硬裁剪，返回原始并额外给裁剪版
    clipped = float(max(0.0, min(1.0, raw)))

    return {
        "score": raw,            # 如你当前管线需要 [0,1]
        #"raw_score": float(raw),     # 建议训练端做 batch z-norm 用这个
        "button_score": float(btn),
        "camera_score": float(cam),
        "length_penalty": float(length_penalty),
        "penalty": float(pen_illegal + pen_noop + pen_len),
        "bonus": float(bonus),
    }
