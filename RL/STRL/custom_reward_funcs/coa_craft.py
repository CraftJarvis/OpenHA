import ray
import re
import math
from rl.utils.action_mapping import OneActionTokenizer
import torch
from minestudio.simulator.entry import MinecraftSim
import random
# def compute_score(data_source, solution_str, ground_truth, extra_info=None):
#     """
#     给定ground_truth中包含的一组reserved_special_token，
#     返回solution_str中正确匹配这些token的比例得分。
#     """
#     try:
#         tokenizer = OneActionTokenizer("qwen2_vl")
#         breakpoint()


#         ##print("solution_str:", solution_str)
#         # 匹配所有 token，例如 <|reserved_special_token_178|>
#         gt_tokens = re.findall(r"<\|reserved_special_token_\d+\|>", ground_truth)
#         if not gt_tokens:
#             #print("[Warning] No valid tokens in ground_truth.")
#             return 0.0

#         matched_count = sum(1 for tok in gt_tokens if tok in solution_str)
#         if matched_count < 3:
#             return 0.0
#         else:
#             score = (matched_count-2) / (len(gt_tokens)-2)

#         # 可选调试输出
#         # #print(f"[Reward] Matched {matched_count}/{len(gt_tokens)} tokens | Score: {score:.3f}")

#         return score

#     except Exception as e:
#         #print(f"[Error] in compute_score: {e}")
#         return 0.0


_sim = None
_tokenizer = None
_qwen2vl_token2id_lst = None
# qwen2vl_token2id_lst = [
#                 [('<|reserved_special_token_178|>', 151835), ('<|reserved_special_token_179|>', 151836),],
#                 [["<|reserved_special_token_180|>", 151837], ["<|reserved_special_token_181|>", 151838], ["<|reserved_special_token_182|>", 151839], ["<|reserved_special_token_183|>", 151840], ["<|reserved_special_token_184|>", 151841], ["<|reserved_special_token_185|>", 151842], ["<|reserved_special_token_186|>", 151843], ["<|reserved_special_token_187|>", 151844], ["<|reserved_special_token_188|>", 151845], ["<|reserved_special_token_189|>", 151846]], 
#                 [["<|reserved_special_token_190|>", 151847], ["<|reserved_special_token_191|>", 151848], ["<|reserved_special_token_192|>", 151849]], 
#                 [["<|reserved_special_token_193|>", 151850], ["<|reserved_special_token_194|>", 151851], ["<|reserved_special_token_195|>", 151852]], 
#                 [["<|reserved_special_token_196|>", 151853], ["<|reserved_special_token_197|>", 151854], ["<|reserved_special_token_198|>", 151855]], 
#                 [["<|reserved_special_token_199|>", 151856], ["<|reserved_special_token_200|>", 151857]],  # use
#                 [["<|reserved_special_token_201|>", 151858], ["<|reserved_special_token_202|>", 151859]],  # drop
#                 [["<|reserved_special_token_203|>", 151860], ["<|reserved_special_token_204|>", 151861]],  # attack
#                 [["<|reserved_special_token_205|>", 151862], ["<|reserved_special_token_206|>", 151863]],  # jump
#                 [["<|reserved_special_token_207|>", 151864], ["<|reserved_special_token_208|>", 151865],],  # camera
#                 [["<|reserved_special_token_176|>", 151833], ["<|reserved_special_token_177|>", 151834],],
#                 [["<|reserved_special_token_209|>", 151866], ["<|reserved_special_token_210|>", 151867], ["<|reserved_special_token_211|>", 151868], ["<|reserved_special_token_212|>", 151869], ["<|reserved_special_token_213|>", 151870], ["<|reserved_special_token_214|>", 151871], ["<|reserved_special_token_215|>", 151872], ["<|reserved_special_token_216|>", 151873], ["<|reserved_special_token_217|>", 151874], ["<|reserved_special_token_218|>", 151875], ["<|reserved_special_token_219|>", 151876], ["<|reserved_special_token_220|>", 151877], ["<|reserved_special_token_221|>", 151878], ["<|reserved_special_token_222|>", 151879], ["<|reserved_special_token_223|>", 151880], ["<|reserved_special_token_224|>", 151881], ["<|reserved_special_token_225|>", 151882], ["<|reserved_special_token_226|>", 151883], ["<|reserved_special_token_227|>", 151884], ["<|reserved_special_token_228|>", 151885], ["<|reserved_special_token_229|>", 151886]], 
#                 [["<|reserved_special_token_230|>", 151887], ["<|reserved_special_token_231|>", 151888], ["<|reserved_special_token_232|>", 151889], ["<|reserved_special_token_233|>", 151890], ["<|reserved_special_token_234|>", 151891], ["<|reserved_special_token_235|>", 151892], ["<|reserved_special_token_236|>", 151893], ["<|reserved_special_token_237|>", 151894], ["<|reserved_special_token_238|>", 151895], ["<|reserved_special_token_239|>", 151896], ["<|reserved_special_token_240|>", 151897], ["<|reserved_special_token_241|>", 151898], ["<|reserved_special_token_242|>", 151899], ["<|reserved_special_token_243|>", 151900], ["<|reserved_special_token_244|>", 151901], ["<|reserved_special_token_245|>", 151902], ["<|reserved_special_token_246|>", 151903], ["<|reserved_special_token_247|>", 151904], ["<|reserved_special_token_248|>", 151905], ["<|reserved_special_token_249|>", 151906], ["<|reserved_special_token_250|>", 151907]],
#         ]


def get_sim():
    global _sim, _tokenizer, _qwen2vl_token2id_lst
    if _sim is None:
        print("[Info] Initializing MinecraftSim and OneActionTokenizer for COA-Craft...")
        _tokenizer = OneActionTokenizer("qwen2_vl")
        _sim = MinecraftSim(camera_config=_tokenizer.camera_config)
        _qwen2vl_token2id_lst = [
                [('<|reserved_special_token_178|>', 151835), ('<|reserved_special_token_179|>', 151836),],
                [["<|reserved_special_token_180|>", 151837], ["<|reserved_special_token_181|>", 151838], ["<|reserved_special_token_182|>", 151839], ["<|reserved_special_token_183|>", 151840], ["<|reserved_special_token_184|>", 151841], ["<|reserved_special_token_185|>", 151842], ["<|reserved_special_token_186|>", 151843], ["<|reserved_special_token_187|>", 151844], ["<|reserved_special_token_188|>", 151845], ["<|reserved_special_token_189|>", 151846]], 
                [["<|reserved_special_token_190|>", 151847], ["<|reserved_special_token_191|>", 151848], ["<|reserved_special_token_192|>", 151849]], 
                [["<|reserved_special_token_193|>", 151850], ["<|reserved_special_token_194|>", 151851], ["<|reserved_special_token_195|>", 151852]], 
                [["<|reserved_special_token_196|>", 151853], ["<|reserved_special_token_197|>", 151854], ["<|reserved_special_token_198|>", 151855]], 
                [["<|reserved_special_token_199|>", 151856], ["<|reserved_special_token_200|>", 151857]],  # use
                [["<|reserved_special_token_201|>", 151858], ["<|reserved_special_token_202|>", 151859]],  # drop
                [["<|reserved_special_token_203|>", 151860], ["<|reserved_special_token_204|>", 151861]],  # attack
                [["<|reserved_special_token_205|>", 151862], ["<|reserved_special_token_206|>", 151863]],  # jump
                [["<|reserved_special_token_207|>", 151864], ["<|reserved_special_token_208|>", 151865],],  # camera
                [["<|reserved_special_token_176|>", 151833], ["<|reserved_special_token_177|>", 151834],],
                [["<|reserved_special_token_209|>", 151866], ["<|reserved_special_token_210|>", 151867], ["<|reserved_special_token_211|>", 151868], ["<|reserved_special_token_212|>", 151869], ["<|reserved_special_token_213|>", 151870], ["<|reserved_special_token_214|>", 151871], ["<|reserved_special_token_215|>", 151872], ["<|reserved_special_token_216|>", 151873], ["<|reserved_special_token_217|>", 151874], ["<|reserved_special_token_218|>", 151875], ["<|reserved_special_token_219|>", 151876], ["<|reserved_special_token_220|>", 151877], ["<|reserved_special_token_221|>", 151878], ["<|reserved_special_token_222|>", 151879], ["<|reserved_special_token_223|>", 151880], ["<|reserved_special_token_224|>", 151881], ["<|reserved_special_token_225|>", 151882], ["<|reserved_special_token_226|>", 151883], ["<|reserved_special_token_227|>", 151884], ["<|reserved_special_token_228|>", 151885], ["<|reserved_special_token_229|>", 151886]], 
                [["<|reserved_special_token_230|>", 151887], ["<|reserved_special_token_231|>", 151888], ["<|reserved_special_token_232|>", 151889], ["<|reserved_special_token_233|>", 151890], ["<|reserved_special_token_234|>", 151891], ["<|reserved_special_token_235|>", 151892], ["<|reserved_special_token_236|>", 151893], ["<|reserved_special_token_237|>", 151894], ["<|reserved_special_token_238|>", 151895], ["<|reserved_special_token_239|>", 151896], ["<|reserved_special_token_240|>", 151897], ["<|reserved_special_token_241|>", 151898], ["<|reserved_special_token_242|>", 151899], ["<|reserved_special_token_243|>", 151900], ["<|reserved_special_token_244|>", 151901], ["<|reserved_special_token_245|>", 151902], ["<|reserved_special_token_246|>", 151903], ["<|reserved_special_token_247|>", 151904], ["<|reserved_special_token_248|>", 151905], ["<|reserved_special_token_249|>", 151906], ["<|reserved_special_token_250|>", 151907]],
        ]
    return _sim, _tokenizer, _qwen2vl_token2id_lst


def raw_str2env(reserved_tokens_str):
    sim, tokenizer, qwen2vl_token2id_lst = get_sim()
    reserved_tokens_list = reserved_tokens_str.split("<")
    reserved_tokens_list = ["<" + token for token in reserved_tokens_list if token]
    
    token2id_dct = {}
    for token_list in qwen2vl_token2id_lst:
        for token, id in token_list:
            token2id_dct[token] = id

    reserved_tokens_list = [token2id_dct[token] for token in reserved_tokens_list if token in token2id_dct]
    return sim.agent_action_to_env_action(tokenizer.decode(reserved_tokens_list)[0])


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    给定ground_truth中包含的一组reserved_special_token，
    返回solution_str中正确匹配这些token的比例得分。
    综合考虑按钮匹配与相机方向相似度。
    """

    #print(f"solution_str: {solution_str}")
    try:
        ground_action = raw_str2env(ground_truth)
        solution_action = raw_str2env(solution_str)
    except Exception as e:
        print(f"[Error] in compute_score: {e}")
        return final_score #{"score": final_score, "button_score": button_score, "camera_score": camera_score}

    # 1. 按钮键（排除 camera）
    button_keys = [k for k in ground_action.keys() if k != 'camera']
    num_buttons = len(button_keys)
    
    # 2. 计算按钮匹配数量
    correct_buttons = sum(
        ground_action[k] == solution_action.get(k, not ground_action[k]) for k in button_keys
    )
    button_score = 2*correct_buttons / (num_buttons + len(solution_action) - 1) if num_buttons > 0 else 0.0

    # 3. 相机 loss → 相似度得分
    try:
        camera_gt = torch.tensor(ground_action["camera"], dtype=torch.float32)
        camera_pred = torch.tensor(solution_action["camera"], dtype=torch.float32)
        
        max_mse = 3.0  # 假设最大均方误差为0.1
        mse = torch.nn.functional.mse_loss(camera_gt, camera_pred).item()
        camera_score = max(0, 1 - mse / max_mse)  # 假设 max_mse 是一个合理的上限

        # camera_score = 1/math.exp(5*torch.nn.functional.mse_loss(camera_gt, camera_pred).item())
        # camera_score = max(0.0, min(1.0, camera_score))
    except Exception as e:
        print(f"[Warning] camera compare failed: {e}")
        return final_score #{"score": final_score, "button_score": button_score, "camera_score": camera_score}

    # 4. 综合得分（按钮为主，camera 为次）
    final_score = button_score  + camera_score 

    return final_score #{"score": final_score, "button_score": button_score, "camera_score": camera_score}

if __name__ == "__main__":
    # 测试用例
    ground_truth = "<|reserved_special_token_178|> <|reserved_special_token_179|> <|reserved_special_token_180|>"
    solution_str = "<|reserved_special_token_178|><|reserved_special_token_221|><|reserved_special_token_235|><|reserved_special_token_179|>"
    
    score = compute_score("test_source", solution_str, ground_truth)
    #print(f"Computed Score: {score}")