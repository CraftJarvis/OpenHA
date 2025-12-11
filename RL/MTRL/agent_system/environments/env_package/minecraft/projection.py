from typing import List
import re

import ray
import re
import math
from openagents.agents.utils.action_mapping import OneActionTokenizer, TextActionTokenizer
import torch
from minestudio.simulator.entry import MinecraftSim
import random


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


# def minecraft_projection(actions: List[str]):
#     """
#     A function to process the actions.
#     actions: the list of actions to be processed, it is a list of strings.
#     Expected format:
#         <think>some reasoning...</think><action>up/down/left/right/still</action>
#     """

#     valids = [0] * len(actions)

#     for i in range(len(actions)):
#         try:
#             actions[i] = raw_str2env(actions[i])
#             valids[i] = 1
#         except Exception as e:
#             print(f"Error processing action {i}: {actions[i]}")
#             print(f"Exception: {e}")
#             actions[i] = None
#             valids[i] = 0
            
#     return actions, valids
    
tokenizer = None
def minecraft_projection(actions: List[str]):
    global tokenizer
    if tokenizer is None:
        tokenizer = TextActionTokenizer()
    valids = [0] * len(actions)

    for i in range(len(actions)):
        try:
            actions[i] = {
                "raw_action": tokenizer.decode(actions[i])[0],
                "thought": actions[i]
            }
            valids[i] = 1
            
        except Exception as e:
            print(f"Error processing action {i}: {actions[i]}")
            print(f"Exception: {e}")
            actions[i] = {
                "raw_action": None,
                "thought": None
            }
            valids[i] = 0
    
    
    return actions, valids