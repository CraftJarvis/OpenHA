
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
import random

tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = TextActionTokenizer()
    return tokenizer


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    from openagents.agents.utils.action_mapping import TextActionTokenizer
    tokenizer = TextActionTokenizer()
    try:
        ga  = tokenizer.decode(ground_truth)[0]
        sa  = tokenizer.decode(solution_str)[0]
    except Exception as e:
        return 0.0
    if not extra_info["seed"]:
        if sa["attack"] == 1:
            #print("Correct attack")
            return 1.0
        else:
            return 0.0
    else:
        if sa["attack"] == 0:
            #print("Correct no attack")
            return 1.0
        else:
            return 0.0