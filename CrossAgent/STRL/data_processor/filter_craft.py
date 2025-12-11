import argparse
from openagents.assets import INSTRUCTION_FILE_PATH
import json
from tqdm import tqdm

with open(INSTRUCTION_FILE_PATH, "r") as f:
    instructions = f.read()
    instructions = json.loads(instructions)

def filter_craft(input_path, output_path):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in tqdm(fin):
            data = json.loads(line)
            instruction = data["conversations"][0]["content"][0]["text"]
            filt = False
            for key in instructions:
                if instruction in instructions[key]:
                    if "craft_item" in key.lower():
                        print("Filtered craft instruction:", instruction)
                        filt = True
                        break

            if not filt:
                fout.write(line)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="/public/hgx14/datasets/minecraft-trajectory.cache/mc-text-action-samples_v1-c1i30w1camerakey-0614-train.jsonl")
    parser.add_argument("--output_path", type=str, default="/public/hgx14/datasets/minecraft-trajectory.cache/mc-text-action-samples_v1-c1i30w1camerakey-0614-train-craftfiltered.jsonl")

    args = parser.parse_args()

    filter_craft(args.train_path, args.output_path)
