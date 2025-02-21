# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

from pydantic_yaml import parse_yaml_raw_as
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from comps.finetuning.src.clip.finetune_config import FinetuneConfig


class FineTuneCallback(TrainerCallback):
    def __init__(self) -> None:
        super().__init__()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("FineTuneCallback:", args, state)


def main():
    parser = argparse.ArgumentParser(description="Runner for clip finetune")
    parser.add_argument("--tool", type=str, required=True, default="")
    parser.add_argument("--trainer", type=str, required=False, default="")
    parser.add_argument("--model", type=str, required=True, default="")
    parser.add_argument("--config_file", type=str, required=True, default="")
    parser.add_argument("--dataset", type=str, required=True, default="")
    parser.add_argument("--device", type=str, required=True, default="")

    args = parser.parse_args()
    finetune_config = FinetuneConfig(tool=args.tool, trainer=args.trainer, model=args.model, config_file=args.config_file, dataset=args.dataset, device=args.device)
    #print(finetune_config)


if __name__ == "__main__":
    main()
