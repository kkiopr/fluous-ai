"""LoRA SFT on episodic (voice samples + card -> held-out text). Loss on the target only.
Untested against your exact transformers/peft versions: expect small API fixes."""
from __future__ import annotations

import argparse

import yaml

from .episodes import load_pools, make_episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16)
    l = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    pools = load_pools(cfg["data"], "train", cfg.get("per_author_cap"), cfg["seed"])
    eps = make_episodes(pools, cfg["ks"], cfg["episodes_per_record"], cfg["seed"])

    class DS(torch.utils.data.Dataset):
        def __init__(self):
            self.items = []
            for e in eps:
                p = tok(e["prompt"], add_special_tokens=True)["input_ids"]
                t = tok(e["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
                if len(p) + len(t) > cfg["max_len"]:
                    continue
                self.items.append({"input_ids": p + t, "labels": [-100] * len(p) + t,
                                   "attention_mask": [1] * (len(p) + len(t))})
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    ds = DS()
    print(f"{len(ds)} episodes from {len(pools)} authors")
    t = cfg["train"]
    args_t = TrainingArguments(output_dir=cfg["output_dir"], num_train_epochs=t["epochs"],
                               learning_rate=t["lr"], per_device_train_batch_size=t["batch_size"],
                               gradient_accumulation_steps=t["grad_accum"], bf16=True,
                               logging_steps=10, save_strategy="epoch", report_to="none")
    Trainer(model=model, args=args_t, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)).train()
    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
