"""DPO on top of the SFT LoRA. TRL's API changes between versions: pin yours and adjust."""
from __future__ import annotations

import argparse

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, cfg["sft_adapter"], is_trainable=True)
    ds = load_dataset("json", data_files=cfg["data"])["train"].select_columns(["prompt", "chosen", "rejected"])

    args = DPOConfig(output_dir=cfg["output_dir"], beta=cfg["beta"], learning_rate=cfg["lr"],
                     num_train_epochs=cfg["epochs"], per_device_train_batch_size=cfg["batch_size"],
                     gradient_accumulation_steps=cfg["grad_accum"], max_length=cfg["max_len"],
                     bf16=True, logging_steps=10, report_to="none")
    # ref_model=None with a PEFT model -> reference = adapter disabled
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
