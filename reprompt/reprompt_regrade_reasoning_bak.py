#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import uuid
import pandas as pd
import numpy as np
import argparse
from R2CCP.main import R2CCP
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split


def score_and_extract(prompt, model, tokenizer, max_new_tokens=10000, top_k=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=0,
            do_sample=False
        )

    # Extract generated sequence and per-step logits
    sequences = generation.sequences        # shape [1, input_length + new_tokens]
    scores    = generation.scores           # list of length new_tokens, each [1, vocab_size]

    # Decode only the newly generated tokens
    generated_ids = sequences[0, input_length:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    tokens        = []
    token_logprobs = []
    top_logprobs   = []
    for step_idx, step_scores in enumerate(scores):
        # step_scores: [1, vocab_size]
        log_probs = F.log_softmax(step_scores, dim=-1)  # shape [1, vocab_size]

        # actual token id & its logprob
        token_id = sequences[0, input_length + step_idx].unsqueeze(0)
        lp = log_probs.gather(1, token_id.unsqueeze(1)).item()

        # top-k tokens and their logprobs
        topk_lp, topk_ids = log_probs.topk(top_k, dim=-1)
        topk_lp   = topk_lp[0].tolist()
        topk_ids  = topk_ids[0].tolist()
        topk_dict = {
            tokenizer.convert_ids_to_tokens(tok): logp
            for tok, logp in zip(topk_ids, topk_lp)
        }

        tokens.append(tokenizer.convert_ids_to_tokens(token_id.item()))
        token_logprobs.append(lp)
        top_logprobs.append(topk_dict)

    
    # m = re.search(r'\d+(?:\.\d+)?', generated_text)
    # raw_score = float(m.group()) if m else None
    # target_logits = top_logprobs[10] 

     # 1) 精确提取 raw_score_str
    m = re.search(r'Final Score\s*[::]\s*([0-9]+(?:\.\d+)?)', generated_text)
    if m:
        raw_score_str = m.group(1)
        raw_score     = float(raw_score_str)
    else:
        # 退而求其次：抓最后一个数字
        nums = re.findall(r'[0-9]+(?:\.\d+)?', generated_text)
        raw_score_str = nums[-1] if nums else None
        raw_score     = float(raw_score_str) if raw_score_str else None

    # 2) 在 tokens 里找对应的 step index
    target_idx = None
    if raw_score_str is not None:
        for i, tok in enumerate(tokens):
            if tok.strip() == raw_score_str:
                target_idx = i
                break
    # 万一下标还没找着，再抓第一个数字型 token
    if target_idx is None:
        for i, tok in enumerate(tokens):
            if re.fullmatch(r'[0-9]+(?:\.\d+)?', tok.strip()):
                target_idx = i
                break

    # 3) 用这个 index 提取 logits 分布
    if target_idx is not None and target_idx < len(top_logprobs):
        target_logits = top_logprobs[target_idx]
    else:
        # 如果实在没找到，就 fallback 全部 top_logprobs[0] 或 raise
        target_logits = top_logprobs[0]

    return generated_text, raw_score, target_logits

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def merge_intervals(sample_intervals):
    if not sample_intervals:
        return (1,5)
    lows = [low for low, high in sample_intervals]
    highs = [high for low, high in sample_intervals]
    return (min(lows), max(highs))

def boundary_adjustment(value, threshold=0.0):
    label_set=np.array([1, 2, 3, 4, 5])
    threshold_max = (label_set[-1] - label_set[0]) / (len(label_set) - 1) / 2
    threshold = min(threshold_max, threshold)
    adjusted_value = next((num for num in label_set if abs(num - value) <= threshold), value)
    adjusted_value = np.clip(adjusted_value, 1, 5)
    return adjusted_value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dimension', type=str, required=True,
        help='which dimension to evaluate'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=10000,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Number of top token logprobs to record at each generation step'
    )
    parser.add_argument(
        '--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help='Model used to evaluate as a judge'
    )

    args = parser.parse_args()
    dimension = args.dimension
    os.makedirs('R2CCP_paths', exist_ok=True)

    df = pd.read_csv(f"./reprompt_regrade/socreval/SocREval_{dimension}_logits.csv") 
    df['human'] = df['human'].astype(float)
    with open(f"./reprompt_regrade/socreval/dsr1_socreval_{dimension}.json","r",encoding="utf-8") as f:
        records = json.load(f) 

    # Initialize model and tokenizer
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=os.environ.get("HF_HOME", None),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME", None),
    )
    model.eval()

    results = []
    for i in tqdm(range(len(df)), desc="LOO & Regrade"):
        calibr_df = df.drop(index=i)
        test_row  = df.iloc[i]
        
        X_cal = calibr_df[['1','2','3','4','5']].to_numpy().astype(np.float32)
        y_cal = calibr_df['human'].to_numpy().astype(np.float32)
        
        X_test = test_row[['1','2','3','4','5']].to_numpy().reshape(1,-1).astype(np.float32)
        
        X_test_softmax = softmax(X_test)
    
        init_score_weight = X_test_softmax @ np.array([1, 2, 3, 4, 5])

        predictor = R2CCP({
            'model_path': f'R2CCP_paths/loo_{i}.pth',
            'max_epochs': 100,
            'alpha': 0.1
        })
        predictor.fit(X_cal, y_cal)
          
        interval = predictor.get_intervals(X_test)

        interval = [merge_intervals(sample_intervals) for sample_intervals in interval]
        low = boundary_adjustment(interval[0][0], threshold=0.5)
        up = boundary_adjustment(interval[0][1], threshold=0.5)
        interval = f"[{low:.2f}, {up:.2f}]"
        
        rec = records[i]
        prompt0 = rec['body']['messages'][0]['content']
        init_ans = rec['judge']

        reprompt = open("./reprompt.txt", 'r', encoding='utf-8').read().replace("{{Interval}}", interval)
        reprompt = ( "Let me show you our evalutaion record. Based on all these information, make dicision and give me final score."
                + "Initial prompt: \n" + prompt0 + "\n" 
                + "Initial response: \n" + init_ans + "\n"
                + "Reprompt and Regrade: \n" + reprompt)
        
        gen_text, re_score_raw, re_logits = score_and_extract(
            reprompt, model, tokenizer,
            args.max_new_tokens, args.top_k
        )
        
        re_logits = {str(k): max(v,math.log(1e-5)) for k, v in re_logits.items()}
        for k in ['1','2','3','4','5']:
            if k not in re_logits or re_logits[k] is None:
                re_logits[k] = math.log(1e-5)
        
        re_logits = np.array([re_logits[l] for l in ['1','2','3','4','5']], dtype=np.float64).reshape(1,-1)
        re_logits_softmax = softmax(re_logits)
        re_score_weight = re_logits_softmax @ np.array([1, 2, 3, 4, 5])

        results.append({
            'index':      i,
            'low':        low,
            'up':         up,
            'init_score_weight': init_score_weight,
            're_score_raw':   re_score_raw,
            're_score_weight':  re_score_weight,
            'ground_truth': test_row['human'],
            'final_text': gen_text
        })


    res_df = pd.DataFrame(results)
    res_df.to_csv(f"loo_reprompt_socreval_{dimension}.csv", index=False)

if __name__ == "__main__":
    main()
