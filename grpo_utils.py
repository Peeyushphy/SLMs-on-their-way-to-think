
import reasoning_gym
import torch
from reasoning_gym import get_score_answer_fn
import re
import numpy as np

FORMAT_REWARD_WEIGHT = 0.15
CORRECTNESS_REWARD_WEIGHT = 0.85

def calculate_logits(llm, input_ids, attention_mask):
    with torch.no_grad():
        output = llm(
            input_ids,
            attention_mask=attention_mask
            )
            
        
        logits = output.logits # B, T, V -> vocab size for this model is somwhere around 49k
        log_probs = torch.log_softmax(logits, dim=-1)  # B, T, V
        
        token_logprobs = log_probs[:, :-1].gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        return token_logprobs

        
        
def extract_answer(answer):
    answer = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
    if answer is not None:
        return answer.group(1)
    return answer

def calculate_format_reward(response):
    # required tags
    required = ["<think>", "</think>", "<answer>", "</answer>"]
    if any(tag not in response for tag in required):
        return -0.5

    think_open = response.find("<think>")
    think_close = response.find("</think>")
    answer_open = response.find("<answer>")
    answer_close = response.find("</answer>")

    # enforce correct order
    if not (0 <= think_open < think_close < answer_open < answer_close):
        return -0.5

    reward = 0.0

    # structure reward
    reward += 0.2  # correct format
    reward += 0.3  # answer block exists

    return reward

def correctness_reward(response, validation_object):
    score_fn = get_score_answer_fn(validation_object['metadata']['source_dataset'])
    return score_fn(response, validation_object)
    
def calculate_rewards(batch_response, validation_objects):
    
    format_reward = np.array([
        calculate_format_reward(response) for response in batch_response
    ])
    correctness_rewards = np.array([
        correctness_reward(extract_answer(response), val_obj) for val_obj, response in zip(validation_objects, batch_response)
    ])
    
    rewards = (
        FORMAT_REWARD_WEIGHT * format_reward + CORRECTNESS_REWARD_WEIGHT * correctness_rewards
    )
    return rewards
    
    