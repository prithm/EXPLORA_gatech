import numpy as np
import random
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import torch
import re
import pickle
import json
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizer, BertModel, logging
import sys
import time
import pynvml
import warnings

def get_completion(pipeline, msg_in, cot_prompt=True):
    """Generates text completion using the specified language model."""

    messages = [{"role": "user", "content": "You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic. Do not generate examples in your answer."},
                {"role": "assistant", "content": "I understand."},
                {"role": "user", "content": msg_in}]

    if cot_prompt:
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = msg_in

    outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10,
                     top_p=1.0)
    out_text = [output["generated_text"] for output in outputs]
    return out_text

def get_batch_completion(pipeline, msg_in_list, cot_prompt=True):
    """Generates batch text completions using the specified language model."""

    all_messages = []
    for msg_in in msg_in_list:
        messages = [{"role": "user", "content": "You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic. Do not generate examples in your answer."},
                    {"role": "assistant", "content": "I understand."},
                    {"role": "user", "content": msg_in}]
        all_messages.append(messages)

    if cot_prompt:
        prompts = [pipeline.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in all_messages]
    else:
        prompts = msg_in_list

    outputs = pipeline(prompts, max_new_tokens=256, do_sample=True, num_return_sequences=10, temperature=0.5, top_k=10,
                     top_p=1.0)

    out_texts = []
    for batch_output in outputs:
        out_text = [output["generated_text"] for output in batch_output]
        out_texts.append(out_text)

    return out_texts

def llm_batch_output(pipeline, user_queries, dataset, cot_prompt=True):
    """Generates LLM output and applies self-consistency."""

    results = get_batch_completion(pipeline, user_queries, cot_prompt)
    final_results = []
    if dataset in ["aquarat", "finqa", "gsm8k", "strategyqa", "tabmwp"]:
        for result in results:
            res = self_con(result, dataset)
            if len(res) == 0:
                return ""
            answer = res[0][0]
            if answer == "" and len(res) > 1:
                answer = res[1][0]
            final_results.append(answer)
            
    return final_results


def self_con(tmp_list, dataset):
    """Applies self-consistency to the generated answers."""

    ans_list = []
    for tmp in tmp_list:
        ans = ""
        if dataset in ["aquarat", "tabmwp"]:
            if len(tmp.split("The answer is:")) > 0:
                ans = tmp.split("The answer is:")[-1]
                ans = ans.split("\n")[0].strip()
            ans = ans.replace("$", "").replace("%", "").replace(",", "").strip()
            if ans != "" and ans[-1] == '.':
                ans = ans[:-1]
        elif dataset == "finqa":
            if len(tmp.split("The answer is ")) > 0:
                ans = tmp.split("The answer is ")[-1]
                ans = ans.split("\n")[0].strip()
            ans = ans.replace("$", "").replace("%", "").strip()

            if 'yes' in ans.lower() or 'true' in ans.lower():
                ans = 'yes'
            elif 'no' in ans.lower() or 'false' in ans.lower():
                ans = 'no'
            try:
                ans = float(ans)
            except:
                pass
            if type(ans) == float:
                ans = round(ans, 2)
        elif dataset == "gsm8k":
            if len(tmp.split("Final Answer:")) > 0:
                ans = tmp.split("Final Answer:")[-1]
                ans = ans.split("\n")[0]
                if "each" in ans:
                    ans = ans.split("each")[0]
                if "=" in ans:
                    ans = ans.split("=")[-1]
                ans = re.sub(r'[^0-9.]', "", ans)
                if len(ans) > 0 and ans[-1] == ".":
                    ans = ans[:-1]
                try:
                    ans = round(float(ans))
                except:
                    pass
        elif dataset == "strategyqa":
            if len(tmp.split("The answer is:")) > 1:
                ans = tmp.split("The answer is:")[1]
                ans = ans.split("\n")[0]
        ans_list.append(ans)

    d = {}
    for a in ans_list:
        if a:
            d[a] = d.get(a, 0) + 1
    return sorted(d.items(), key=lambda x: x[1], reverse=True)


def llm_output(pipeline, user_query, dataset, cot_prompt=True):
    """Generates LLM output and applies self-consistency."""

    results = get_completion(pipeline, user_query, cot_prompt)
    if dataset in ["aquarat", "finqa", "gsm8k", "strategyqa", "tabmwp"]:
        res = self_con(results, dataset)
        if len(res) == 0:
            return ""
        answer = res[0][0]
        if answer == "" and len(res) > 1:
            answer = res[1][0]
        return answer
    else:
        return results[0]


def get_embeddings(text_list):
    """Gets embeddings for a list of texts."""
    inputs = tokenizer_bert(text_list, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy().tolist()
    return embeddings


def prompt_for_manual_prediction(ex, shots, dataset):
    """Constructs the prompt for manual prediction."""

    if dataset == "aquarat":
        prompt = "\n".join(["Q: {}\nO: {} \nA: {}. The option is {}\n".format(s["question"], s["options"], s["rationale"], s["correct"]) for index, s in shots.iterrows()])
        input_example = "\nQ: {}\n O: {}\nA:".format(ex['question'], ex['options'])
        prompt = "\n".join(
            ["You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic.Do not generate examples in your answer",
             prompt, input_example])
    elif dataset == "finqa":
        prompt = "\n".join(["Read the following table, and then answer the question:\nTable: {}\nQuestion: {}\nEquation: {}\n. The answer is {}\n".format(s["table"], s["question"], s["program"], s["answer"]) for index, s in shots.iterrows()])
        input_example = "Read the following table, and then answer the question:\nTable: {}\nQuestion: {}\nEquation:".format(ex['table'], ex['question'])
        prompt = "\n".join(
            ["You are a helpful, respectful and honest assistant helping to solve math word problems or tasks requiring reasoning or math, use the Chain-of-Thought methodology by following given examples to explain your step-by-step calculations or logic.Do not generate examples in your answer",
             prompt, input_example])
    elif dataset == "gsm8k":
        prompt = "Follow given examples and solve the Test Question at end in similar manner by giving step by step reasoning followed by the Final Answer.\n\n"
        for index, s in shots.iterrows():
            s["answer"] = re.sub("<<.*?>>", "", s["answer"])
            s["answer"] = s["answer"].replace("#### ", "Final Answer:")
            prompt += "Question:" + s["question"] + "\n" + s["answer"] + "\n\n"
        prompt += "Following the given examples generate step by step reasoning in Answer and generate Final Answer for the below question.\n\n"
        prompt += "Question:" + ex["question"]
    elif dataset == "strategyqa":
        prompt = """\n\nFollow the given examples that use the facts to answer a question by decomposing into sub questions first and then predicting the final answer as "Yes" or "No" only."""
        for index, s in shots.iterrows():
            facts_str = "\n".join(["Facts:" + fact if i == 0 else fact for i, fact in enumerate(s["facts"])])
            decomposition_str = "\n".join(["Sub-question {}: {}\n".format(i + 1, subq) for i, subq in enumerate(s["decomposition"])])
            prompt += facts_str + "\nQuestion:" + s["question"] + "\nAnswer:\n" + decomposition_str + "The answer is:" + s["answer"] + "\n"
        facts_str = "\n".join(["Facts:" + fact if i == 0 else fact for i, fact in enumerate(ex["facts"])])
        prompt += "\n\nGenerate the answer for the test example now:\n" + facts_str + "\nQuestion:" + ex["question"]
    elif dataset == "tabmwp":
        prompt = """Follow the giving Examples each using its Table to find the answer for its Question with the reasoning after Answer: and final answer after The answer is:.
            Examples:
            """
        for index, s in shots.iterrows():
            choices_str = "Please select from the following options:" + s["choices"] if type(s["choices"]) == str else ""
            prompt += "\nTable:\n" + s["table"] + "\nQuestion:" + s["question"] + choices_str + "\nAnswer:" + s["solution"] + "\nThe answer is:" + s["answer"]

        choices_str = "Please select from the following options:" + ex["choices"] if type(ex["choices"]) == str else ""
        prompt += "Following the given examples generate the answer for:\nTable:\n" + ex["table"] + "\nQuestion:" + ex["question"] + choices_str

    return prompt


def llm_avg_error(pipeline, exemplars_set, val_data, dataset):
    """Calculates the average error of the LLM predictions."""

    error = []
    for exemplars in tqdm(exemplars_set, total=len(exemplars_set), desc="predicting"):
        matches = 0
        mismatches = 0
        for index, row in val_data.iterrows():
            prompt = prompt_for_manual_prediction(row, exemplars, dataset)
            answer = llm_output(pipeline, prompt, dataset)

            if dataset in ["aquarat", "finqa"]:
                gt = row["answer"]
                if answer == gt:
                    matches += 1
                else:
                    mismatches += 1
            elif dataset == "gsm8k":
                gt = int(re.sub(r'[^0-9.]', "", row["answer"].split("#### ")[1]))
                try:
                    answer = int(answer)
                    if answer == gt:
                        matches += 1
                    else:
                        mismatches += 1
                except:
                    mismatches += 1
            elif dataset == "strategyqa":
                gt = row["answer"].lower()
                if gt in answer.lower() or answer.lower() in gt:
                    matches += 1
                else:
                    mismatches += 1
            elif dataset == "tabmwp":
                gt = row["answer"].lower()
                if answer != "" and (gt in answer.lower() or answer.lower() in gt):
                    matches += 1
                else:
                    mismatches += 1

        error.append(mismatches / (matches + mismatches))
    return error


def llm_error_indicator(pipeline, exemplars_set, val_data, dataset):
    """Calculates a binary error indicator for LLM predictions."""

    error = []
    for exemplars in tqdm(exemplars_set, total=len(exemplars_set), desc="predicting"):
        for index, row in val_data.iterrows():
            prompt = prompt_for_manual_prediction(row, exemplars, dataset)
            answer = llm_output(pipeline, prompt, dataset)
            loss = 0

            if dataset in ["aquarat", "finqa"]:
                gt = row["answer"]
                if answer != gt:
                    loss = 1
            elif dataset == "gsm8k":
                gt = int(re.sub(r'[^0-9.]', "", row["answer"].split("#### ")[1]))
                try:
                    answer = int(answer)
                    if answer != gt:
                        loss = 1
                except:
                    loss = 1
            elif dataset == "strategyqa":
                gt = row["answer"].lower()
                if gt not in answer.lower() and answer.lower() not in gt:
                    loss = 1
            elif dataset == "tabmwp":
                gt = row["answer"].lower()
                if answer == "" or (gt not in answer.lower() and answer.lower() not in gt):
                    loss = 1
            error.append(loss)
    return error


def static_subset_selection(pipeline, val_data, train_data, k, dataset, train_emb, val_emb):
    """Selects a static subset of exemplars."""

    train_data['cluster'] = KMeans(n_clusters=k, random_state=0).fit_predict(train_emb)
    train_data['index'] = np.arange(len(train_data))
    num_groups = len(train_data['cluster'].unique())
    L = [pd.concat([group.sample(1, random_state=i) for name, group in train_data.groupby('cluster')]) for i in range(100)]
    U = random.sample(L, 10)
    V = random.sample([l for l in L if l not in U], 5)
    L_minus_U = [l for l in L if l not in U]
    E_val = cosine_similarity(train_emb, val_emb)
    
    LLM_loss_U = llm_error_indicator(pipeline, U, val_data, dataset)
    LLM_loss_V = llm_error_indicator(pipeline, V, val_data, dataset)
    LLM_loss_L_minus_U = llm_error_indicator(pipeline, L_minus_U, val_data, dataset)
    W_val = np.zeros((len(U) * len(val_emb), len(train_data)))
    V_val = np.zeros((len(V) * len(val_emb), len(train_data)))
    X_val = np.zeros((len(L_minus_U) * len(val_emb), len(train_data)))

    U_indices = [u['index'].tolist() for u in U]
    V_indices = [v['index'].tolist() for v in V]
    L_minus_U_indices = [l_minus_u['index'].tolist() for l_minus_u in L_minus_U]

    for u_idx, u in enumerate(U_indices):
        for i, train_idx in enumerate(train_emb):
            if i in u:
                W_val[u_idx * len(val_emb):(u_idx + 1) * len(val_emb), i] = E_val[i]
    for v_idx, v in enumerate(V_indices):
        for i, train_idx in enumerate(train_emb):
            if i in v:
                V_val[v_idx * len(val_emb):(v_idx + 1) * len(val_emb), i] = E_val[i]
    for x_idx, x in enumerate(L_minus_U_indices):
        for i, train_idx in enumerate(train_emb):
            if i in x:
                X_val[x_idx * len(val_emb):(x_idx + 1) * len(val_emb), i] = E_val[i]

    for t in range(10):
        alpha = np.linalg.lstsq(np.concatenate((W_val, V_val), axis=0), np.concatenate((LLM_loss_U, LLM_loss_V)),
                              rcond=None)[0]
        S_worst_ind = np.argmax(np.sum(W_val @ alpha, axis=1)) // len(val_data)
        S_star_ind = np.argmin(np.sum(X_val @ alpha, axis=1)) // len(val_data)

        U.pop(S_worst_ind)
        U.append(L_minus_U.pop(S_star_ind))
        L_minus_U.append(V.pop(np.argmax(np.sum(V_val @ alpha, axis=1)) // len(val_data)))
        V.append(U[S_worst_ind])

        LLM_loss_U = LLM_loss_U[:S_worst_ind * len(val_data)] + LLM_loss_U[(S_worst_ind + 1) * len(val_data):]
        LLM_loss_U.extend(llm_error_indicator(pipeline, [U[-1]], val_data, dataset))
        LLM_loss_L_minus_U = LLM_loss_L_minus_U[:S_star_ind * len(val_data)] + LLM_loss_L_minus_U[
                                                                            (S_star_ind + 1) * len(val_data):]
        LLM_loss_L_minus_U.extend(llm_error_indicator(pipeline, [L_minus_U[-1]], val_data, dataset))
        LLM_loss_V = llm_error_indicator(pipeline, V, val_data, dataset)

        U_indices = [u['index'].tolist() for u in U]
        V_indices = [v['index'].tolist() for v in V]
        L_minus_U_indices = [l_minus_u['index'].tolist() for l_minus_u in L_minus_U]

        W_val = np.zeros((len(U) * len(val_data), len(train_data)))
        V_val = np.zeros((len(V) * len(val_data), len(train_data)))
        X_val = np.zeros((len(L_minus_U) * len(val_data), len(train_data)))
        for u_idx, u in enumerate(U_indices):
            for i, train_idx in enumerate(train_emb):
                if i in u:
                    W_val[u_idx * len(val_data):(u_idx + 1) * len(val_data), i] = E_val[i]
        for v_idx, v in enumerate(V_indices):
            for i, train_idx in enumerate(train_emb):
                if i in v:
                    V_val[v_idx * len(val_data):(v_idx + 1) * len(val_data), i] = E_val[i]
        for x_idx, x in enumerate(L_minus_U_indices):
            for i, train_idx in enumerate(train_emb):
                if i in x:
                    X_val[x_idx * len(val_data):(x_idx + 1) * len(val_data), i] = E_val[i]
    return U

def get_gpu_utilization():
    """Returns GPU utilization percentage."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu
    except Exception as e:
        print(f"Error getting GPU utilization: {e}")
        return ""

def get_open_source_completions(dataset_name, model_name_prefix, pipeline, train_data, test_data, mode, batch_size=8):
    """Main function to get completions for a given dataset."""

    train_emb_file = ""
    val_emb_file = ""
    if dataset_name == "aquarat":
        train_emb_file = "datasets/AQUA_RAT/aquarat_embeddings/transfer_emb.pkl"
        val_emb_file = "datasets/AQUA_RAT/aquarat_embeddings/val_emb.pkl"
    elif dataset_name == "finqa":
        train_emb_file = "datasets/FinQA/finqa_embeddings/transfer_emb.pkl"
        val_emb_file = "datasets/FinQA/finqa_embeddings/val_emb.pkl"
    elif dataset_name == "gsm8k":
        train_emb_file = "datasets/GSM8K/gsm8k_embeddings/transfer_emb.pkl"
        val_emb_file = "datasets/GSM8K/gsm8k_embeddings/val_emb.pkl"
    elif dataset_name == "strategyqa":
        train_emb_file = "datasets/Strategyqa/strategy_qa_embeddings/strategy_train_emb.pkl"
        val_emb_file = "datasets/Strategyqa/strategy_qa_embeddings/strategy_val_emb.pkl"
    elif dataset_name == "tabmwp":
        train_emb_file = "datasets/tabmwp/tabmwp_embeddings/pickle_train_new.pkl"
        val_emb_file = "datasets/tabmwp/tabmwp_embeddings/pickle_train_new.pkl"  # Reuses train embeddings
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
    train_emb = None
    val_emb = None
    
    train_data, val_data = train_test_split(train_data, test_size=0.3, random_state=42)
    val_data = val_data[:20]
    
    if mode == "full":
        # if train_emb_file != "":
        with open(val_emb_file, 'rb') as f:
            val_emb = pickle.load(f)
        with open(train_emb_file, 'rb') as f:
            train_emb = pickle.load(f)
        # else:
        #     train_emb = get_embeddings(train_data["question"].tolist())
        #     val_emb = get_embeddings(val_data["question"].tolist())
        start_time = time.time()
        exemplars = static_subset_selection(pipeline, val_data, train_data, 5, dataset_name, train_emb, val_emb)
        avg_err = llm_avg_error(pipeline, exemplars, val_data, dataset_name)
        ind = np.argmin(avg_err)
        start_time = time.time()    
        end_time = time.time()
        train_time = end_time - start_time
        gpu_utilization = get_gpu_utilization()
        
        exemplars_df = pd.concat(exemplars)
        exemplars_df.to_csv(f"output/{dataset_name}_{model_name_prefix}_subset_selection.csv")

        exemplars = exemplars[ind]
        exemplars_df.to_csv(f"output/{dataset_name}_{model_name_prefix}_selected_exemplar.csv")
    
    elif mode == "test":
        exemplars = pd.read_csv(f"output/{dataset_name}_{model_name_prefix}_selected_exemplar.csv")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    question_df = {"question": [], "answers": [], "ground_truth": []}
    matches = 0
    mismatches = 0

    # for index, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Generating"):
    #     prompt = prompt_for_manual_prediction(row, exemplars, dataset_name)
    #     answer = llm_output(pipeline, prompt, dataset_name)
    #     gt = row["answer"]

    #     if dataset_name in ["aquarat", "finqa"]:
    #         if answer == gt:
    #             matches += 1
    #         else:
    #             mismatches += 1
    #     elif dataset_name == "gsm8k":
    #         try:
    #             answer = int(answer)
    #             gt = int(re.sub(r'[^0-9.]', "", gt.split("#### ")[1]))
    #             if answer == gt:
    #                 matches += 1
    #             else:
    #                 mismatches += 1
    #         except:
    #             mismatches += 1
    #     elif dataset_name == "strategyqa":
    #         if gt.lower() in answer.lower() or answer.lower() in gt.lower():
    #             matches += 1
    #         else:
    #             mismatches += 1
    #     elif dataset_name == "tabmwp":
    #         if answer != "" and (gt.lower() in answer.lower() or answer.lower() in gt.lower()):
    #             matches += 1
    #         else:
    #             mismatches += 1

    #     question_df['question'].append(row["question"])
    #     question_df["answers"].append(answer)
    #     question_df["ground_truth"].append(gt)
        
    #     if index >= 10:
    #         break
    
    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating Batch"):
        batch = test_data.iloc[i:i + batch_size]
        prompts = [prompt_for_manual_prediction(row, exemplars, dataset_name) for _, row in batch.iterrows()]
        answers = llm_batch_output(pipeline, prompts, dataset_name)
        
        for (index, row), answer in zip(batch.iterrows(), answers):
            if dataset_name in ["finqa"]:
                gt = row["answer"]
                if answer == gt:
                    matches += 1
                else:
                    mismatches += 1
            elif dataset_name in ["aquarat"]:
                gt = row["correct"]
                if(answer==gt):
                    matches+=1
                else:
                    mismatches+=1
            elif dataset_name == "gsm8k":
                gt = row["answer"]
                try:
                    answer = int(answer)
                    gt = int(re.sub(r'[^0-9.]', "", gt.split("#### ")[1]))
                    if answer == gt:
                        matches += 1
                    else:
                        mismatches += 1
                except:
                    mismatches += 1
            elif dataset_name == "strategyqa":
                gt = row["answer"]
                if gt.lower() in answer.lower() or answer.lower() in gt.lower():
                    matches += 1
                else:
                    mismatches += 1
            elif dataset_name == "tabmwp":
                gt = row["answer"]
                if answer != "" and (gt.lower() in answer.lower() or answer.lower() in gt.lower()):
                    matches += 1
                else:
                    mismatches += 1

            question_df['question'].append(row["question"])
            question_df["answers"].append(answer)
            question_df["ground_truth"].append(gt)
            
        # if (i+1) * batch_size >= 48:
        #     break

    final_questions = pd.DataFrame(question_df)
    final_questions.to_csv(f"output/{dataset_name}_{model_name_prefix}_question_answer.tsv", sep="\t", index=False)
    em = matches / (matches + mismatches)
    result_dict = {"matches": [matches],
                   "mismatches": [mismatches], "EM": [em], "val_data_len": [len(val_data)],
                   "train_data_len": [len(train_data)], "test_data_len": [len(test_data)],
                   "model_name_prefix": [model_name_prefix], "dataset": [dataset_name]}
    if mode == "full":
        result_dict["gpu_utilization"] = [gpu_utilization]
        result_dict["training_time"] = [train_time]
    print(result_dict)
    print("\n\n")
    pd.DataFrame(result_dict).to_csv(f"output/{dataset_name}_{model_name_prefix}_result_summary.csv")
    return final_questions


def read_strategyqa(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    examples = []
    for d in data:
        answer = "Yes" if d["answer"] else "No"
        ex = {"id": d["id"], "question": d["question"].lstrip(), "answer": answer, "facts": d["facts"],
              "decomposition": d["decomposition"]}
        examples.append(ex) 
    return pd.DataFrame(examples)

def read_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
    examples = []
    for d in data:
        ex = {"question": d["question"].lstrip(), "answer": d["answer"]}
        examples.append(ex) 
    return pd.DataFrame(examples)

def run_pipeline(model_name, model_name_prefix, torch_dtype, dataset_name, mode, batch_size):
    warnings.filterwarnings("ignore") # Suppress all warnings
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model Configuration
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch_dtype)
    # Explicitly set pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id # or another appropriate value
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # model.generation_config.pad_token_id = model.generation_config.eos_token_id
    pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    # tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
    # model_bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    # logging.set_verbosity_error()
    
    # "aquarat", "finqa", "gsm8k", "strategyqa", "tabmwp"
    if dataset_name == "aquarat":
        train_data = "datasets/AQUA_RAT/aquarat_train.csv"
        test_data = "datasets/AQUA_RAT/aquarat_dev.csv"
    elif dataset_name == "finqa":
        train_data = "datasets/FinQA/finqa_train.csv"
        test_data = "datasets/FinQA/finqa_test.csv"
    elif dataset_name == "gsm8k":
        train_data = "datasets/GSM8K/gsm8k_train.csv"
        test_data = "datasets/GSM8K/gsm8k_test_v2.json"
    elif dataset_name == "strategyqa":
        train_data = "datasets/Strategyqa/strategyqa_train_v2.json"
        test_data = "datasets/Strategyqa/strategyqa_test.json"
    elif dataset_name == "tabmwp":
        train_data = "datasets/tabmwp/problems_train.json"
        test_data = "datasets/tabmwp/problems_dev.json"
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    if dataset_name in ["aquarat", "finqa", "tabmwp"]:
        train_data = pd.read_csv(train_data) if train_data.endswith(".csv") else pd.read_json(train_data, orient='index')
        test_data = pd.read_csv(test_data) if test_data.endswith(".csv") else pd.read_json(test_data, orient='index')
    elif dataset_name == "gsm8k":
        train_data = pd.read_csv(train_data)
        test_data = read_json(test_data)
    elif dataset_name == "strategyqa":
        train_data = read_strategyqa(train_data)
        test_data = read_strategyqa(test_data)
    #     exit(1)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    final_df = get_open_source_completions(dataset_name, model_name_prefix, pipeline, train_data, test_data, mode, batch_size)


# Example usage: python generic/explora_sc.py mistral7b_16 tabmwp test 12
# Example usage: python generic/explora_sc.py mistral7b_16 tabmwp full 12

if __name__ == '__main__':
     # "mistral7b_16", "mistral7b_32", "llama3b_16", "llama3b_32", "llama1b_16", "llama1b_32"
    model_name_prefix = sys.argv[1]
    # "aquarat", "finqa", "gsm8k", "strategyqa", "tabmwp"
    dataset_name = sys.argv[2]
    mode = sys.argv[3]
    batch_size = int(sys.argv[4])
    
    print(f"model_name_prefix: {model_name_prefix}, dataset_name: {dataset_name}, mode: {mode}")
    print("\n\n")
    
    torch_dtype=torch.float16
    model_name = ""
    
    if model_name_prefix.startswith("mistral7b"):
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif model_name_prefix.startswith("llama3b"):
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name_prefix.startswith("llama1b"):
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    else:
        raise ValueError(f"Invalid model name: {model_name_prefix}")
    
    if model_name_prefix.split("_")[-1] == "16":
        torch_dtype=torch.float16
    elif model_name_prefix.split("_")[-1] == "32":
        torch_dtype=torch.float32
    else:
        raise ValueError(f"Invalid model name: {model_name_prefix}")

    run_pipeline(model_name, model_name_prefix, torch_dtype, dataset_name, mode, batch_size)