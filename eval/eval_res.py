import random

import torch
import json
import argparse
# Lazy-import SentenceTransformer ONLY when an *_nd evaluator actually needs it;
# nc and lp don't, and loading sentence_transformers eagerly costs ~10s per
# invocation.
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


import os
from utils.paths import dataset_dir
def sbert(model_type, device):
    from sentence_transformers import SentenceTransformer  # lazy
    model = SentenceTransformer(model_type, device=device)
    return model

def get_sbert_embedding(model_type, texts, device):
    if model_type == 'sbert':
        model_type = 'all-MiniLM-L6-v2'
    sbert_model = sbert(model_type, f'cuda:{device}')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)

def eval_arxiv_nd(res_path):
    data=torch.load(os.path.join(dataset_dir("arxiv"), "processed_data.pt"), weights_only=False)
    labels=data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys=data.y.numpy().tolist()

    titles = data.title

    all_sample=0
    short_correct=0
    all_correct=0
    gt=[]
    out=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            id=res["question_id"]
            y=ys[id]
            short_label = short_labels[y]
            label=labels[y]
            if label.strip() in ans.strip():
                all_correct+=1
            if short_label in ans:
                short_correct+=1
            out.append(ans)
            gt.append(f"This is a paper in {label} domain, it's about {titles[id]}.")
    short_acc = short_correct/all_sample
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}\nshort_correct: {short_correct}\nshort_acc: {short_acc:.4f}\nall_correct: {all_correct}\nall_acc: {all_acc:.4f}")
    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding=F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding=F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim=(gt_embedding*out_embedding).sum(1).mean().item()
    gt_sim_matrix=torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n=gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)]=0
    gt_sim=(gt_sim_matrix.sum()/(n*(n-1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")


def eval_arxiv_nc(res_path):
    data=torch.load(os.path.join(dataset_dir("arxiv"), "processed_data.pt"), weights_only=False)
    raw_labels=data.label_texts
    # `label_texts` are stored as "cs.LG(Machine Learning)".  The prompts ask
    # the model to predict the natural-language part only (e.g. "Machine
    # Learning"), so we extract that for matching.  Keep the cs.* prefix
    # available as an alternative key.
    import re
    def _split(lt):
        m = re.match(r"^\s*([a-zA-Z.]+)\s*\((.+)\)\s*$", lt)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return "", lt.strip()
    parsed = [_split(l) for l in raw_labels]   # list of (short, name)
    short_codes = [p[0].lower() for p in parsed]   # e.g. "cs.lg"
    names       = [p[1] for p in parsed]           # e.g. "Machine Learning"
    names_lc    = [n.lower() for n in names]
    ys=data.y.numpy().tolist()

    all_sample=0
    overall_correct=0
    strict_correct=0
    error=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"].strip()
            ans_lc = ans.lower()
            y = ys[res["question_id"]]
            gt_short = short_codes[y]
            gt_name  = names[y]
            gt_name_lc = names_lc[y]
            # strict: model output exactly equals the natural-language name
            # (case-insensitive, whitespace-trimmed).
            if ans_lc == gt_name_lc or ans.strip() == raw_labels[y].strip():
                strict_correct += 1
                overall_correct += 1
                continue
            # overall: gt name appears in answer AND no other label name does;
            # OR gt short code (cs.lg) appears as a substring uniquely.
            name_hits = sum(n in ans_lc for n in names_lc)
            short_hits = sum(c in ans_lc for c in short_codes if c)
            if (gt_name_lc in ans_lc and name_hits == 1) or \
               (gt_short and gt_short in ans_lc and short_hits == 1):
                overall_correct += 1
            else:
                error.append((ans, gt_name))
            if args.sample > 0 and all_sample >= args.sample:
                break
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")
    print(f"acc: {overall_acc:.4f}")


def eval_lp(res_path):
    all_sample=0
    correct = 0
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            ans = res["text"].strip()
            label=res["gt"].strip()
            all_sample += 1
            if ("yes" in ans and "yes" in label) or ("yes" not in ans and "no" in label):
                correct += 1
            if args.sample > 0 and all_sample >=  args.sample:
                break
    acc = correct / all_sample
    print(f"Test samples: {all_sample}\ncorrect: {correct}\n acc: {acc:.4f}")

def eval_lprank(res_path):
    all_sample=0
    correct = 0
    y_true = []
    y_pred=[]
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            logit = res["logit"]
            score = torch.softmax(torch.tensor(logit[:2]), dim=-1)[0].item()
            # score = logit[0]
            label=res["gt"].strip()
            if label == "yes":
                y_true.append(1)
            else:
                y_true.append(0)
            y_pred.append(score)
    auc = roc_auc_score(y_true, y_pred)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    acc = ((y_pred>0.5)==y_true).sum()/y_pred.shape[0]

    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    y_pos=y_pred[y_true==1]
    y_neg=y_pred[y_true==0]
    y_neg_sort, _ = torch.sort(y_neg)
    for n in [10,50,100,200,500,1000]:
        if n > y_neg_sort.shape[0]:
            break
        th = y_neg_sort[-n]
        h = (y_pos>th).sum()/y_pos.shape[0]
        print(f"Hits@{n}: {h:.4f}")

# here
def eval_products_nc(res_path):
    eval_set = set()
    data=torch.load(os.path.join(dataset_dir("products"), "processed_data.pt"), weights_only=False)
    labels=data.label_names
    ys=data.y.numpy().tolist()

    all_sample=0
    strict_correct=0
    overall_correct=0
    with open(res_path, 'r') as f:
        for line in f:
            if args.sample > 0 and all_sample >= args.sample:
                break
            all_sample+=1
            res = json.loads(line)
            if res['question_id'] in eval_set:
                print(f"{res['question_id']} repeat!!")
                return
            eval_set.add(res['question_id'])
            ans = res["text"].strip()
            y=ys[res["question_id"]][0]
            label=labels[y].strip()
            if label.lower()==ans.lower():
                strict_correct+=1
                overall_correct+=1
            elif label.lower() in ans.lower() and sum([l.lower() in ans.lower() for l in labels])<=2:
                overall_correct += 1

    overall_acc = overall_correct / all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")

def eval_products_nd(res_path):
    eval_set = set()
    data=torch.load(os.path.join(dataset_dir("products"), "processed_data.pt"), weights_only=False)
    labels=data.label_names
    ys=data.y.numpy().tolist()

    all_sample=0
    all_correct=0
    gt = []
    out = []
    with open(res_path, 'r') as f:
        for line in f:
            if args.sample > 0 and all_sample >= args.sample:
                break
            all_sample+=1
            res = json.loads(line)
            if res['question_id'] in eval_set:
                print(f"{res['question_id']} repeat!!")
            eval_set.add(res['question_id'])
            ans = res["text"].strip()
            y=ys[res["question_id"]][0]
            label=labels[y].strip()
            if label.lower() in ans.lower():
                all_correct+=1
            desc = data.raw_texts[res['question_id']]
            assistant_prompt = f"This is an amazon product which can be categorized as {label}. It can be described as {desc}"
            gt.append(assistant_prompt)
            out.append(ans)
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}acc: {all_acc:.4f}")

    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding = F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding = F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim = (gt_embedding * out_embedding).sum(1).mean().item()
    gt_sim_matrix = torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n = gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)] = 0
    gt_sim = (gt_sim_matrix.sum() / (n * (n - 1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")


def eval_pubmed_nc(res_path):
    data=torch.load(os.path.join(dataset_dir("pubmed"), "processed_data.pt"), weights_only=False)
    labels=data.label_texts
    short_labels = [l[18:] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    strict_correct=0
    overall_correct=0
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            short_label = short_labels[y]
            label=labels[y]
            if ans.lower().strip() == label.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower().strip() in ans.lower().strip() and sum([la.lower().strip() in ans.lower().strip() for la in short_labels]) == 1:
                overall_correct += 1
            if args.sample > 0 and all_sample >= args.sample:
                break

    overall_acc = overall_correct / all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")


def eval_pubmed_nd(res_path):
    data = torch.load(os.path.join(dataset_dir("pubmed"), "processed_data.pt"), weights_only=False)
    labels = data.label_texts
    short_labels = [l[18:] for l in labels]
    ys = data.y.numpy().tolist()

    titles = data.title
    abs = data.abs

    all_sample=0
    short_correct=0
    all_correct=0
    gt=[]
    out=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            id=res["question_id"]
            y=ys[id]
            short_label = short_labels[y]
            label=labels[y]
            if label.strip() in ans.strip():
                all_correct+=1
            if short_label in ans:
                short_correct+=1
            out.append(ans)
            gt.append(f"This is a paper in {label} domain, it's about {titles[id]}.")
    short_acc = short_correct/all_sample
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}\nshort_correct: {short_correct}\nshort_acc: {short_acc:.4f}\nall_correct: {all_correct}\nall_acc: {all_acc:.4f}")
    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding=F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding=F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim=(gt_embedding*out_embedding).sum(1).mean().item()
    gt_sim_matrix=torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n=gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)]=0
    gt_sim=(gt_sim_matrix.sum()/(n*(n-1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")


def eval_cora_nc(res_path):
    data=torch.load(os.path.join(dataset_dir("cora"), "processed_data.pt"), weights_only=False)
    labels=data.label_texts
    short_labels = [l.split('_')[0] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    correct=0
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            label=labels[y]
            short_label=short_labels[y]
            if short_label.strip().lower() in ans.strip().lower() and sum([l.strip().lower() in ans.strip().lower() for l in short_labels])==1:
                correct+=1
    acc=correct/all_sample
    print(f"Test samples: {all_sample}\nacc: {acc:.4f}")



def eval_cora_nd(res_path):
    data = torch.load(os.path.join(dataset_dir("cora"), "processed_data.pt"), weights_only=False)
    labels = data.label_texts
    ys = data.y.numpy().tolist()

    titles = data.title
    all_sample=0
    short_correct=0
    all_correct=0
    gt=[]
    out=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            id=res["question_id"]
            y=ys[id]
            label=labels[y]
            if label.strip() in ans.strip():
                all_correct+=1
                short_correct+=1
            out.append(ans)
            gt.append(f"This is a paper in {label} domain, it's about {titles[id]}.")
    short_acc = short_correct/all_sample
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}\nshort_correct: {short_correct}\nshort_acc: {short_acc:.4f}\nall_correct: {all_correct}\nall_acc: {all_acc:.4f}")
    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding=F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding=F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim=(gt_embedding*out_embedding).sum(1).mean().item()
    gt_sim_matrix=torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n=gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)]=0
    gt_sim=(gt_sim_matrix.sum()/(n*(n-1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="./results/llaga-opt-2.7b-v1-simteg_all_origin_tape_multihop-laplacian_-1-2-10-linear-only-train-pretrain_acc1_nc_test_nc.jsonl")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    func_dict = {
        "arxiv":{
            "nc": eval_arxiv_nc,
            "nd": eval_arxiv_nd,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "products": {
            "nc": eval_products_nc,
            "nd": eval_products_nd,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "pubmed": {
            "nc": eval_pubmed_nc,
            "nd": eval_pubmed_nd,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "cora": {
            "nc": eval_cora_nc,
            "nd": eval_cora_nd,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
    }
    # ----- generic NC evaluator (works for any dataset whose ocs_*.jsonl
    # carries the canonical label string under the "gt" key -- which all of
    # our pipelines do).  Uses substring match w/ uniqueness guard, the same
    # rule as eval_cora_nc.  Datasets registered here don't need a hand-
    # written evaluator.
    #
    # IMPORTANT: gen.py's gt strings are NOT guaranteed to match the class
    # strings embedded in the test prompt verbatim.  Known divergences:
    #   * cora:    gt "Case_Based"            vs prompt "Case Based"
    #              gt "Neural_Networks"       vs prompt "Neural Networks"
    #              ... (underscore vs space)
    #   * pubmed:  gt "Experimentally induced diabetes"
    #              vs prompt "Diabetes Mellitus Experimental"
    # The model is supervised on gt during training but at eval time we want
    # to recognize EITHER form (the synonym table below) AND we want
    # underscore<->space and case-insensitive matching to be permissive.
    _LABEL_SYNONYMS = {
        # left = whatever appears in gt, right = list of synonyms (incl. variants
        # the model might emit because the test prompt uses a different spelling).
        "experimentally induced diabetes": [
            "diabetes mellitus experimental",
            "experimentally induced diabetes",
        ],
        # pubmed: gt writes "Type1"/"Type2" (no space), but the model often
        # emits "Type 1"/"Type 2" -- both should count as the same label.
        "diabetes mellitus type1": [
            "diabetes mellitus type1", "diabetes mellitus type 1",
        ],
        "diabetes mellitus type2": [
            "diabetes mellitus type2", "diabetes mellitus type 2",
        ],
        # Add more as we discover them.
    }
    def _norm(s: str) -> str:
        # Lowercase, replace _ with space, collapse repeated whitespace, strip.
        return " ".join(s.replace("_", " ").lower().split()).strip()
    def _norm_compact(s: str) -> str:
        # Even stricter: also strip ALL whitespace.  Used as the fallback so
        # that "Type 1" and "Type1" and "type_1" all match.
        return _norm(s).replace(" ", "")
    def _label_variants(s: str):
        """Return all plausible spellings (in normalized form) for a gt label."""
        out = set()
        n = _norm(s)
        out.add(n)
        out.add(_norm(s.replace(" ", "_")))
        out.add(_norm(s.replace("_", " ")))
        for x in _LABEL_SYNONYMS.get(n, []):
            out.add(_norm(x))
        return [v for v in out if v]
    def _matches(label_variants, ans_norm: str, ans_compact: str) -> bool:
        for v in label_variants:
            if v in ans_norm:
                return True
            if _norm_compact(v) in ans_compact:
                return True
        return False
    def _eval_generic_nc(res_path):
        all_sample = 0
        correct = 0
        labels_set = set()
        with open(res_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                if "gt" in d:
                    labels_set.add(d["gt"].strip())
        labels = sorted(labels_set)
        labels_norm = {l: _label_variants(l) for l in labels}

        with open(res_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                ans_n = _norm(d["text"])
                ans_c = _norm_compact(d["text"])
                gt = d["gt"].strip()
                gt_hit = _matches(labels_norm[gt], ans_n, ans_c)
                # Uniqueness guard: count how many DIFFERENT labels are
                # matched.  We use the strict (compact) matcher to count
                # because that's where collisions between e.g.
                # "Diabetes Mellitus Type1" and "Diabetes Mellitus Type 1"
                # happen.  However "Type1" should NOT separately match
                # "Type2" -- and the substring _matches() does the right
                # thing because the variants already include the digit.
                hits = 0
                for l, vs in labels_norm.items():
                    if _matches(vs, ans_n, ans_c):
                        hits += 1
                if gt_hit and hits == 1:
                    correct += 1
                all_sample += 1
        acc = correct / max(all_sample, 1)
        print(f"Test samples: {all_sample}\nacc: {acc:.4f}")

    # Datasets that don't yet have a hand-written nc evaluator fall back to
    # the generic one.  Same for any other reasonable fallback dataset.
    for ds in ["reddit", "instagram", "computer", "history", "photo",
               "wikics", "citeseer", "cornell", "texas", "wisconsin",
               "washington", "bookchild", "sportsfit"]:
        func_dict.setdefault(ds, {})
        func_dict[ds].setdefault("nc", _eval_generic_nc)
        func_dict[ds].setdefault("lp", eval_lp)
        func_dict[ds].setdefault("lprank", eval_lprank)
    # Force-route cora and pubmed nc through the generic evaluator too.
    # The hand-written ones use lookup-by-question_id into processed_data.pt
    # which is brittle (relies on the same y vector across train/eval) and
    # they don't apply our _LABEL_SYNONYMS table -- so e.g. vicuna's
    # "Diabetes Mellitus Type 2" output gets scored 0.36 by eval_pubmed_nc
    # while the generic evaluator gives ~0.85.
    func_dict.setdefault("cora", {})["nc"] = _eval_generic_nc
    func_dict.setdefault("pubmed", {})["nc"] = _eval_generic_nc

    func=func_dict[args.dataset][args.task]
    func(args.res_path)
