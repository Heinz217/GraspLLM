import sys
from utils.paths import dataset_dir
sys.path.append("./")
sys.path.append("./utils")
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from utils.constants import GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_PAD_ID, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from utils.conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from utils.utils import disable_torch_init, tokenizer_graph_token, get_model_name_from_path
from torch_geometric.utils import k_hop_subgraph, degree, remove_self_loops, add_self_loops
from torch_geometric.nn import MessagePassing
import math

SMALL_DATASETS=["pubmed", "cora"]


class MP(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


# def get_chunk(lst, n, k):
#     chunks = split_list(lst, n)
#     return chunks[k]

def load_pretrain_embedding_graph(data_dir):
    """Load Qwen3-Embedding-8B node features (matches train.py)."""
    emb_path = os.path.join(data_dir, "qwen3_emb_x.pt")
    obj = torch.load(emb_path, map_location="cpu", weights_only=False)
    emb = obj["emb"] if isinstance(obj, dict) else obj
    return emb.float()


def load_pretrain_embedding_hop_lp(data_dir, pretrained_embedding_type, hop):
    mask = torch.load(os.path.join(data_dir, f"no_test_link_mask.pt"), weights_only=False)
    if pretrained_embedding_type == "simteg":
        simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"), weights_only=False)[mask]] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x_notestlink.pt"), weights_only=False) for i in range(1, hop + 1)]
        simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"), weights_only=False)[mask]] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x_notestlink.pt"), weights_only=False) for i in range(1, hop + 1)]
        simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"), weights_only=False)[mask]] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x_notestlink.pt"), weights_only=False) for i in range(1, hop + 1)]
        pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
    else:
        pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"), weights_only=False)[mask]]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x_notestlink.pt"), weights_only=False) for i in range(1, hop+1)]

    return pretrained_embs, mask

def eval_model(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()
    # Install our unified chat-template onto the tokenizer (matches the one
    # used during training).
    from utils.chat_format import (install_chat_template, build_eval_prompt,
                                    stop_token_ids)
    backbone_kind = install_chat_template(
        tokenizer, model_name_or_path=(args.model_base or args.model_path))
    print(f"[chat-format] installed kind={backbone_kind!r}")
    # data_dir=os.path.expanduser(args.data_dir)
    if args.dataset == "arxiv":
        data_dir = dataset_dir("arxiv")
    elif args.dataset == "products":
        data_dir = dataset_dir("products")
    elif args.dataset == "pubmed":
        data_dir = dataset_dir("pubmed")
    elif args.dataset == "cora":
        data_dir = dataset_dir("cora")
    elif args.dataset == "history":
        data_dir = dataset_dir("history")
    elif args.dataset == "computer":
        data_dir = dataset_dir("computer")
    elif args.dataset == "photo":
        data_dir = dataset_dir("photo")
    elif args.dataset == "instagram":
        data_dir = dataset_dir("instagram")
    elif args.dataset == "wikics":
        data_dir = dataset_dir("wikics")
    elif args.dataset == "wisconsin":
        data_dir = dataset_dir("wisconsin")
    elif args.dataset == "cornell":
        data_dir = dataset_dir("cornell")
    elif args.dataset == "reddit":
        data_dir = dataset_dir("reddit")
    elif args.dataset == "washington":
        data_dir = dataset_dir("washington")
    elif args.dataset == "texas":
        data_dir = dataset_dir("texas")
    elif args.dataset == "citeseer":
        data_dir = dataset_dir("citeseer")
    elif args.dataset == "bookchild":
        data_dir = dataset_dir("bookchild")
    elif args.dataset == "sportsfit":
        data_dir = dataset_dir("sportsfit")
    else:
        print(f"{args.dataset} not exists")
        raise ValueError
    if args.task in  ["nc", "nd", "nda", "nctext"]:
        prompt_file = os.path.join(data_dir, f"ocs_test.jsonl")
        data_path = os.path.join(data_dir, f"processed_data.pt")
    elif args.task in ["lp"]:
        prompt_file = os.path.join(data_dir, f"ocs_test.jsonl")
        data_path = os.path.join(data_dir, f"processed_data.pt")
    else:
        raise ValueError

    # Allow --test_path to override the default ocs_test.jsonl, e.g. for
    # quick subset evals or for evaluating with custom prompt formats.
    if args.test_path:
        prompt_file = args.test_path
        print(f"[eval] --test_path overrides default; using {prompt_file}")

    data = torch.load(data_path, weights_only=False)
    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if "tmp" not in args.answers_file and os.path.exists(answers_file):
        line_number = len(open(answers_file, 'r').readlines())
        print(f"{args.answers_file} already exists! it has {line_number} lines!!")
        if line_number >= len(lines):
            return
        lines = lines[line_number:]
        ans_file = open(answers_file, "a")
    else:
        ans_file = open(answers_file, "w")

    questions = [json.loads(q) for q in lines]

    index = None
    pretrained_emb = load_pretrain_embedding_graph(data_dir)
    structure_emb = None

    for line in tqdm(questions):
        idx = line["id"]
        if args.task in ["nd", "nda"]:
            qs=f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
        elif args.task == "nc":
            if args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to?"
            else:
                qs = line["conversations"][0]['value']
        elif args.task == "nctext":
            text = data.raw_texts[line['id']]
            text = text[:2000]
            if args.dataset == "arxiv":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "products":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent products sold in Amazon, and edges between products indicate they are purchased together, the node feature of center node is {text}. We need to classify the center node into 47 classes: Home & Kitchen, Health & Personal Care, Beauty, Sports & Outdoors, Books, Patio, Lawn & Garden, Toys & Games, CDs & Vinyl, Cell Phones & Accessories, Grocery & Gourmet Food, Arts, Crafts & Sewing, Clothing, Shoes & Jewelry, Electronics, Movies & TV, Software, Video Games, Automotive, Pet Supplies, Office Products, Industrial & Scientific, Musical Instruments, Tools & Home Improvement, Magazine Subscriptions, Baby Products, label 25, Appliances, Kitchen & Dining, Collectibles & Fine Art, All Beauty, Luxury Beauty, Amazon Fashion, Computers, All Electronics, Purchase Circles, MP3 Players & Accessories, Gift Cards, Office & School Supplies, Home Improvement, Camera & Photo, GPS & Navigation, Digital Music, Car Electronics, Baby, Kindle Store, Buy a Kindle, Furniture & D&#233;cor, #508510, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "pubmed":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers about Diabetes and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to? Direct tell me the class name."
            elif args.dataset == "cora":
                qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to? Direct tell me the class name."
            else:
                raise ValueError
        elif args.task == "lp":
            qs = line["conversations"][0]['value']
        else:
            print(f"NOT SUPPORT {args.task}!!!")
            raise ValueError
        cur_prompt = qs

        # Build the eval prompt via apply_chat_template (matches train).
        input_ids = build_eval_prompt(tokenizer, qs, has_graph=True).unsqueeze(0).cuda()

        if not isinstance(line['graph'][0], list):
            line['graph'] = [line['graph']]
        graph = torch.LongTensor(line['graph'])
        mask = graph != DEFAULT_GRAPH_PAD_ID
        masked_graph_emb = pretrained_emb[graph[mask]]
        s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
        graph_emb = torch.zeros((s, n, d))
        graph_emb[mask] = masked_graph_emb
        if structure_emb is not None:
            graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)

        # Stop tokens: EOS + any chat-template-specific terminator (<|im_end|>,
        # <|eot_id|>, </s>).  This is what tells generate() to stop, and works
        # even if the trained ckpt didn't perfectly learn to emit EOS.
        _stop_ids = stop_token_ids(tokenizer) or None

        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    do_sample=False,         # greedy: deterministic + faster
                    num_beams=1,
                    max_new_tokens=32,        # cora labels are <= 7 tokens
                    eos_token_id=_stop_ids,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0),
                    use_cache=True)

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
        except Exception as e:
            print(f"!!!!!!Error!!!!! {e}")
            outputs=""

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "graph": line['graph'],
                                   "text": outputs,
                                   "gt":line["conversations"][1]['value'],
                                   "answer_id": ans_id}) + "\n")
        ans_file.flush()
    ans_file.close()
    
def llm_node_embedding(args):
    # Model
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(f"Loaded from {model_path}. Model Base: {args.model_base}")
    tokenizer, model, context_len = load_pretrained_model(model_path, args.model_base, model_name,
                                                          cache_dir=args.cache_dir)
    model = model.to(torch.float16).cuda()

    # 设置数据目录
    if args.dataset == "arxiv":
        data_dir = dataset_dir("arxiv")
    elif args.dataset == "products":
        data_dir = dataset_dir("products")
    elif args.dataset == "pubmed":
        data_dir = dataset_dir("pubmed")
    elif args.dataset == "cora":
        data_dir = dataset_dir("cora")
    elif args.dataset == "history":
        data_dir = dataset_dir("history")
    elif args.dataset == "computer":
        data_dir = dataset_dir("computer")
    elif args.dataset == "photo":
        data_dir = dataset_dir("photo")
    elif args.dataset == "instagram":
        data_dir = dataset_dir("instagram")
    elif args.dataset == "wikics":
        data_dir = dataset_dir("wikics")
    elif args.dataset == "wisconsin":
        data_dir = dataset_dir("wisconsin")
    elif args.dataset == "washington":
        data_dir = dataset_dir("washington")
    elif args.dataset == "texas":
        data_dir = dataset_dir("texas")
    elif args.dataset == "washington":
        data_dir = dataset_dir("washington")
    elif args.dataset == "cornell":
        data_dir = dataset_dir("cornell")
    elif args.dataset == "reddit":
        data_dir = dataset_dir("reddit")
    else:
        print(f"{args.dataset} not exists")
        raise ValueError

    data_path = os.path.join(data_dir, f"processed_data.pt")
    data = torch.load(data_path, weights_only=False)

    prompt_file = os.path.join(data_dir, f"ocs_test.jsonl")
    if args.test_path:
        prompt_file = args.test_path
        print(f"[eval/lp] --test_path overrides default; using {prompt_file}")

    print(f"Load from {prompt_file}\n")
    lines = open(prompt_file, "r").readlines()

    # 限制处理数量
    if args.start >= 0:
        if args.end < 0:
            args.end = len(lines)
        lines = lines[args.start:args.end]
    elif args.end > 0:
        lines = lines[:args.end]

    questions = [json.loads(q) for q in lines]

    pretrained_emb = load_pretrain_embedding_graph(data_dir, args.pretrained_embedding_type)
    structure_emb = None

    embedding_dim = model.config.hidden_size  
    all_node_embeddings = torch.zeros((len(questions), embedding_dim), dtype=torch.float16)
    node_indices = []

    print(f"Extracting node embeddings for {len(questions)} nodes...")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        node_indices.append(idx)

        qs = f"Please represent the center node of {DEFAULT_GRAPH_TOKEN} as an embedding."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_graph_token(prompt, tokenizer, GRAPH_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        if not isinstance(line['graph'][0], list):
            line['graph'] = [line['graph']]

        graph = torch.LongTensor(line['graph'])
        mask = graph != DEFAULT_GRAPH_PAD_ID
        masked_graph_emb = pretrained_emb[graph[mask]]
        s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
        graph_emb = torch.zeros((s, n, d))
        graph_emb[mask] = masked_graph_emb
        if structure_emb is not None:
            graph_emb = torch.cat([graph_emb, structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)

        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids,
                    graph_emb=graph_emb.half().cuda(),
                    graph=graph.cuda(),
                    output_hidden_states=True,
                    return_dict=True
                )

                last_hidden_state = outputs.hidden_states[-1]

                node_embedding = last_hidden_state.mean(dim=1).squeeze()
                all_node_embeddings[i] = node_embedding

        except Exception as e:
            print(f"Error processing node {idx}: {e}")
            all_node_embeddings[i] = torch.zeros(embedding_dim, dtype=torch.float16)

    node_embedding_dict = {idx: emb for idx, emb in zip(node_indices, all_node_embeddings)}

    output_file = os.path.join(data_dir, f"emb/node_embeddings.pt")
    torch.save(node_embedding_dict, output_file)
    print(f"Node embeddings saved to {output_file}")

    output_tensor_file = os.path.join(data_dir, f"emb/node_embeddings_tensor.pt")
    torch.save(all_node_embeddings, output_tensor_file)
    print(f"Node embedding tensor saved to {output_tensor_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--pretrained_embedding_type", type=str, default="sbert")
    parser.add_argument("--answers_file", type=str, default="answer.jsonl")
    parser.add_argument("--conv_mode", type=str, default="v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--mm_use_graph_start_end",default=False, action="store_true")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--cache_dir", type=str, default="../../checkpoint")
    args = parser.parse_args()
    
    if args.task == "node_embedding":
        llm_node_embedding(args)
    else:
        eval_model(args)
