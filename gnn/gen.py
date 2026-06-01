from seq import generate_final_sequence, build_adjacency_dict
import json
import torch
from tqdm import tqdm


import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import processed_data_path  # noqa: E402
def cora_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.2, beta=0.55):
    tens = torch.load(processed_data_path('cora'), weights_only=False)
    titles = tens['title']
    abs = tens['abs']

    label_dict = {2: "Neural Networks", 5: "Rule Learning", 4: "Reinforcement Learning", 3: "Probabilistic Methods",
                  6: "Theory", 1: "Genetic Algorithms", 0: "Case Based"}

    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            title = titles[int(center_node_id)]
            abstract = abs[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a paper."
                            f"The 0th node is the central node, with the following information: Title: {title}, Abstract: {abstract}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 7 classes: Case Based, Genetic Algorithms, Neural Networks, Probabilistic Methods, Reinforcement Learning, Rule Learning, Theory, please tell me which class the center node belongs to?")
                    },
                    {"from": "gpt", "value": label}
                ]
            }

            json_line = json.dumps(data)
            # print(json_line)
            f.write(json_line + '\n')

        return


def pubmed_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.2, beta=0.55):
    tens = torch.load(processed_data_path('pubmed'), weights_only=False)
    titles = tens['title']
    abs = tens['abs']

    label_dict = {0: 'Experimentally induced diabetes', 1: 'Diabetes Mellitus Type1', 2: 'Diabetes Mellitus Type2'}

    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            title = titles[int(center_node_id)]
            abstract = abs[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a paper."
                            f"The 0th node is the central node, with the following information: Title: {title}, Abstract: {abstract}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node, into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to?")
                    },
                    {"from": "gpt", "value": label}
                ]
            }

            json_line = json.dumps(data)
            # print(json_line)
            f.write(json_line + '\n')

        return


def arxiv_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.1, beta=0.55):
    tens = torch.load(processed_data_path('arxiv'), weights_only=False)
    titles = tens['title']
    abs = tens['abs']

    label_dict = {0: 'Numerical Analysis', 1: 'Multimedia', 2: 'Logic in Computer Science',
                  3: 'Computers and Society',
                  4: 'Cryptography and Security', 5: 'Distributed, Parallel, and Cluster Computing',
                  6: 'Human-Computer Interaction',
                  7: 'Computational Engineering, Finance, and Science',
                  8: 'Networking and Internet Architecture',
                  9: 'Computational Complexity', 10: 'Artificial Intelligence',
                  11: 'Multiagent Systems',
                  12: 'General Literature', 13: 'Neural and Evolutionary Computing',
                  14: 'Symbolic Computation',
                  15: 'Hardware Architecture', 16: 'Computer Vision and Pattern Recognition',
                  17: 'Graphics',
                  18: 'Emerging Technologies', 19: 'Systems and Control',
                  20: 'Computational Geometry', 21: 'Other Computer Science',
                  22: 'Programming Languages', 23: 'Software Engineering', 24: 'Machine Learning',
                  25: 'Sound', 26: 'Social and Information Networks', 27: 'Robotics',
                  28: 'Information Theory',
                  29: 'Performance', 30: 'Computation and Language', 31: 'Information Retrieval',
                  32: 'Mathematical Software',
                  33: 'Formal Languages and Automata Theory', 34: 'Data Structures and Algorithms',
                  35: 'Operating Systems',
                  36: 'Computer Science and Game Theory', 37: 'Databases', 38: 'Digital Libraries',
                  39: 'Discrete Mathematics'}
    adjacency_dict = build_adjacency_dict(adj_lists)

    with open(train_path, 'w') as f:
        # for i in range(len(train_indices)):
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            title = titles[int(center_node_id)]
            abstract = abs[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered citation graph: <graph>,each node represents a paper."
                            f"The 0th node is the central node, with the following information: Title: {title}, Abstract: {abstract}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 40 classes: Numerical Analysis, Multimedia, Logic in Computer Science, Computers and Society, Cryptography and Security, Distributed, Parallel, and Cluster Computing, Human-Computer Interaction, Computational Engineering, Finance, and Science, Networking and Internet Architecture, Computational Complexity, Artificial Intelligence, Multiagent Systems, General Literature, Neural and Evolutionary Computing, Symbolic Computation, Hardware Architecture, Computer Vision and Pattern Recognition, Graphics, Emerging Technologies, Systems and Control, Computational Geometry, Other Computer Science, Programming Languages, Software Engineering, Machine Learning, Sound, Social and Information Networks, Robotics, Information Theory, Performance, Computation and Language, Information Retrieval, Mathematical Software, Formal Languages and Automata Theory, Data Structures and Algorithms, Operating Systems, Computer Science and Game Theory, Databases, Digital Libraries, Discrete Mathematics, please tell me which class the center node belongs to?")
                    },
                    {"from": "gpt", "value": label}
                ]
            }

            json_line = json.dumps(data)
            # print(json_line)
            f.write(json_line + '\n')

        return


def history_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.05, beta=0.55):
    tens = torch.load(processed_data_path('history'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: 'World', 1: 'Americas', 2: 'Asia', 3: 'Military', 4: 'Europe', 5: 'Russia', 6: 'Africa',
                  7: 'Ancient Civilizations', 8: 'Middle East', 9: 'Historical Study & Educational Resources',
                  10: 'Australia & Oceania', 11: 'Arctic & Antarctica'}
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a history book."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 12 classes: World, Americas, Asia, Military, Europe, Russia, Africa, Ancient Civilizations, Middle East, Historical Study & Educational Resources, Australia & Oceania, Arctic & Antarctica ,please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }

            json_line = json.dumps(data)
            # print(json_line)
            f.write(json_line + '\n')



def photo_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.05, beta=0.55):
    tens = torch.load(processed_data_path('photo'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: 'Video Surveillance', 1: 'Accessories', 2: 'Binoculars & Scopes', 3: 'Video',
                  4: 'Lighting & Studio', 5: 'Bags & Cases', 6: 'Tripods & Monopods', 7: 'Flashes',
                  8: 'Digital Cameras', 9: 'Film Photography',
                  10: 'Lenses', 11: 'Underwater Photography'}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a product and its user review."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 12 classes: Video Surveillance, Accessories, Binoculars & Scopes, Video, Lighting & Studio, Bags & Cases, Tripods & Monopods, Flashes, Digital Cameras, Film Photography, Lenses, Underwater Photography, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            # print(json_line)
            f.write(json_line + '\n')

        return


def computer_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.05, beta=0.55):
    tens = torch.load(processed_data_path('computer'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: 'Computer Accessories & Peripherals', 1: 'Tablet Accessories', 2: 'Laptop Accessories',
                  3: 'Computers & Tablets', 4: 'Computer Components', 5: 'Data Storage', 6: 'Networking Products',
                  7: 'Monitors', 8: 'Servers', 9: 'Tablet Replacement Parts'}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a product and its user review."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 10 classes: Computer Accessories & Peripherals, Tablet Accessories, Laptop Accessories, Computers & Tablets, Computer Components, Data Storage, Networking Products, Monitors, Servers, Tablet Replacement Parts, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            # print(json_line)
            f.write(json_line + '\n')

        return


def wikics_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.1, beta=0.55):
    tens = torch.load(processed_data_path('wikics'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: "Computational linguistics", 1: "Databases", 2: "Operating systems", 3: "Computer architecture",
                  4: "Computer security", 5: "Internet protocols", 6: "Computer file systems",
                  7: "Distributed computing architecture", 8: "Web technology", 9: "Programming language topics"}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            center_node_id = int(center_node_id)
                
            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents an article on wikipedia about computer science."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 10 classes: Computational linguistics, Databases, Operating systems, Computer architecture, Computer security, Internet protocols, Computer file systems, Distributed computing architecture, Web technology, Programming language topics, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            f.write(json_line + '\n')

        return


def instagram_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.1, beta=0.55):
    tens = torch.load(processed_data_path('instagram'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: "Normal Users", 1: "Commercial Users"}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a user from instagram."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 2 classes: Normal Users, Commercial Users, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            f.write(json_line + '\n')

        return


def cornell_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.1, beta=0.55):
    tens = torch.load(processed_data_path('cornell'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: "student", 1: "project", 2: "course", 3: "staff", 4: "faculty"}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a webpage from a university."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 5 classes: student, project, course, staff, faculty, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            f.write(json_line + '\n')

        return
            

def wisconsin_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.1, beta=0.55):
    tens = torch.load(processed_data_path('wisconsin'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: "project", 1: "faculty", 2: "course", 3: "student", 4: "staff"}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a webpage from a university."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 5 classes: student, project, course, staff, faculty, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            f.write(json_line + '\n')


def washington_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.1, beta=0.55):
    tens = torch.load(processed_data_path('washington'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: "project", 1: "course", 2: "student", 3: "faculty", 4: "staff"}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a webpage from a university."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 5 classes: student, project, course, staff, faculty, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            print(json_line)
            f.write(json_line + '\n')


def texas_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.1, beta=0.55):
    tens = torch.load(processed_data_path('texas'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: "course", 1: "project", 2: "student", 3: "faculty", 4: "staff"}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a webpage from a university."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 5 classes: student, project, course, staff, faculty, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            f.write(json_line + '\n')


def reddit_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.1, beta=0.55):
    tens = torch.load(processed_data_path('reddit'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: "Normal Users", 1: "Popular Users"}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>,each node represents a user on Reddit."
                            f"The 0th node is the central node, with the following posts: {raw_text}"
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 2 classes: Normal Users, Popular Users, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            f.write(json_line + '\n')


def citeseer_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped, labels, train_path, threshold=0.2, beta=0.55):
    tens = torch.load(processed_data_path('citeseer'), weights_only=False)
    raw_texts = tens['raw_texts']

    label_dict = {0: "Agents", 1: "Machine Learning", 2: "Information Retrieval", 3: "Databases",
                  4: "Human-Computer Interaction", 5: "Artificial Intelligence"}
    
    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs", unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])

            graph = generate_final_sequence(center_node_id, node_embeddings, adjacency_dict, threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]

            raw_text = raw_texts[int(center_node_id)]

            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered citation graph: <graph>,each node represents a paper."
                            f"The 0th node is the central node, with the following information: Title: , Abstract: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into 6 classes: Agents, Machine Learning, Information Retrieval, Databases, Human-Computer Interaction, Artificial Intelligence, please tell me which class the center node belongs to?"),
                    },
                    {"from": "gpt", "value": label}
                ]
            }
            json_line = json.dumps(data)

            # print(json_line)
            f.write(json_line + '\n')


# ---------------------------------------------------------------------------
# bookchild & sportsfit (CSTAG / Graph-COM TAG datasets).
# These datasets ship with their own `label_texts` list, so the label dict is
# built dynamically.  The prompt mirrors the e-commerce style of history/photo.
# ---------------------------------------------------------------------------
def bookchild_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped,
                    labels, train_path, threshold=0.05, beta=0.55):
    tens = torch.load(processed_data_path('bookchild'), weights_only=False)
    raw_texts = tens['raw_texts']
    label_texts = list(tens['label_texts'])
    label_dict = {i: t for i, t in enumerate(label_texts)}
    classes_str = ", ".join(label_texts)

    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs",
                      unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])
            graph = generate_final_sequence(center_node_id, node_embeddings,
                                            adjacency_dict,
                                            threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]
            raw_text = raw_texts[int(center_node_id)]
            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>, each node represents a children's book."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into {len(label_texts)} classes: {classes_str}, please tell me which class the center node belongs to?"
                        ),
                    },
                    {"from": "gpt", "value": label},
                ],
            }
            f.write(json.dumps(data) + '\n')


def sportsfit_input(train_indices, node_embeddings, adj_lists, adj_lists_dropped,
                    labels, train_path, threshold=0.05, beta=0.55):
    tens = torch.load(processed_data_path('sportsfit'), weights_only=False)
    raw_texts = tens['raw_texts']
    label_texts = list(tens['label_texts'])
    label_dict = {i: t for i, t in enumerate(label_texts)}
    classes_str = ", ".join(label_texts)

    adjacency_dict = build_adjacency_dict(adj_lists)
    with open(train_path, 'w') as f:
        for i in tqdm(range(len(train_indices)), desc="Processing graphs",
                      unit="graph", ascii=True, ncols=100):
            center_node_id = int(train_indices[i])
            graph = generate_final_sequence(center_node_id, node_embeddings,
                                            adjacency_dict,
                                            threshold=threshold, beta=beta)
            graph = [int(x) for x in graph]
            raw_text = raw_texts[int(center_node_id)]
            label = label_dict[int(labels[center_node_id])]
            data = {
                "id": center_node_id,
                "graph": graph,
                "conversations": [
                    {
                        "from": "human",
                        "value": (
                            f"Given a node-centered graph: <graph>, each node represents a sports & fitness product."
                            f"The 0th node is the central node, with the following information: {raw_text}."
                            f"The graph consists of the center node, the neighbors of the center node, and the context-relevant nodes."
                            f"We need to classify the center node into {len(label_texts)} classes: {classes_str}, please tell me which class the center node belongs to?"
                        ),
                    },
                    {"from": "gpt", "value": label},
                ],
            }
            f.write(json.dumps(data) + '\n')
