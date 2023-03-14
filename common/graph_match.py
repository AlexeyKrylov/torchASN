from copy import deepcopy


class Node:
    def __init__(self, value):
        self.value = value
        self.predicat2next_node = {}

    def set_next_node(self, predicat, node):
        self.predicat2next_node[predicat] = node

    def __eq__(self, other):
        if self.value == other.value:
            return True


class Graph:
    def __init__(self, text=None):
        self.str2nodes = {}
        self.all_units = {}

        if text:
            self.build_from_str(text)

    def build_from_str(self, text):
        for triplet in text.split('.'):
            triplet = triplet.strip().split()
            if len(triplet) != 3:
                continue
            subj, pred, obj = triplet
            cur_node = self.str2nodes.setdefault(subj, Node(subj))
            next_node = self.str2nodes.setdefault(obj, Node(obj))
            cur_node.set_next_node(pred, next_node)
            self.all_units[subj] = 0
            self.all_units[pred] = 0
            self.all_units[obj] = 0

    def get_metric(self, other_graph):
        marked = deepcopy(self.all_units)
        for node_str, node in self.str2nodes.items():
            one_match = False
            other_node = other_graph.str2nodes.get(node_str, Node(''))
            for pred, next_node in node.predicat2next_node.items():
                if other_node.predicat2next_node.get(pred, Node('')) == next_node:
                    marked[next_node.value] = 1
                    marked[pred] = 1
                    one_match = True
            if one_match:
                marked[node.value] = 1
        res = 0 if not marked else sum([v for v in marked.values()]) / len(marked)
        return res