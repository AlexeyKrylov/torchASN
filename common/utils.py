# coding=utf-8
import os
import re
from common.graph_match import Graph

def exact_match(pred_sparql, true_sparql):
    pred_sparql = pred_sparql.lower()
    true_sparql = true_sparql.lower()
    exact_match_status = 0
    if pred_sparql == true_sparql:
        exact_match_status = 1
    return exact_match_status
def calculate_batch_metrics(pred_sparql_list, true_sparql_list):
    func_dict = {'exact_match': exact_match,
                 'graph_match': graph_match}
    result_dict = {key: 0 for key in func_dict}
    for idx in range(len(true_sparql_list)):
        for eval_func_name, eval_func in func_dict.items():
            result_dict[eval_func_name] += eval_func(pred_sparql_list[idx], true_sparql_list[idx])

    for key in result_dict:
        result_dict[key] /= len(true_sparql_list)

    return result_dict

def graph_match(pred_sparql, true_sparql):
    true_triplet = get_triplet_from_sparql(true_sparql)
    pred_triplet = get_triplet_from_sparql(pred_sparql)
    graph1 = Graph(true_triplet)
    graph2 = Graph(pred_triplet)
    return graph1.get_metric(graph2)
def get_triplet_from_sparql(sparql_query):
    triplet = re.findall(r"{(.*?)}", sparql_query)
    if triplet:
        triplet = triplet[0].split()
        triplet = ' '.join([elem for elem in triplet if elem]).strip()
    else:
        triplet = ''
    return triplet
class TXTLogger:
    def __init__(self, work_dir):
        self.save_dir = work_dir
        self.filename = "progress_log.txt"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.log_file_path = os.path.join(self.save_dir, self.filename)
        log_file = open(self.log_file_path, 'w')
        log_file.close()

    def log(self, data):
        with open(self.log_file_path, 'a') as f:
            f.write(f'{str(data)}\n')