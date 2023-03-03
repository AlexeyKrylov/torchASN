import pandas as pd
import json
import re
import numpy as np


def select_splitter(to_split: str) -> str:
    if to_split[:3] == 'ask': return to_split
    replace_from = to_split.find("select ") + len("select ")
    replace_to = to_split.find(" where ")
    select_expr_splitted = []
    start_loc, finish_loc, current_loc = 0, 1, 0

    adjust_str = to_split[replace_from:replace_to].split()

    while adjust_str:
        if current_loc >= len(adjust_str): break

        if adjust_str[current_loc] == "distinct":
            finish_loc += 1

        elif adjust_str[current_loc] in ["count", 'sum', 'avg', 'min', 'max']:
            finish_loc += 2

        elif adjust_str[current_loc] == "(":
            finish_loc += 4

        else:

            splitted = adjust_str[start_loc:finish_loc]

            if len(splitted) != 1:
                select_expr_splitted.append(" ".join(splitted))
            else:
                select_expr_splitted.append(splitted[0])
            start_loc = finish_loc
            current_loc = finish_loc
            finish_loc += 1
            continue

        current_loc += 1

    replacement = " , ".join(select_expr_splitted) if len(select_expr_splitted) > 1 else select_expr_splitted[0]

    before_change = to_split[:replace_from - 1]

    after_change = to_split[replace_to + 1:]

    return " ".join([before_change, replacement, after_change])


def change_quotes(match, x):
    for sub_str in match:

        start, end = sub_str.span()
        changed_x = x[start: end].strip()

        if not re.findall(r"^((?!filter).)*$", changed_x):  # 'p' ) ) . filter ( lang ( ?OBJ_3 ) = 'en'
            continue

        changed_x = changed_x.replace(" . ", ".")
        changed_x = changed_x.replace(" = ", "=")
        changed_x = changed_x.replace(" , ", ",")

        x = " ".join([x[:start], changed_x, x[end:]])
    return x


def numbers_adj(x):
    # '34 . 196944444444 36 . 3525' -> ' 34.196944444444 36.3525'
    sub_strs = re.findall(r"[\s|'][-]?[0-9]*\s\.\s[0-9]+(?:[eE][-+]?[0-9]+)?", x)

    for to_replace in sub_strs:
        replace_with = "".join([" "] + [i for i in to_replace if (i.isdigit() or i in ['.', "'", "-"])])
        x = x.replace(to_replace, replace_with)

    else:
        # '   34 . 196944444444 36 . 3525' -> '34 . 196944444444 36 . 3525'
        quotes_spaces = re.findall(r"['].*[']", x)

        for to_replace in quotes_spaces:
            replace_with = "".join(["'", to_replace[1:-1].strip(), "'"])
            x = x.replace(to_replace, replace_with)

    return x

def parse_json(path, spec_path, src_path):

    data = pd.DataFrame(json.load(open(path, 'rb')))
    dl = list()
    for i in ['t1417656573', 't1410874016', 'wd:P5388', 'wd:P3633', 't1270953452', 'wd: ', '\xe7', '\uc2a4', "\n", "\xe9", '\u2642', '\xf9', '\xe4', '\xe1', '\u0b87', '\xfc', '\u0259',
              '\u05d9', '\u05d0', '\xf6', '\xe5', '\u09b6', '\ub2e8', '\u03b2', '\u672c', '\xe6', '\xed', '\u010d', '\xd7', '\u0627', '-num_value_1', '-num_value_2', 'transtulit', '男', '|',
              '\u014d', '\xf3', '\xe8', '\u0101', '\xc9', '\u0161', '\u0107', '\xf8', '\u0142', '\u0151', '\xd6', '\xc1', '\xe3', '\xc4', '\xfa', '\xe2', '\xea', '\xef', '\xfd', '\xdf',
              '\u0144', '\xf1', '\u016b', '\xd3', '\xf2', '\u011b', '\xe0', '\u0103', '\xc6', '\u0160', '\u017e', '\u010c', '\u30eb', '\xb4', '\u0148']:
        dl.extend(list(data[data.masked_sparql.str.find(i) != -1].masked_sparql.index))
        dl.extend(list(data[data.question.str.find(i) != -1].question.index))
    drop_list = np.array(list(set(dl)))
    print(drop_list)
    data = data.drop(drop_list).reset_index(drop=True)
    data = data.iloc[data.question.str.lower().drop_duplicates().index]
    X = data["question"]
    Y = data["masked_sparql"]

    # adjustments
    whrong_filter_sntx = Y[Y.str.contains(' filter', regex=False)][~Y.str.contains(" . filter", regex=False)].index
    Y.loc[whrong_filter_sntx] = Y.loc[whrong_filter_sntx].str.replace("filter", ". filter", regex=False)

    without_where_sntx = Y[~Y.str.contains("where {", regex=False)].index
    Y.loc[without_where_sntx] = Y.loc[without_where_sntx].str.replace("{", "where {", regex=False)

    bad_where_close_sntx = Y[Y.str.contains(' }', regex=False)][~Y.str.contains(r" . }", regex=False)].index
    Y.loc[bad_where_close_sntx] = Y.loc[bad_where_close_sntx].str.replace("}", ". }", regex=False)

    Y = Y.apply(lambda x: numbers_adj(x))
    Y = Y.apply(select_splitter)

    Y = Y.str.replace("  ", " ")
    Y = Y.apply(lambda x: change_quotes(re.finditer(r"\s['].{1,20}[']\s", x), x))  # Low scale
    Y = Y.apply(lambda x: change_quotes(re.finditer(r"\s['].{1,60}[']\s", x), x))  # Large scale
    Y = Y[~Y.str.contains(" ")]
    X = X.loc[Y.index]
    X = X.str.strip()

    input_to_save = [(Y, spec_path),
                     (X, src_path)]

    for inp in input_to_save:
        open(inp[1], "wb").close()
        with open(inp[1], "a", encoding="utf8") as file:
            print(*inp[0].to_list(), sep="\n", file=file)
