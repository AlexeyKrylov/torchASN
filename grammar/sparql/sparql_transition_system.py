# coding=utf-8

from grammar.transition_system import TransitionSystem
from collections import  deque
from grammar.dsl_ast import RealizedField, AbstractSyntaxTree

"""
SPARQL Transition system.
This code contains functions that implements:
1. Code to ASDL tree parsing
2. ASDL tree code processing
3. ASDL trees comparison
"""


def build_ast_from_toks(grammar, token_subset, rule):
    children = []
    production = grammar.get_prod_by_ctr_name(rule)

    if rule in ["QUERY"]:
        # SELECT or ASK
        if "select" in token_subset:
            select_tokens = token_subset[1:token_subset.index('where')]

            select_tokens = " ".join(select_tokens)

            select_node = build_ast_from_toks(grammar, select_tokens, 'SELECT')
            select_or_ask_field = RealizedField(production['select_or_ask'], select_node)
        elif "ask" in token_subset:
            rule = "ASK"
            ask_production = grammar.get_prod_by_ctr_name(rule)
            ask_node = AbstractSyntaxTree(ask_production, None)
            select_or_ask_field = RealizedField(production['select_or_ask'], ask_node)
        else:
            raise ValueError("Wrong SPARQL syntax found! select/ask clause wasn't matched.")

        children.append(select_or_ask_field)

        # WHERE
        where_tokens = token_subset[token_subset.index('where'):token_subset.index('}') + 1]

        # ["where", "{", "SUBJ_1", "dr:locatedinclosest", "?dummy", "." "?dummy", "dr:date_country_end", "?x0", ".", "}"]
        # -> ["SUBJ_1 dr:locatedinclosest?dummy . ?dummy dr:date_country_end ?x0"]
        where_tokens = " ".join(where_tokens[2:-2])

        where_node = build_ast_from_toks(grammar, where_tokens, "WHERE")

        where_field = RealizedField(production['where'], where_node)
        children.append(where_field)

        # ORDER BY
        limit = token_subset.index('limit') if "limit" in token_subset else -1
        order = token_subset.index('order') if "order" in token_subset else -1

        if order != -1:
            order_tokens = token_subset[order:limit] if limit != -1 else token_subset[order:]
        else:
            order_tokens = ""

        order_node = build_ast_from_toks(grammar, order_tokens, "ORDERBY")
        order_field = RealizedField(production['order'], order_node)
        children.append(order_field)

        # LIMIT
        if limit != -1:
            limit_tokens = token_subset[limit:]
        else:
            limit_tokens = ""

        limit_node = build_ast_from_toks(grammar, limit_tokens, "LIMIT")
        limit_field = RealizedField(production['limit'], limit_node)
        children.append(limit_field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["SELECT"]:
        token_subset = token_subset.strip().split(" , ")
        select_nodes = [build_ast_from_toks(grammar, select_val, "SELECT_COL") for select_val in token_subset]
        select_nodes.append(None)  # Reduce

        select_field = RealizedField(production['select_val'], select_nodes)
        children.append(select_field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["SELECT_COL"]:
        column_idx = 0
        token_subset = token_subset.strip().split(" ")

        # 1. ["count", "(", "distinct", "?x0", ")"]
        # 2. ["count", "(", "?x0", ")"]
        # 3. ["distinct", "?x0"]
        # 4. ["?x0"]
        # 5. ["(", "count", "(", "distinct", "?x0", ")", "as", '?value", ")"]
        # 6. ["(", "count", "(", "?x0", ")", "as", '?value", ")"]
        # 7. ["(", "distinct", "?x0", "as", '?value", ")"]
        # 8. ["(", "?x0", "as", '?value", ")"]

        # agg_op
        agg_ops = ["count", "min", "max", "avg", "sum"]
        idx_of_op_token = [x in token_subset for x in agg_ops]

        if any(idx_of_op_token):
            agg_op_token = agg_ops[idx_of_op_token.index(True)]
            rule = agg_op_token.upper()
            agg_production = grammar.get_prod_by_ctr_name(rule)
            agg_node = AbstractSyntaxTree(agg_production, None)

            column_idx += 2
        else:
            agg_node = None
        agg_field = RealizedField(production['agg_op_val'], agg_node)
        children.append(agg_field)

        # as
        if "as" in token_subset:
            rule = "AS"
            as_production = grammar.get_prod_by_ctr_name(rule)
            as_field = RealizedField(as_production["as_val"], token_subset[-2])
            as_node = AbstractSyntaxTree(as_production, as_field)

            column_idx += 1
        else:
            as_node = None
        as_field = RealizedField(production['as'], as_node)

        # col_type
        if "distinct" in token_subset:
            column_idx += 1
            rule = "DISTINCT"
        else:
            rule = "CONST"

        col_type_production = grammar.get_prod_by_ctr_name(rule)
        col_type_field = RealizedField(col_type_production['column'], str(token_subset[column_idx]))
        col_type_node = AbstractSyntaxTree(col_type_production, col_type_field)

        select_col = RealizedField(production['column_type'], col_type_node)
        children.append(select_col)
        children.append(as_field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["WHERE"]:
        token_subset = token_subset.strip().split(" . ")
        where_filter_nodes = []

        for where_val in token_subset:
            if where_val.startswith('filter'):
                where_filter_nodes.append(build_ast_from_toks(grammar, where_val, "FILTER"))
            else:
                where_filter_nodes.append(build_ast_from_toks(grammar, where_val, "TRIPLET"))

        where_filter_nodes.append(None)  # Reduce

        where_field = RealizedField(production['where_val'], where_filter_nodes)
        children.append(where_field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["ACCESSORS", "DATE"]:
        # ["lang", "(", "?OBJ_3", ")"] / ["year", "(", "?OBJ_1", ")"]
        op_token = token_subset[0]
        column_name = token_subset[2]

        op_field_name = {"ACCESSORS":"accessor_val", "DATE":"date_val"}[rule]
        op_production = grammar.get_prod_by_ctr_name(op_token.upper())
        op_node = AbstractSyntaxTree(op_production, None)
        op_field = RealizedField(production[op_field_name], op_node)

        val_field_name = {"ACCESSORS": "access_to", "DATE": "date_of_column"}[rule]
        val_field = RealizedField(production[val_field_name], str(column_name))
        children.extend([op_field, val_field])

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["COLUMN", "STRING", "LCASE"]:
        # ["?OBJ_3"]
        field_name = {"COLUMN":"column", "LCASE":"column", "STRING":"str_val"}.get(rule)
        token_index = {"COLUMN":0, "LCASE":2, "STRING":2}.get(rule)
        field = RealizedField(production[field_name], str(token_subset[token_index]))
        children.append(field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["STRING_FUNC"]:
        op_token = token_subset[0]

        if op_token in ["now", "year", "month", "day", "hours", "minutes", "seconds", "timezone", "tz"]:  # DATE
            rule = "DATE"
        elif op_token in ["lang", "str", "datatype"]:  # ACCESSORS
            rule = "ACCESSORS"
        elif op_token in ["lcase"]:  # LCASE
            rule = "LCASE"
        elif op_token in ["str"]:  # STRING
            rule = "STRING"
        else:  # COLUMN
            rule = "COLUMN"

        string_node = build_ast_from_toks(grammar, token_subset, rule)
        string_func_field = RealizedField(production["str_func_val"], string_node)

        ast_node = AbstractSyntaxTree(production, string_func_field)
        return ast_node

    if rule in ["FILTER"]:
        # ["filter", "(", "strstarts", "(", "lcase", "(", "?OBJ_3", ")", ",", "'ş'", ")", ")"]
        # ["filter", "(", "lang", "(", "?OBJ_3", ")", "=", "'en'", ")"]
        field = token_subset.strip().split(" ")

        comparison = [x in token_subset.strip() for x in [" = ", " != ", " > ", " < ", " in ", " not in "]]
        if any(comparison):  # COMPARISON
            rule = "COMPARISON"
            comp_production = grammar.get_prod_by_ctr_name(rule)
            comp_children = []

            # comparison_val
            rule = [("EQUAL", "="), ("NOTEQUAL", "!="), ("MORE", ">"), ("LESS", "<"), ("IN", "in"), ("NOTIN", "not in")][comparison.index(True)]
            comparison_production = grammar.get_prod_by_ctr_name(rule[0])
            comparison_op_node = AbstractSyntaxTree(comparison_production, None)
            comparison_op_field = RealizedField(comp_production["comparison_val"], comparison_op_node)
            # В fields добавляю ниже для корректного учёта порядка, как в грамматике

            # left_comp / right_comp
            fields = []

            if rule[0] != "NOTIN":
                left_comp_tokens = field[2:field.index(rule[1])]
                right_comp_tokens = field[field.index(rule[1])+1:]
            else:
                left_comp_tokens = token_subset[len("filter ( "):token_subset.find(" not in")].split()
                right_comp_tokens = token_subset[token_subset.find("not in")+len("not in "):-1].split()

            for comp_tokens in [left_comp_tokens, right_comp_tokens]:
                fields.append(build_ast_from_toks(grammar, comp_tokens, "STRING_FUNC"))

            comp_children = [RealizedField(comp_production["left_comp"], fields[0]),
                             comparison_op_field,
                             RealizedField(comp_production["right_comp"], fields[1])]

            comp_ast = AbstractSyntaxTree(comp_production, comp_children)

            filter_field = RealizedField(production['filter'], comp_ast)
            children.append(filter_field)

        else:  # STRINGS
            strings_production = grammar.get_prod_by_ctr_name("STRINGS")
            strings_children = []

            # ////////////
            rule = field[2].upper()  # "strstarts"
            str_production = grammar.get_prod_by_ctr_name(rule)

            token_str_func = field[4:field.index(")") if not field[4].startswith("?") else 5]
            filter_string_node = build_ast_from_toks(grammar, token_str_func, "STRING_FUNC")
            filter_string_field = RealizedField(str_production["filter_string"], filter_string_node)
            strings_children.append(filter_string_field)

            sub_search = str(field[field.index(",")+1])
            for x in field[field.index(",")+2:]:
                if x == ")":
                    break
                sub_search = " ".join([sub_search, str(x)])

            sub_search_field = RealizedField(str_production["sub_search"], sub_search)

            strings_children.append(sub_search_field)
            str_node = AbstractSyntaxTree(str_production, strings_children)
            # ////////////

            strings_field = [RealizedField(strings_production["strings_val"], str_node)]
            strings_node = AbstractSyntaxTree(strings_production, strings_field)

            filter_field = RealizedField(production['filter'], strings_node)
            children.append(filter_field)

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["TRIPLET"]:
        # ["SUBJ_1", "dr:locatedinclosest", "?dummy"]
        field = token_subset.strip().split(" ")
        subject, predicate, object_ = field

        children.extend([RealizedField(production["subject_val"], str(subject)),
                         RealizedField(production["predicate_val"], str(predicate)),
                         RealizedField(production["object_val"], str(object_))
                         ])

        ast_node = AbstractSyntaxTree(production, children)
        return ast_node

    if rule in ["ORDERBY"]:
        column_idx = 2

        # 1. ['order', 'by', 'desc', '(', '?x0', ')']
        # 2. ['order', 'by', 'asc', '(', '?x0', ')']
        # 3. ['order', 'by', '?x0']
        # 4. []

        if token_subset:
            if ("desc" in token_subset) or ("asc" in token_subset):
                rule = token_subset[2].upper()
                desc_production = grammar.get_prod_by_ctr_name(rule)
                order_flag = AbstractSyntaxTree(desc_production, None)
                column_idx = 4
            else:
                order_flag = None

            desc_field = RealizedField(production['flag_val'], order_flag)
            children.append(desc_field)

            token = token_subset[column_idx]
            ord_column = RealizedField(production["ord_column"], token)
            children.append(ord_column)

            ast_node = AbstractSyntaxTree(production, children)
            return ast_node

        return None

    if rule in ["LIMIT"]:

        # 1. ['limit', 'x']
        # 2. []

        if token_subset:
            limit_val = token_subset[-1]

            limit_field = RealizedField(production['limit_val'], limit_val)
            children.append(limit_field)

            ast_node = AbstractSyntaxTree(production, children)
            return ast_node

        return None

def build_sparql_expr_from_ast(sparql_ast):

    def _string_func_processor(string_ast):

        if string_ast.production.constructor.name == "ACCESSORS":
            accessor_val = string_ast.fields[0].value.production.constructor.name
            access_to = string_ast.fields[1].value

            return [accessor_val.lower(), "(", access_to, ")"]

        elif string_ast.production.constructor.name == "DATE":
            date_val = string_ast.fields[0].value.production.constructor.name
            date_of_column = string_ast.fields[1].value

            return [date_val.lower(), "(", date_of_column, ")"]

        elif string_ast.production.constructor.name == "COLUMN":
            column_id = string_ast.fields[0].value

            return [column_id]

        elif string_ast.production.constructor.name == "LCASE":
            column_id = string_ast.fields[0].value

            return ["lcase", "(", column_id, ")"]

        else:
            raise ValueError("Wrong AST structure.")

    tokens = deque()
    select, where, order, limit = [x.value for x in sparql_ast.fields]

    # ASK
    if select.production.constructor.name == "ASK":
        tokens.append("ask")

    else:  # SELECT
        tokens.append("select")
        for select_col in select.fields[0].value:
            loop_tokens = deque()

            if select_col is None:
                continue  # Reduce skipping

            aggrigation_field, col_type_field, as_ = [x.value for x in select_col.fields]

            # column_type field
            col_type = col_type_field.production.constructor.name

            if col_type == "DISTINCT":
                loop_tokens.append(col_type.lower())

            # col_type -> column
            column_id = col_type_field.fields[0].value
            loop_tokens.append("<unk>" if column_id is None else column_id)

            # agg_op_val
            if aggrigation_field is not None:
                agg_op = aggrigation_field.production.constructor.name
                loop_tokens.extendleft(["(", agg_op.lower()])
                loop_tokens.append(")")

            # as
            if as_ is not None:
                as_col_name = as_.fields[0].value
                loop_tokens.appendleft("(")
                loop_tokens.extend(["as", as_col_name, ")"])

            loop_tokens.append(",")
            tokens += loop_tokens
        else:
            tokens.pop()  # removes last ","

    # WHERE
    tokens.append("where {")

    for where_val in where.fields[0].value:
        if where_val is None:
            continue  # reduce skipping
        if where_val.production.constructor.name == "TRIPLET":
            tokens.extend([x.value for x in where_val.fields])

        elif where_val.production.constructor.name == "FILTER":
            tokens.extend(["filter", "("])
            filter = where_val.fields[0].value

            if filter.production.constructor.name == "COMPARISON":
                comparison_val = filter.fields[1].value.production.constructor.name

                left_comp, right_comp = filter.fields[0].value, filter.fields[2].value

                comparison_val_token = {"EQUAL": "=", "NOTEQUAL": "!=", "MORE": ">",
                                        "LESS": "<", "IN": "in", "NOTIN": "not in"}[comparison_val]

                left_comp = _string_func_processor(left_comp.fields[0].value)
                right_comp = _string_func_processor(right_comp.fields[0].value)

                tokens.extend(left_comp+[comparison_val_token]+right_comp)

            elif filter.production.constructor.name == "STRINGS":
                strings_val = filter.fields[0].value

                rule = strings_val.production.constructor.name
                filter_string = _string_func_processor(strings_val.fields[0].value.fields[0].value)
                sub_search = strings_val.fields[1].value
                tokens.extend([rule.lower(), "("] + filter_string + [",", sub_search, ")"])
            tokens.append(")")

        tokens.append(".")


    tokens.append('}')

    # ORDERBY
    if order is not None:
        order_fields = order.fields
        tokens.extend(["order", "by"])

        order_flag_token = order_fields[0].value.production.constructor.name.lower()
        tokens.append(order_flag_token)

        tokens.extend(["(", order_fields[1].value, ")"])

    # LIMIT
    if limit is not None:
        tokens.extend(["limit", limit.fields[0].value])

    return tokens


def sparql_expr_to_ast(grammar, sparql_tokens):
    sparql_ast = build_ast_from_toks(grammar, sparql_tokens, rule="QUERY")
    return sparql_ast


def ast_to_sparql_expr(sparql_ast):
    tokens = build_sparql_expr_from_ast(sparql_ast)
    return " ".join(tokens)

# neglet created time

def is_equal_ast(this_ast, other_ast):
    if not isinstance(other_ast, this_ast.__class__):
        return False

    if isinstance(this_ast, AbstractSyntaxTree):
        if this_ast.production != other_ast.production:
            return False

        if len(this_ast.fields) != len(other_ast.fields):
            return False
        for this_f, other_f in zip(this_ast.fields, other_ast.fields):
            if not is_equal_ast(this_f.value, other_f.value):
                return False
        return True
    else:
        return this_ast == other_ast


class SparqlTransitionSystem(TransitionSystem):
    def compare_ast(self, hyp_ast, ref_ast):
        return is_equal_ast(hyp_ast, ref_ast)

    def ast_to_surface_code(self, sparql_ast):
        return ast_to_sparql_expr(sparql_ast)

    def surface_code_to_ast(self, code):
        return sparql_expr_to_ast(self.grammar, code)

    def tokenize_code(self, code, mode):
        raise NotImplementedError
