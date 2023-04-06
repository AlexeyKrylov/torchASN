import torch
import torch.nn as nn
import torch.nn.functional as F
from components.dataset import Batch
from grammar.transition_system import ApplyRuleAction, GenTokenAction, ActionTree, ReduceAction
from grammar.hypothesis import Hypothesis
import numpy as np
import os
from common.config import update_args
from transformers import AutoModel


class ReduceModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.type = "Reduce"
        self.w = nn.Linear(args.enc_hid_size + args.field_emb_size, 2)

    def forward(self, x):
        return self.w(x)

    # x b * h
    def score(self, x, contexts):
        x = torch.cat([x, contexts], dim=1)

        return F.log_softmax(self.w(x), 1)


class CompositeTypeModule(nn.Module):
    def __init__(self, args, type_, productions):
        super().__init__()
        self.type = type_
        self.productions = productions
        self.w = nn.Linear(args.enc_hid_size + args.field_emb_size, len(productions))

    def forward(self, x):
        return self.w(x)

    # x b * h
    def score(self, x, contexts):
        x = torch.cat([x, contexts], dim=1)

        return F.log_softmax(self.w(x), 1)


class ConstructorTypeModule(nn.Module):
    def __init__(self,  args, production):
        super().__init__()
        self.production = production
        self.n_field = len(production.constructor.fields)+1
        self.field_embeddings = nn.Embedding(self.n_field, args.field_emb_size)
        self.w = nn.Linear(args.enc_hid_size + args.field_emb_size, args.enc_hid_size)
        self.dropout = nn.Dropout(args.dropout)
    
    def update(self, v_lstm, v_state, contexts):
        # v_state, h_n, c_n where 1 * b * h
        # input: seq_len, batch, input_size
        # h_0 of shape (1, batch, hidden_size)
        # v_lstm(, v_state)
        inputs = self.field_embeddings.weight
        inputs = self.dropout(inputs)
        contexts = contexts.expand([self.n_field, -1])
        inputs = self.w(torch.cat([inputs, contexts], dim=1)).unsqueeze(0)
        v_state = (v_state[0].expand(self.n_field, -1).unsqueeze(0).contiguous(), v_state[1].expand(self.n_field, -1).unsqueeze(0).contiguous())
        _, outputs = v_lstm(inputs, v_state)

        hidden_states = outputs[0].unbind(1)
        cell_states = outputs[1].unbind(1)

        return list(zip(hidden_states, cell_states))


class PrimitiveTypeModule(nn.Module):
    def __init__(self, args, type_, vocab):
        super().__init__()
        self.type = type_
        self.vocab = vocab
        self.w = nn.Linear(args.enc_hid_size + args.field_emb_size, len(vocab))

    def forward(self, x):
        return self.w(x)

    # x b * h
    def score(self, x, contexts):
        x = torch.cat([x, contexts], dim=1)

        return F.log_softmax(self.w(x), 1)


class ASNParser(nn.Module):
    def __init__(self, args, transition_system, vocab):
        super().__init__()

        # encoder
        self.args = args

        self.transition_system = transition_system
        self.vocab = vocab
        grammar = transition_system.grammar
        self.grammar = grammar

        self.encoder = Encoder(args.src_emb_size,
                               args.dropout,
                               train=args.train,
                               bert_name=args.bert_name)

        comp_type_modules = {}
        for dsl_type in grammar.composite_types:
            comp_type_modules.update({dsl_type.name:
                                      CompositeTypeModule(args, dsl_type, grammar.get_prods_by_type(dsl_type))})
        self.comp_type_dict = nn.ModuleDict(comp_type_modules)

        cnstr_type_modules = {}
        for prod in grammar.productions:
            cnstr_type_modules.update({prod.constructor.name:
                                       ConstructorTypeModule(args, prod)})
        self.const_type_dict = nn.ModuleDict(cnstr_type_modules)

        prim_type_modules = {}
        for prim_type in grammar.primitive_types:
            prim_type_modules.update({prim_type.name:
                                      PrimitiveTypeModule(args, prim_type, vocab.primitive_vocabs[prim_type])})

        self.prim_type_dict = nn.ModuleDict(prim_type_modules)

        self.reduce_module = ReduceModule(args)

        self.v_lstm = nn.LSTM(args.enc_hid_size, args.enc_hid_size)
        self.attn = LuongAttention(args.enc_hid_size, 2 * args.enc_hid_size)
        self.dropout = nn.Dropout(args.dropout)

    def score(self, examples):
         batch = Batch(examples, self.grammar, self.vocab, cuda=self.args.cuda, bert_name=self.args.bert_name)

         return torch.stack(self._score(batch))

    def _score(self, batch):
        # batch_size, sent_len, enc_dim
         encoder_outputs = self.encoder(batch.sents)  # sent_embedding

         encoder_outputs = encoder_outputs.transpose(0, 1)
         init_state = encoder_outputs[-1, :, :]

         # TODO: Подумать как это распараллелить
         return [self._score_node(self.grammar.root_type, (init_state[ex, :].unsqueeze(0), init_state[ex, :].unsqueeze(0)), batch[ex].tgt_actions, encoder_outputs[:, ex, :].unsqueeze(1), batch.sent_masks[ex], "single") for ex in range(len(batch))]

    def _score_node(self, node_type, v_state, action_node, context_vecs, context_masks, cardinality):
        v_output = self.dropout(v_state[0])

        contexts = self.attn(v_output.unsqueeze(0), context_vecs).squeeze(0)
        score = 0

        if cardinality == "optional":
            scores = self.reduce_module.score(v_state[0], contexts)
            scores = -1 * scores.view([-1])

            if action_node.action.choice_index == -2:
                return scores[1]

            score += scores[0]

        if cardinality == "multiple":

            self.recursion_v_state = v_state

            for field in action_node:
                score += self._score_node(node_type, self.recursion_v_state, field, context_vecs, context_masks, "optional")

            return score

        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]
            scores = module.score(v_output, contexts)
            score += -1 * scores.view([-1])[action_node.action.choice_index]

            return score

        else:
            cnstr = action_node.action.choice.constructor

            comp_module = self.comp_type_dict[node_type.name]
            scores = comp_module.score(v_output, contexts)
            score += -1 * scores.view([-1])[action_node.action.choice_index]

            cnstr_module = self.const_type_dict[cnstr.name]

            cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)

            for next_field, next_state, next_action in zip(cnstr.fields, cnstr_results[:-1], action_node.fields):
                self.recursion_v_state = next_state
                score += self._score_node(next_field.type, next_state, next_action, context_vecs, context_masks, next_field.cardinality)
            self.recursion_v_state = cnstr_results[-1]
            return score

    def naive_parse(self, batch):

        encoder_outputs = self.encoder(batch.sents)  # sent_embedding

        encoder_outputs = encoder_outputs.transpose(0, 1)
        init_state = encoder_outputs[-1, :, :]

        action_tree_list = [self._naive_parse(self.grammar.root_type,
                                              (init_state[ex, :].unsqueeze(0), init_state[ex, :].unsqueeze(0)),
                                              encoder_outputs[:, ex, :].unsqueeze(1),
                                              batch.sent_masks[ex], 1, 'single') for ex in range(len(batch))]

        return [self.transition_system.build_ast_from_actions(action_tree) for action_tree in action_tree_list]

    def _naive_parse(self, node_type, v_state, context_vecs, context_masks, depth, cardinality="single"):

        contexts = self.attn(v_state[0].unsqueeze(0), context_vecs).squeeze(0)

        if cardinality == "optional":

            scores = self.reduce_module.score(v_state[0], contexts).cpu().numpy().flatten()
            is_reduce = np.argmax(scores)

            if is_reduce or (depth > self.args.max_depth):
                return ActionTree(ReduceAction())

        if cardinality == "multiple":
            action_fields = []
            self.recursion_v_state = v_state

            for _ in range(self.args.max_depth):
                depth += 1

                action_tree = self._naive_parse(node_type, self.recursion_v_state, context_vecs, context_masks, depth, "optional")
                action_fields.append(action_tree)

                if isinstance(action_tree.action, ReduceAction):
                    break

            return action_fields

        else:

            # Primitive
            if node_type.is_primitive_type():
                module = self.prim_type_dict[node_type.name]
                scores = module.score(v_state[0], contexts).cpu().numpy().flatten()

                choice_idx = np.argmax(scores)

                a = module.vocab.get_word(choice_idx)

                return ActionTree(GenTokenAction(node_type, a))

            else:  # Composite

                comp_module = self.comp_type_dict[node_type.name]
                scores = comp_module.score(v_state[0], contexts).cpu().numpy().flatten()
                choice_idx = np.argmax(scores)
                production = comp_module.productions[choice_idx]

                action = ApplyRuleAction(node_type, production)

                cnstr = production.constructor

                cnstr_module = self.const_type_dict[cnstr.name]

                cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)

                action_fields = []
                for next_field, next_state in zip(cnstr.fields, cnstr_results[:-1]):
                    self.recursion_v_state = next_state
                    action_fields.append(self._naive_parse(next_field.type, next_state,
                                                           context_vecs, context_masks, depth + 1,
                                                           next_field.cardinality))
                self.recursion_v_state = cnstr_results[-1]
                return ActionTree(action, action_fields)

    def parse(self, batch):  # Is not using
        context_vecs, encoder_outputs = self.encode(batch)
        init_state = encoder_outputs

        completed_hyps = []
        cur_beam = [Hypothesis.init_hypothesis(self.grammar.root_type, init_state)]
        
        for ts in range(self.args.max_decode_step):
            hyp_pools = []
            for hyp in cur_beam:
                continuations = self.continuations_of_hyp(hyp, context_vecs, batch.sent_masks)
                hyp_pools.extend(continuations)
            
            hyp_pools.sort(key=lambda x: x.score, reverse=True)
            # next_beam = next_beam[:self.args.beam_size]
            
            num_slots = self.args.beam_size - len(completed_hyps)

            cur_beam = []
            for hyp_i, hyp in enumerate(hyp_pools[:num_slots]):
                if hyp.is_complete():
                    completed_hyps.append(hyp)
                else:
                    cur_beam.append(hyp)
            
            if not cur_beam:
                break
        
        completed_hyps.sort(key=lambda x: x.score, reverse=True)
        return completed_hyps

    def continuations_of_hyp(self, hyp, context_vecs, context_masks):
        
        pending_node, v_state = hyp.get_pending_node()
        
        contexts = self.attn(v_state[0].unsqueeze(0), context_vecs).squeeze(0)

        node_type = pending_node.action.type

        if node_type.is_primitive_type():
            module = self.prim_type_dict[node_type.name]

            scores = module.score(v_state[0], contexts).cpu().numpy().flatten()

            continuous = []
            for choice_idx, score in enumerate(scores):
                continuous.append(hyp.copy_and_apply_action(GenTokenAction(node_type, module.vocab.get_word(choice_idx)), score))
            return continuous

        comp_module = self.comp_type_dict[node_type.name]
        scores = comp_module.score(v_state[0], contexts).cpu().numpy().flatten()

        continuous = []
        for choice_idx, score in enumerate(scores):
            production = comp_module.productions[choice_idx]
            action = ApplyRuleAction(node_type, production)
            cnstr = production.constructor
            cnstr_module = self.const_type_dict[cnstr.name]
            cnstr_results = cnstr_module.update(self.v_lstm, v_state, contexts)
            continuous.append(hyp.copy_and_apply_action(action, score, cnstr_results))

        return continuous

    def save(self, filename):
        dir_name = os.path.dirname(filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        params = {
            'args': self.args,
            'transition_system': self.transition_system,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, filename)

    @classmethod
    def load(cls, model_path, ex_args=None, cuda=False):
        params = torch.load(model_path)
        vocab = params['vocab']
        transition_system = params['transition_system']
        saved_args = params['args']
        # update saved args
        saved_state = params['state_dict']
        saved_args.cuda = cuda
        if ex_args:
            update_args(saved_args, ex_args)
        parser = cls(saved_args, transition_system, vocab)
        parser.load_state_dict(saved_state)

        if cuda:
            parser = parser.cuda()

        parser.eval()

        return parser

    def forward(self, input_):
        return [self.naive_parse(ex) for ex in input_]


class Encoder(nn.Module):
    def __init__(self, embedding_dim, embedding_dropout_rate, train=False, bert_name="cointegrated/rubert-tiny"):
        super(Encoder, self).__init__()
        self.model = AutoModel.from_pretrained(bert_name)

        for param in self.model.parameters():
            param.requires_grad = train

        self.linear = nn.Linear(self.model.encoder.layer[0].output.dense.out_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(embedding_dropout_rate)

    def forward(self, input):
        model_output = self.model(**input)
        embeddings = model_output.last_hidden_state
        embeddings = self.linear(embeddings)
        return embeddings


class LuongAttention(nn.Module):

    def __init__(self, hidden_size, context_size=None):
        super(LuongAttention, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = hidden_size # if context_size is None else context_size CHANGED
        self.attn = torch.nn.Linear(self.context_size, self.hidden_size)

    # input query: batch * q * hidden, contexts: c * batch * hidden
    # output: batch * len * q * c
    def forward(self, query, context, inf_mask=None, requires_weight=False):
        # Calculate the attention weights (energies) based on the given method
        query = query.transpose(0, 1)
        context = context.transpose(0, 1)
        e = self.attn(context)
        # e: B * Q * C

        e = torch.matmul(query, e.transpose(1, 2))
        if inf_mask is not None:
            e = e + inf_mask.unsqueeze(1)

        # dim w: B * Q * C, context: B * C * H, wanted B * Q * H
        w = F.softmax(e, dim=2)
        c = torch.matmul(w, context)
        # Return the softmax normalized probability scores (with added dimension
        if requires_weight:
            return c.transpose(0, 1), w

        return c.transpose(0, 1)
