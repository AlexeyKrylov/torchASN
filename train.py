from common.config import *
from components.dataset import *
from common.utils import calculate_batch_metrics
import common.lr_scheduler


from grammar.grammar import Grammar
import common.utils as utils
from grammar.sparql.sparql_transition_system import SparqlTransitionSystem
from models.ASN import ASNParser
from models import nn_utils
from datetime import datetime
from torch import optim
from tqdm import tqdm
import time
import os
from datasets.sparql.make_dataset import make_dataset

def get_lr(optimizer):
    # TODO: Remove default lr_schaduler has .get_lr() method
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return

def train(args):
    # make_dataset(args.language, args.project_path)
    path_save_to = args.save_to + "ASN_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.mkdir(path_save_to)
    os.mkdir(path_save_to + '/models')
    os.mkdir(path_save_to + '/logs')

    if args.make_log:
        logger = utils.TXTLogger(work_dir=path_save_to + "/logs")

    train_set = Dataset.from_bin_file(args.train_file)
    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else:
        dev_set = Dataset(examples=[])
    
    vocab = pickle.load(open(args.vocab, 'rb'))
    grammar = Grammar.from_text(open(args.asdl_file).read())

    transition_system = SparqlTransitionSystem(grammar)

    parser = ASNParser(args, transition_system, vocab)

    if args.cuda:
        parser = parser.cuda()

    # Костыль в студию
    encoder_params_size = len([*parser.src_embedding.model.parameters()])
    nn_utils.glorot_init(parser.parameters(), encoder_params_size)

    optimizer = optim.AdamW([{"params": iter([*parser.parameters()][encoder_params_size:])},
                             {"params": parser.src_embedding.model.parameters(), "lr": args.bert_finetune_rate}], lr=args.lr)
    # optimizer = optim.Adam(parser.parameters(), lr=args.lr)

    lr_scheduler = common.lr_scheduler.InverseSquareRootScheduler(optimizer=optimizer,
                                                           warmup_init_lrs=[
                                                               args.bert_warmup_init_finetuning_learning_rate,
                                                               args.warm_up_init_learning_rate],
                                                           num_warmup_steps=[
                                                               args.num_warmup_steps,
                                                               args.num_warmup_steps],
                                                           num_steps=int(
                                                               len(train_set) // args.batch_size
                                                                              * args.max_epoch))
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sch_step_size, gamma=args.gamma)
    best_acc = 0
    
    train_begin = time.time()
    for epoch in range(1, args.max_epoch + 1):
        train_iter = 0
        val_loss = 0.
        train_loss = 0.

        parser.train()

        epoch_begin = time.time()
        for batch_example in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = parser.score(batch_example)
            train_loss += torch.sum(loss).data.item()
            loss = torch.mean(loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parser.parameters(), args.clip_grad)

            optimizer.step()
            lr_scheduler.step()
            train_iter += 1

        print('[epoch {}] train loss {:.3f}, epoch time {:.0f}, total time {:.0f}, lr {:.5f}'.format(epoch, train_loss / len(train_set), time.time() - epoch_begin, time.time() - train_begin, get_lr(optimizer)))

        target_data: list = []
        eval_result: list = []
        input_data: list = []

        val_exm_epoch_acc = 0
        val_gm_epoch_acc = 0
        if epoch > args.run_val_after:

            for dev_set_ in tqdm(dev_set.batch_iter(batch_size=args.batch_size, shuffle=False)):
                parser.eval()

                with torch.no_grad():
                    batch = Batch(dev_set_, parser.grammar, parser.vocab, train=False, cuda=parser.args.cuda)
                    parse_results = list(zip(parser.naive_parse(batch), [dev_set_[ex].tgt_ast for ex in range(len(batch))]))

                    loss = parser.score(dev_set_)
                    val_loss += torch.sum(loss).data.item()

                    input_data.extend([x.src_toks for x in dev_set_])
                    target_data.extend([transition_system.ast_to_surface_code(x[1]) for x in parse_results])
                    eval_result.extend([transition_system.ast_to_surface_code(x[0]) for x in parse_results])

                    val_metrics = calculate_batch_metrics(eval_result, target_data)
                    val_exm_epoch_acc += val_metrics['exact_match']
                    val_gm_epoch_acc += val_metrics['graph_match']

            val_exm_epoch_acc /= len(dev_set)
            val_gm_epoch_acc /= len(dev_set)
            val_loss /= len(dev_set)
            train_loss /= len(train_set)

            if args.make_log:
                logger.log({"epoch": epoch,
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "val_exact_match": val_exm_epoch_acc,
                            "val_graph_match": val_gm_epoch_acc,
                            "learning_rate": lr_scheduler.get_lr()})

                logger.log('********** Translation example **********')
                for input_question, true_sparql, pred_sparql in zip(input_data[:5],
                                                                    target_data[:5],
                                                                    eval_result[:5]):
                    logger.log(f"NL: {input_question}")
                    logger.log(f"AQ: {true_sparql}")
                    logger.log(f"PQ: {pred_sparql}")
                    logger.log(" ")
                logger.log('******************************')

            if val_exm_epoch_acc >= best_acc:
                best_acc = val_exm_epoch_acc
                parser.save(path_save_to+"/models/ASN_model_file.pt")

            print(f"BEST ACCURACY = {best_acc}")

if __name__ == '__main__':
    args = parse_args('train')
    train(args)
