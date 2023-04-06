# coding=utf-8
import argparse


def update_args(args, ex_args):
    for k, v in ex_args.__dict__.items():
        setattr(args, k, v)


def _add_test_args(parser):
    parser.add_argument('--test_file', default='/data/sparql/test.bin', type=str, help='path to the test set file')
    parser.add_argument('--model_file', default='/checkpoints/ASN_12_03_2023_18_19_58/models/ASN_model_file.pt', type=str, help='path to the model file')
    parser.add_argument('--beam_size', default=100, type=int, help='decoder beam size')
    parser.add_argument('--max_decode_step', default=100, type=int, help='maximum decode step')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--asdl_file', default='/data/sparql/sparql_asdl.txt', type=str, help='Path to ASDL grammar specification')
    parser.add_argument('--batch_size', default=10, type=int, help='batch size')
    parser.add_argument('--dev_file', default='/data/sparql/dev.bin',  type=str, help='path to the dev source file')
    parser.add_argument('--language', type=str, default='russian', help='language: {english/russian}')
    parser.add_argument('--project_path', type=str, default='C:/Users/krilo/PycharmProjects/torchASN', help='Project path')
    parser.add_argument('--bert_name', type=str, default='DeepPavlov/rubert-base-cased', help='Name of the bert')




def _add_train_args(parser):
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--bert_name', type=str, default='cointegrated/rubert-tiny', help='Name of the bert: DeepPavlov/rubert-base-cased # cointegrated/rubert-tiny')
    parser.add_argument('--language', type=str, default='russian', help='language: {english/russian}')
    parser.add_argument('--project_path', type=str, default='./', help='Project path')
    parser.add_argument('--make_log', action='store_true', default=True, help='make tensorboard log')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use gpu')
    parser.add_argument('--asdl_file', default='data/sparql/sparql_asdl.txt', type=str, help='Path to ASDL grammar specification')
    parser.add_argument('--vocab', default='data/sparql/vocab.bin', type=str, help='Path of the serialized vocabulary')
    parser.add_argument('--save_to', type=str, default='checkpoints/', help='save the model to')
    parser.add_argument('--train_file', default='data/sparql/train.bin', type=str, help='path to the training target file')
    parser.add_argument('--dev_file', default='data/sparql/dev.bin',  type=str, help='path to the dev source file')
    parser.add_argument('--enc_hid_size', default=256,  type=int, help='encoder hidden size')
    parser.add_argument('--num_warmup_steps', default=50, type=int, help='num of warmups to target lr')
    parser.add_argument('--warm_up_init_learning_rate', default=0.005, type=int, help='initial warmup rate for parser')
    parser.add_argument('--bert_finetune_rate', default=0.0005, type=int, help='finetune step')
    parser.add_argument('--bert_warmup_init_finetuning_learning_rate', default=0.00003, type=int, help='initial warmup rate for BERT')
    parser.add_argument('--src_emb_size', default=256,  type=int, help='sentence embedding size')
    parser.add_argument('--field_emb_size', default=256, type=int, help='field embedding size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--batch_size', default=1024,  type=int, help='batch size')
    parser.add_argument('--max_epoch', default=200, type=int, help='max epoch')
    parser.add_argument('--max_depth', default=10, type=int, help='maximum depth of action tree')
    parser.add_argument('--clip_grad', type=float, default=10.0, help='clip grad to')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='lr scheduler gamma')
    parser.add_argument('--run_val_after', type=int, default=0, help='run validation after')
    parser.add_argument('--max_decode_step', default=100, type=int, help='maximum decode step')


def parse_args(mode):
    parser = argparse.ArgumentParser()

    if mode == 'train':
        _add_train_args(parser)
    elif mode == 'test':
        _add_test_args(parser)
    else:
        raise RuntimeError('unknown mode')
    
    args = parser.parse_args()
    print(args)
    return args
