from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import numpy as np
import torch
import math
import pickle
import shutil
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from argparse import ArgumentParser
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from Model.common.InputExample import InputExample
from Model.cal_f1 import eval_rank1
from Model.common.DataProcessorForAnswer import DataProcessor
from Model.common.BertEncoderX import BertForSequenceWithAnswerType, BertForSequence


def test(best_model_dir_name, fout_res, args):
    print('测试选用的模型是', best_model_dir_name)
    fout_res.write('测试选用的模型是:' + best_model_dir_name + '\n')
    processor = DataProcessor(args)
    device = torch.device("cuda", 0)
    merge_mode = ['pairwise']
    tokenizer = BertTokenizer.from_pretrained(best_model_dir_name, do_lower_case=args.do_lower_case)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
    model = BertForSequence.from_pretrained(best_model_dir_name,cache_dir=cache_dir,num_labels=2)
    model.to(device)
    # 构建验证集数据  
    eval_examples = processor.get_test_examples(args.data_dir)
    # import pdb; pdb.set_trace()   
    eval_data = processor.convert_examples_to_features(eval_examples, tokenizer)
    eval_data = processor.build_data_for_model(eval_data, tokenizer, device)
    # import pdb; pdb.set_trace()
    file_name1 = args.output_dir + 'prediction_test_all'
    f_valid = open(file_name1, 'w', encoding='utf-8')
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    P_dev = 0
    n_batch_correct_test = 0.0
    len_test_data = 0.0
    TruePos = 0
    TAll = 0
    FalseNeg = 0
    FAll = 0
    for input_ids, input_mask, segment_ids, label_ids, rels_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device).view(-1, args.max_seq_length)
        input_mask = input_mask.to(device).view(-1, args.max_seq_length)
        segment_ids = segment_ids.to(device).view(-1, args.max_seq_length)
        label_ids = label_ids.to(device).view(-1)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)        
        logitsSoftmax = torch.softmax(logits, 1)
        argmaxId = torch.argmax(logitsSoftmax, 1)
        # import pdb; pdb.set_trace()
        for i, item in enumerate(label_ids):
            if(label_ids[i] == argmaxId[i]):
                if(item == 1):
                    TruePos += 1
                else:
                    FalseNeg += 1
            if(item == 1):
                TAll += 1
            else:
                FAll += 1
        n_batch_correct_test += torch.sum(torch.eq(argmaxId.long(),label_ids.long()))
        len_test_data += logits.size(0)
        for item in logitsSoftmax:
            f_valid.write(str(float(item[1])) + '\n')
        f_valid.flush()
    F_dev = float(n_batch_correct_test.float() / len_test_data)
    print((str(F_dev) + '正例预测正确：' + str(1.0 * TruePos / TAll) + '负例预测正确：' + str(1.0 * FalseNeg / FAll) + '\n'))
    fout_res.write(str(F_dev) + '正例预测正确：' + str(1.0 * TruePos / TAll) + '负例预测正确：' + str(1.0 * FalseNeg / FAll) + '\n')
    rank1Percentage = eval_rank1(file_name1, args.data_dir + args.t_file_name, 't')
    fout_res.write('rank1Percentage:' + str(rank1Percentage) + '\n')
    fout_res.flush()


def dev(best_model_dir_name, fout_res, args):
    print('测试选用的模型是', best_model_dir_name)
    fout_res.write('测试选用的模型是:' + best_model_dir_name + '\n')
    processor = DataProcessor(args)
    device = torch.device("cuda", 0)
    merge_mode = ['pairwise']
    tokenizer = BertTokenizer.from_pretrained(best_model_dir_name, do_lower_case=args.do_lower_case)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE))
    model = BertForSequence.from_pretrained(best_model_dir_name,cache_dir=cache_dir,num_labels=2)
    model.to(device)
    # 构建验证集数据  
    eval_examples = processor.get_dev_examples(args.data_dir)
    # import pdb; pdb.set_trace()   
    eval_data = processor.convert_examples_to_features(eval_examples, tokenizer)
    eval_data = processor.build_data_for_model(eval_data, tokenizer, device)
    # import pdb; pdb.set_trace()
    file_name1 = args.output_dir + 'prediction_dev_all'
    f_valid = open(file_name1, 'w', encoding='utf-8')
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    P_dev = 0
    n_batch_correct_test = 0.0
    len_test_data = 0.0
    TruePos = 0
    TAll = 0
    FalseNeg = 0
    FAll = 0
    for input_ids, input_mask, segment_ids, label_ids, rels_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device).view(-1, args.max_seq_length)
        input_mask = input_mask.to(device).view(-1, args.max_seq_length)
        segment_ids = segment_ids.to(device).view(-1, args.max_seq_length)
        label_ids = label_ids.to(device).view(-1)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)        
        logitsSoftmax = torch.softmax(logits, 1)
        argmaxId = torch.argmax(logitsSoftmax, 1)
        # import pdb; pdb.set_trace()
        for i, item in enumerate(label_ids):
            if(label_ids[i] == argmaxId[i]):
                if(item == 1):
                    TruePos += 1
                else:
                    FalseNeg += 1
            if(item == 1):
                TAll += 1
            else:
                FAll += 1
        n_batch_correct_test += torch.sum(torch.eq(argmaxId.long(),label_ids.long()))
        len_test_data += logits.size(0)
        for item in logitsSoftmax:
            f_valid.write(str(float(item[1])) + '\n')
        f_valid.flush()
    F_dev = float(n_batch_correct_test.float() / len_test_data)
    print((str(F_dev) + '正例预测正确：' + str(1.0 * TruePos / TAll) + '负例预测正确：' + str(1.0 * FalseNeg / FAll) + '\n'))
    fout_res.write(str(F_dev) + '正例预测正确：' + str(1.0 * TruePos / TAll) + '负例预测正确：' + str(1.0 * FalseNeg / FAll) + '\n')
    rank1Percentage = eval_rank1(file_name1, args.data_dir + args.v_file_name, 'v')
    fout_res.write('rank1Percentage:' + str(rank1Percentage) + '\n')
    fout_res.flush()

if __name__ == "__main__":
    seed = 42
    steps = 50
    # for N in [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140]:
    modelDic = {19: '/home/jiayonghui/github/sum/RankingQueryGraphs/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_19_42_50/0.964625902596405_0.9826192855834961_0',
                9: '/home/jiayonghui/github/sum/RankingQueryGraphs/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_9_42_50/0.9772776156091566_0.9763732552528381_1',
                4: '/home/jiayonghui/github/sum/RankingQueryGraphs/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_4_42_50/0.9938853894607467_0.9584431052207947_4'}
    for N in [4, 9, 19]:
    # for N in [4, 9, 19]:
        logger = logging.getLogger(__name__)
        print(seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = '2'
        parser = ArgumentParser(description = 'For KBQA')
        parser.add_argument("--data_dir",default=BASE_DIR + '/runnings/train_data/webq/',type=str)
        # parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
        # parser.add_argument("--bert_vocab", default='bert-base-uncased', type=str)
        parser.add_argument("--bert_model", default= '/home/jiayonghui/github/bert_rank_data/bert_base_uncased', type=str)
        parser.add_argument("--bert_vocab", default='/home/jiayonghui/github/bert_rank_data/bert_base_uncased', type=str)
        parser.add_argument("--task_name",default='mrpc',type=str,help="The name of the task to train.")
        parser.add_argument("--output_dir",default=BASE_DIR + '/runnings/model/webq/bert_group1_webq_pointwise_only_que_answertype_neg_' + str(N) + '_' + str(seed) + '_' + str(steps) + '/',type=str)
        parser.add_argument("--input_model_dir", default='0.9675389502344577_0.4803025192052977_3', type=str)
        parser.add_argument("--T_file_name",default='webq_only_answer_info_train_1_' + str(N) + '.txt',type=str)
        # parser.add_argument("--v_file_name",default='pairwise_with_freebase_id_dev_all_cut.txt',type=str)
        parser.add_argument("--v_file_name",default='webq_only_answer_info_dev_all.txt',type=str)
        # parser.add_argument("--v_file_name",default='webq_rank1_f01_gradual_label_position_1_' + str(N) + '_type_entity_time_ordinal_mainpath_is_train.txt',type=str)
        parser.add_argument("--t_file_name",default='webq_only_answer_info_test_all.txt',type=str)

        parser.add_argument("--T_model_data_name",default='train_all_518484_from_1_500000000.pkl',type=str)
        parser.add_argument("--v_model_data_name",default='dev_all_135428_from_v_bert_rel_answer_pairwise_1_500000000.pkl',type=str)
        parser.add_argument("--t_model_data_name",default='test_all_344985_from_1_500000000.pkl',type=str)
        ## Other parameters
        parser.add_argument("--group_size",default=1,type=int,help="")
        parser.add_argument("--cache_dir",default="",type=str,help="Where do you want to store the pre-trained models downloaded from s3")
        parser.add_argument("--max_seq_length",default=100,type=int)
        parser.add_argument("--do_train",default='true',help="Whether to run training.")
        parser.add_argument("--do_eval",default='true',help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case",default='True', action='store_true',help="Set this flag if you are using an uncased model.")
        parser.add_argument("--train_batch_size",default=16,type=int,help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",default=100,type=int,help="Total batch size for eval.")
        parser.add_argument("--learning_rate",default=5e-5,type=float,help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",default=5.0,type=float,help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",default=0.1,type=float,)
        parser.add_argument("--no_cuda",action='store_true',help="Whether not to use CUDA when available")
        parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
        parser.add_argument('--seed',type=int,default=seed,help="random seed for initialization")
        parser.add_argument('--gradient_accumulation_steps',type=int,default=steps,help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
        parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")  
        args = parser.parse_args()
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        
        # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        fout_res = open(args.output_dir + 'result_predict.log', 'w', encoding='utf-8')
        # import pdb; pdb.set_trace()
        # best_model_dir_name = main(fout_res, args)
        best_model_dir_name = modelDic[N]
        test(best_model_dir_name, fout_res, args)
        dev(best_model_dir_name, fout_res, args)
        