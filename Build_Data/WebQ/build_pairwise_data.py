import sys
sys.path.append('../')
from read_query_graph import *

# NEG_NUM = 100
# NEG_NUM = 60
# NEG_NUM = 50
# NEG_NUM = 40
# NEG_NUM = 30
# NEG_NUM = 20

if __name__ == "__main__":
    # init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    # entity_dic = read_entity(init_dir_name)
    # qid2comp_dic = read_comp(init_dir_name)
    # qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    # qid2cands = get_pos_neg_accord_f1(qid2cands)
    # f = open('/data2/yhjia/kbqa_sp/webq_qid2question.pkl', 'rb')
    # qid2question = pickle.load(f)
    # qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
    # # file_name = './webq_pairwise_rank1_p01_f01_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
    # # file_name = '/data2/yhjia/kbqa_train_data/WebQ/pairwise/webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'

    # # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
    # # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_mainpath_'
    # # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_mainrel_'
    # train_data = select_pairwise_data(qid2question, qid2cands_train, pos_only1=False, data_type='T', neg=NEG_NUM)
    # write2file(file_name + '_train_all.txt', train_data)
    # # 验证集和测试集都是使用全集
    # dev_data = select_pairwise_data(qid2question, qid2cands_dev, pos_only1=False, data_type='v', neg=NEG_NUM)
    # write2file(file_name + 'dev_all.txt', dev_data)
    # test_data = select_pairwise_data(qid2question, qid2cands_test, pos_only1=False, data_type='t', neg=NEG_NUM)
    # write2file(file_name + 'test_all.txt', test_data)

    # *****************************标记出不同子路径的位置*******************
    init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    entity_dic = read_entity(init_dir_name)
    qid2comp_dic = read_comp(init_dir_name)
    qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    qid2cands = get_pos_neg_accord_f1(qid2cands)
    f = open('../../data/webq_qid2question.pkl', 'rb')
    qid2question = pickle.load(f)
    qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    for NEG_NUM in [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 140]:
        # file_name = './webq_one_answer_label_position_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        # file_name = './webq_600rank1_f01_label_position_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        # file_name = './webq_pairwise_rank1_p01_f01_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        # file_name = '/data2/yhjia/kbqa_train_data/WebQ/pairwise/webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        file_name = '../../../bert_rank_data/webq/webq_rank1_f01_true1_label_position_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'


        # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_type_entity_time_ordinal_mainpath_'
        # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_mainpath_'
        # file_name = './webq_pairwise_neg_' + str(NEG_NUM) + '_mainrel_'
        # train_data = select_pairwise_data(qid2question, qid2cands_train, pos_only1=False, data_type='T', neg=NEG_NUM)
        # train_data = select_pairwise_gradual_true1(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=NEG_NUM)
        train_data = select_pairwise_gradual_true1(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=NEG_NUM)
        write2file_label_position(file_name + '_train_all.txt', train_data)
    # 验证集和测试集都是使用全集
    # dev_data = select_pairwise_data(qid2question, qid2cands_dev, pos_only1=False, data_type='v', neg=NEG_NUM)
    # write2file_label_position(file_name + 'dev_all.txt', dev_data)
    # test_data = select_pairwise_data(qid2question, qid2cands_test, pos_only1=False, data_type='t', neg=NEG_NUM)
    # write2file_label_position(file_name + 'test_all.txt', test_data)