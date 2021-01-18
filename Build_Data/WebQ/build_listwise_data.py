import sys
sys.path.append('../')
from read_query_graph import *

# N = 50
N = 40
# N = 30

# N = 20
# N = 10

if __name__ == "__main__":
    # init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    # entity_dic = read_entity(init_dir_name)
    # qid2cands = read_query_graph(init_dir_name, entity_dic)
    # qid2cands = get_pos_neg_accord_f1(qid2cands)
    # f = open('/data2/yhjia/kbqa_sp/webq_qid2question.pkl', 'rb')
    # qid2question = pickle.load(f)
    # qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    # file_name = './webq_listwise_random_top_' + str(N) + '_relall_main_rel1_type_entity_time_ordinal_before_is'
    # train_data = select_top_n_listwise(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=N)
    # write2file(file_name + 'is_train.txt', train_data)

    # # 验证集和测试集都是使用全集
    # dev_data = select_top_n_listwise(qid2question, qid2cands_dev, pos_only1=False, data_type='v', N=N)
    # write2file(file_name + 'dev_all.txt', dev_data)
    # test_data = select_top_n_listwise(qid2question, qid2cands_test, pos_only1=False, data_type='t', N=N)
    # write2file(file_name + 'test_all.txt', test_data)

    #*******************按照正负比例构建listwise训练数据***********************************
    # init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    # entity_dic = read_entity(init_dir_name)
    # qid2comp_dic = read_comp(init_dir_name)
    # qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    # qid2cands = get_pos_neg_accord_f1(qid2cands)
    # f = open('/data2/yhjia/kbqa_sp/webq_qid2question.pkl', 'rb')
    # qid2question = pickle.load(f)
    # qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    # file_name = './webq_listwise_1_' + str(N) + '_type_entity_time_ordinal_mainpath_'
    # # file_name = '/data2/yhjia/kbqa_train_data/WebQ/listwise/webq_listwise_1_' + str(N) + '_type_entity_time_ordinal_mainpath_'
    # train_data = select_top_1_n_listwise(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=N)
    # write2file(file_name + 'is_train.txt', train_data)

    # # 验证集和测试集都是使用全集
    # dev_data = select_top_1_n_listwise(qid2question, qid2cands_dev, pos_only1=False, data_type='v', N=N)
    # write2file(file_name + 'dev_all.txt', dev_data)
    # test_data = select_top_1_n_listwise(qid2question, qid2cands_test, pos_only1=False, data_type='t', N=N)
    # write2file(file_name + 'test_all.txt', test_data)

    # *************************按照正负例构建listwise数据，并且标记出每种子路径的位置***************
    init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    entity_dic = read_entity(init_dir_name)
    qid2comp_dic = read_comp(init_dir_name)
    qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    qid2cands = get_pos_neg_accord_f1(qid2cands)
    f = open('/data2/yhjia/kbqa_sp/webq_qid2question.pkl', 'rb')
    qid2question = pickle.load(f)
    qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    file_name = './webq_rank1_f01_label_position_listwise_1_' + str(N) + '_type_entity_time_ordinal_mainpath_'
    # file_name = './webq_three_answer_label_position_listwise_1_' + str(N) + '_type_entity_time_ordinal_mainpath_'
    # file_name = '/data2/yhjia/kbqa_train_data/WebQ/listwise/webq_listwise_1_' + str(N) + '_type_entity_time_ordinal_mainpath_'
    train_data = select_top_1_n_listwise(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=N)
    write2file_label_position(file_name + 'is_train.txt', train_data)

    # 验证集和测试集都是使用全集
    dev_data = select_top_1_n_listwise(qid2question, qid2cands_dev, pos_only1=False, data_type='v', N=N)
    write2file_label_position(file_name + 'dev_all.txt', dev_data)
    test_data = select_top_1_n_listwise(qid2question, qid2cands_test, pos_only1=False, data_type='t', N=N)
    write2file_label_position(file_name + 'test_all.txt', test_data)
    
    # **************************按照正负例比例构建listwise数据，并且标记出每种子路径的位置，同时每组使用更小的数目*************
    # init_dir_name = '../../runnings/candgen_WebQ/20201202_entity_time_type_ordinal/data/'
    # entity_dic = read_entity(init_dir_name)
    # qid2comp_dic = read_comp(init_dir_name)
    # qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    # qid2cands = get_pos_neg_accord_f1(qid2cands)
    # f = open('/data2/yhjia/kbqa_sp/webq_qid2question.pkl', 'rb')
    # qid2question = pickle.load(f)
    # qid2cands_train, qid2cands_dev, qid2cands_test = split_data_webq(qid2cands)
    # file_name = './webq_group10_label_position_listwise_1_' + str(N) + '_type_entity_time_ordinal_mainpath_'
    # train_data = select_top_1_n_listwise_11_per_group(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=N)
    # write2file_label_position(file_name + 'is_train.txt', train_data)

    # # 验证集和测试集都是使用全集
    # dev_data = select_top_1_n_listwise_11_per_group(qid2question, qid2cands_dev, pos_only1=False, data_type='v', N=N)
    # write2file_label_position(file_name + 'dev_all.txt', dev_data)
    # test_data = select_top_1_n_listwise_11_per_group(qid2question, qid2cands_test, pos_only1=False, data_type='t', N=N)
    # write2file_label_position(file_name + 'test_all.txt', test_data)