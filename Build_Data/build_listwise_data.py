from read_query_graph import *
N = 50
# N = 30
# N = 40
# N = 20
# N = 10

if __name__ == "__main__":
    # init_dir_name = '../Generate_QueryGraph/Luo/runnings/candgen_CompQ/20201130_entity_time_type_ordinal/data/'
    # entity_dic = read_entity(init_dir_name)
    # qid2cands = read_query_graph(init_dir_name, entity_dic)
    # qid2cands = get_pos_neg_accord_f1(qid2cands)
    # f = open('/data2/yhjia/kbqa_sp/compq_qid2question.pkl', 'rb')
    # qid2question = pickle.load(f)
    # qid2cands_train, qid2cands_dev, qid2cands_test = split_data_compq(qid2cands)
    # # file_name = './compq_listwise_top_' + str(N) + '_relall_main_type_entity_timestr_before_'
    # file_name = './compq_listwise_top_' + str(N) + '_relall_main_type_entity_time_ordinal_rank_before_'
    # train_data = select_top_n_listwise(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=N)
    # write2file(file_name + 'is_train.txt', train_data)

    # # 验证集和测试集都是使用全集
    # dev_data = select_top_n_listwise(qid2question, qid2cands_dev, pos_only1=False, data_type='v', N=N)
    # write2file(file_name + 'dev_all.txt', dev_data)
    # test_data = select_top_n_listwise(qid2question, qid2cands_test, pos_only1=False, data_type='t', N=N)
    # write2file(file_name + 'test_all.txt', test_data)


    #*******************按照正负比例构建listwise训练数据***********************************
    init_dir_name = '../Generate_QueryGraph/Luo/runnings/candgen_CompQ/20201130_entity_time_type_ordinal/data/'
    entity_dic = read_entity(init_dir_name)
    qid2comp_dic = read_comp(init_dir_name)
    qid2cands = read_query_graph(init_dir_name, entity_dic, qid2comp_dic)
    qid2cands = get_pos_neg_accord_f1(qid2cands)
    f = open('/data2/yhjia/kbqa_sp/compq_qid2question.pkl', 'rb')
    qid2question = pickle.load(f)
    qid2cands_train, qid2cands_dev, qid2cands_test = split_data_compq(qid2cands)
    # file_name = './compq_listwise_top_' + str(N) + '_relall_main_type_entity_timestr_before_'
    file_name = './compq_listwise_1_' + str(N) + '_type_entity_time_ordinal_mainpath_'
    train_data = select_top_1_n_listwise(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=N)
    write2file(file_name + '_train.txt', train_data)

    # 验证集和测试集都是使用全集
    dev_data = select_top_1_n_listwise(qid2question, qid2cands_dev, pos_only1=False, data_type='v', N=N)
    write2file(file_name + '_dev.txt', dev_data)
    test_data = select_top_1_n_listwise(qid2question, qid2cands_test, pos_only1=False, data_type='t', N=N)
    write2file(file_name + '_test.txt', test_data)

    #**************按照正负比例构建listwise训练数据，同时使用new_search候选数据******************
    # init_dir_name = '../Generate_QueryGraph/Luo/runnings/candgen_CompQ/20201130_entity_time_type_ordinal/data/'
    # entity_dic = read_entity(init_dir_name)
    # qid2cands = read_query_graph(init_dir_name, entity_dic)
    # qid2cands_improvement = get_cands_from_improvement('/data2/yhjia/20201210_qid2cands_compq_yh')
    # qid2cands = get_pos_neg_accord_f1_with_new_search(qid2cands, qid2cands_improvement)
    # f = open('/data2/yhjia/kbqa_sp/compq_qid2question.pkl', 'rb')
    # qid2question = pickle.load(f)
    # qid2cands_train, qid2cands_dev, qid2cands_test = split_data_compq(qid2cands)
    # file_name = './compq_listwise_new_search_1_' + str(N) + '_relall_main_type_entity_time_ordinal_before_'
    # train_data = select_top_1_n_listwise(qid2question, qid2cands_train, pos_only1=False, data_type='T', N=N)
    # write2file(file_name + 'is_train.txt', train_data)