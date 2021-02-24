# Goal: make paired time predicates

from kangqi.util.LogUtil import LogInfo


tp_pred_dict = {}

with open('time_predicates.txt', 'r') as br:
    for line in br.readlines():
        pred = line.strip()
        last_dot_idx = pred.rfind('.')
        tp = pred[:last_dot_idx]
        tp_pred_dict.setdefault(tp, []).append(pred[last_dot_idx+1:])

begin_word_set = set(['from', 'begin', 'begins', 'began', 'beginning',
                      'start', 'starts', 'started', 'starting'])
end_word_set = set(['to', 'end', 'ends', 'ended', 'ending'])

bw = open('time_pairs.txt', 'w')
for tp, pred_list in tp_pred_dict.items():
    tup_list = []       # [(pred_name, category, keyword, remain_word)]
    for pred_name in pred_list:
        name_set = set(pred_name.split('_'))
        category = 'None'
        keyword = ''
        for name in name_set:
            if name in begin_word_set:
                keyword = name
                category = 'start'
                break
            elif name in end_word_set:
                keyword = name
                category = 'end'
                break
        if category != 'None':
            name_set.remove(keyword)
            remain_word = '_'.join(sorted(name_set))
            tup = (category, keyword, remain_word, pred_name)
            tup_list.append(tup)
            LogInfo.logs('%s', tup)
    if len(tup_list) > 0:
        LogInfo.logs('tups = %d', len(tup_list))

    for cate_a, kw_a, rem_a, name_a in tup_list:
        if cate_a != 'start':
            continue
        for cate_b, kw_b, rem_b, name_b in tup_list:
            if cate_b != 'end':
                continue
            if rem_a != rem_b:
                continue
            bw.write('%s.%s\t%s.%s\n' % (tp, name_a, tp, name_b))
bw.close()
