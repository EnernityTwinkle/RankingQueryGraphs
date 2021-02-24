with open('candidateMediator.txt.3', 'r') as br:
    lines = br.readlines()

good_tups = []
bad_tups = []
for line in lines:
    spt = line.strip().split('\t')
    hit = int(spt[1])
    ratio = float(spt[2])
    if ratio < 0.9:
        bad_tups.append((spt[0], hit, ratio))
    else:
        good_tups.append((spt[0], hit, ratio))

good_tups.sort(key=lambda tup: tup[1], reverse=True)        # good: sort by hit num DESC
bad_tups.sort(key=lambda tup: tup[2], reverse=True)         # bad: sort by ratio DESC

with open('med_sort.txt', 'w') as bw:
    for mid, hit, ratio in good_tups:
        bw.write('%s\t%d\t%f\n' % (mid, hit, ratio))
    for mid, hit, ratio in bad_tups:
        bw.write('# %s\t%d\t%f\n' % (mid, hit, ratio))

