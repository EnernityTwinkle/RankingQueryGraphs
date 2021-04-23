import numpy as np



class RelationEmbedding:
    def __init__(self) -> None:
        self.relation2idFile = '/data2/yhjia/KGEmbeddings/Freebase/knowledge graphs/relation2id.txt'
        self.relation2vecFile = '/data2/yhjia/KGEmbeddings/Freebase/embeddings/dimension_50/transe/relation2vec.bin'
        self.rel2id = self.readRelation2id()
        self.embedding = self.readVec()

    def readRelation2id(self):
        fread = open(self.relation2idFile, 'r', encoding='utf-8')
        rel2id = {}
        for line in fread:
            lineCut = line.strip().split('\t')
            if(len(lineCut) == 2):
                rel2id[lineCut[0]] = lineCut[1]
        # import pdb; pdb.set_trace()
        return rel2id


    def readVec(self):
        vec = np.memmap(self.relation2vecFile, dtype='float32', mode='r')
        embedding = np.reshape(vec, (-1, 50))
        # import pdb; pdb.set_trace()
        return embedding




if __name__ == "__main__":
    relEmb = RelationEmbedding()
    import pdb; pdb.set_trace()