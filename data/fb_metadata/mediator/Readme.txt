from_data_0911/:
   Raw data: MKY: /home/xusheng/data_0911
   mediators_old.tsv (223 lines):
       the same as candidateMediator.txt.0
       Format: type \t number_of_entities_without_name \t ratio_of_such_entities_in_the_type
   meidators.tsv (319 lines):
       * derived from mediators_sort.tsv
       * remove all types with ratio < 0.9
       * remove all types with hit < 10
       * remove some special types


Currently, we use the data from Galaxy (~/Freebase/mediatorId.txt)
    Though I forgot how the data is generated, I believe this table is complete and correct.
