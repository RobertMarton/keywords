python tfidf.py --train_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_tfidf/train.csv \
--train_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_tfidf/train_tfidf.csv \
--dev_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_tfidf/dev.csv \
--dev_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_tfidf/dev_tfidf.csv \
--test_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_tfidf/Test_DataSet.csv \
--test_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_tfidf/Test_DataSet_tfidf.csv \
--stop_words /home/rhk/projects/gcp_vm/keywords/data/stop_words.txt > log_tfidf.txt

python hanlp_keywords.py --train_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_hanlp/train.csv \
--train_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_hanlp/train_hanlp.csv \
--dev_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_hanlp/dev.csv \
--dev_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_hanlp/dev_hanlp.csv \
--test_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_hanlp/Test_DataSet.csv \
--test_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_hanlp/Test_DataSet_hanlp.csv \
--stop_words /home/rhk/projects/gcp_vm/keywords/data/stop_words.txt > log_hanlp.txt

python jieba_tfidf.py --train_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/train.csv \
--train_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/train_jiebatfidf.csv \
--dev_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/dev.csv \
--dev_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/dev_jiebatfidf.csv \
--test_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/Test_DataSet.csv \
--test_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatfidf/Test_DataSet_jiebatfidf.csv \
--stop_words /home/rhk/projects/gcp_vm/keywords/data/stop_words.txt > log_jiebatfidf.txt

python jieba_textrank.py --train_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatr/train.csv \
--train_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatr/train_jiebatr.csv \
--dev_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatr/dev.csv \
--dev_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatr/dev_jiebatr.csv \
--test_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatr/Test_DataSet.csv \
--test_tfidf_path /home/rhk/projects/gcp_vm/keywords/data/ccf_data_jiebatr/Test_DataSet_jiebatr.csv \
--stop_words /home/rhk/projects/gcp_vm/keywords/data/stop_words.txt > log_jiebatr.txt