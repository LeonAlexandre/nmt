#!/bin/bash

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2t/data0.1/train --dev_prefix=./nmt/temp/data20_2t/data0.1/val --test_prefix=./nmt/temp/data20_2t/data0.1/test --out_dir=./nmt/temp/model20_2encoder_avg/model0.1 --num_train_steps=10000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type=uni --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=30 --tgt_max_len=30 --src_max_len_infer=30 --num_traces=2

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2t/data0.3/train --dev_prefix=./nmt/temp/data20_2t/data0.3/val --test_prefix=./nmt/temp/data20_2t/data0.3/test --out_dir=./nmt/temp/model20_2encoder_avg/model0.3 --num_train_steps=10000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type=uni --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=30 --tgt_max_len=30 --src_max_len_infer=30 --num_traces=2

