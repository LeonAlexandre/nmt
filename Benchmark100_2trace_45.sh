#!/bin/bash


python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data100_2trace/data0.4/train --dev_prefix=./nmt/temp/data100_2trace/data0.4/val --test_prefix=./nmt/temp/data100_2trace/data0.4/test --out_dir=./nmt/temp/Benchmark100_2trace_uni2_adam_0.0001/model0.4 --num_train_steps=100000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics="edit_distance","hamming_distance" --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=1000 --tgt_max_len=1000

git pull
git add --all
git commit -m "completed benchmark100_2trace_uni2_adam_0.0001 0.4 10000 steps"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data100_2trace/data0.5/train --dev_prefix=./nmt/temp/data100_2trace/data0.5/val --test_prefix=./nmt/temp/data100_2trace/data0.5/test --out_dir=./nmt/temp/Benchmark100_2trace_uni2_adam_0.0001/model0.5 --num_train_steps=100000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics="edit_distance","hamming_distance" --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=1000 --tgt_max_len=1000

git pull
git add --all
git commit -m "completed benchmark100_2trace_uni2_adam_0.0001 0.5 10000 steps"
git push
