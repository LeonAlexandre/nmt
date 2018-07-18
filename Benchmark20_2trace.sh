#!/bin/bash

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2trace/data0.1/train --dev_prefix=./nmt/temp/data20_2trace/data0.1/val --test_prefix=./nmt/temp/data20_2trace/data0.1/test --out_dir=./nmt/temp/Benchmark20_2trace_uni2_adam_0.0001/model0.1 --num_train_steps=50000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=["edit_distance","hamming_distance"] --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=1000 --tgt_max_len=1000

git pull
git add --all
git commit -m "completed benchmark20_2trace_uni2_adam_0.0001 0.1"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2trace/data0.2/train --dev_prefix=./nmt/temp/data20_2trace/data0.2/val --test_prefix=./nmt/temp/data20_2trace/data0.2/test --out_dir=./nmt/temp/Benchmark20_2trace_uni2_adam_0.0001/model0.2 --num_train_steps=50000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=["edit_distance","hamming_distance"] --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=1000 --tgt_max_len=1000

git pull
git add --all
git commit -m "completed benchmark20_2trace_uni2_adam_0.0001 0.2"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2trace/data0.3/train --dev_prefix=./nmt/temp/data20_2trace/data0.3/val --test_prefix=./nmt/temp/data20_2trace/data0.3/test --out_dir=./nmt/temp/Benchmark20_2trace_uni2_adam_0.0001/model0.3 --num_train_steps=50000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=["edit_distance","hamming_distance"] --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=1000 --tgt_max_len=1000

git pull
git add --all
git commit -m "completed benchmark20_2trace_uni2_adam_0.0001 0.3"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2trace/data0.4/train --dev_prefix=./nmt/temp/data20_2trace/data0.4/val --test_prefix=./nmt/temp/data20_2trace/data0.4/test --out_dir=./nmt/temp/Benchmark20_2trace_uni2_adam_0.0001/model0.4 --num_train_steps=50000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=["edit_distance","hamming_distance"] --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=1000 --tgt_max_len=1000

git pull
git add --all
git commit -m "completed benchmark20_2trace_uni2_adam_0.0001 0.4"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2trace/data0.5/train --dev_prefix=./nmt/temp/data20_2trace/data0.5/val --test_prefix=./nmt/temp/data20_2trace/data0.5/test --out_dir=./nmt/temp/Benchmark20_2trace_uni2_adam_0.0001/model0.5 --num_train_steps=50000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=["edit_distance","hamming_distance"] --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=1000 --tgt_max_len=1000

git pull
git add --all
git commit -m "completed benchmark20_2trace_uni2_adam_0.0001 0.5"
git push
