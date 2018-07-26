#!/bin/bash

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2t/data0.2/train --dev_prefix=./nmt/temp/data20_2t/data0.2/val --test_prefix=./nmt/temp/data20_2t/data0.2/test --out_dir=./nmt/temp/Benchmark20_2encoder_beam2/model0.2 --num_train_steps=30000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance,hamming_distance --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=20 --src_max_len_infer=20 --num_traces=2 --beam_width=2 --embed_size=5

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20_2t/data0.4/train --dev_prefix=./nmt/temp/data20_2t/data0.4/val --test_prefix=./nmt/temp/data20_2t/data0.4/test --out_dir=./nmt/temp/Benchmark20_2encoder_beam2/model0.4 --num_train_steps=50000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance,hamming_distance --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001 --src_max_len=20 --src_max_len_infer=20 --num_traces=2 --beam_width=2 --embed_size=5

echo "benchmark done"
