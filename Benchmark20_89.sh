#!/bin/bash

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20/data20_0.8/train --dev_prefix=./nmt/temp/data20/data20_0.8/val --test_prefix=./nmt/temp/data20/data20_0.8/test --out_dir=./nmt/temp/Benchmark20_uni2_adam_0.0001/model20_0.8 --num_train_steps=30000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001

git pull
git add --all
git commit -m "completed benchmark100_uni2_adam_0.0001 0.8"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20/data20_0.9/train --dev_prefix=./nmt/temp/data20/data20_0.9/val --test_prefix=./nmt/temp/data20/data20_0.9/test --out_dir=./nmt/temp/Benchmark20_uni2_adam_0.0001/model20_0.9 --num_train_steps=30000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type="uni" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001

git pull
git add --all
git commit -m "completed benchmark100_uni2_adam_0.0001 0.9"
git push
