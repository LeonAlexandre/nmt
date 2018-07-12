#!/bin/bash

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20/data20_0.1/train --dev_prefix=./nmt/temp/data20/data20_0.1/val --test_prefix=./nmt/temp/data20/data20_0.1/test --out_dir=./nmt/temp/Benchmark20_bi12_adam_0.0001/model20_0.1 --num_train_steps=30000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type="bi" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001

git pull
git add --all
git commit -m "finished benchmark20_bi12 0.1"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20/data20_0.2/train --dev_prefix=./nmt/temp/data20/data20_0.2/val --test_prefix=./nmt/temp/data20/data20_0.2/test --out_dir=./nmt/temp/Benchmark20_bi12_adam_0.0001/model20_0.2 --num_train_steps=30000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type="bi" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001

git pull
git add --all
git commit -m "finished benchmark20_bi12 0.2"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20/data20_0.3/train --dev_prefix=./nmt/temp/data20/data20_0.3/val --test_prefix=./nmt/temp/data20/data20_0.3/test --out_dir=./nmt/temp/Benchmark20_bi12_adam_0.0001/model20_0.3 --num_train_steps=30000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type="bi" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001

git pull
git add --all
git commit -m "finished benchmark20_bi12 0.3"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20/data20_0.4/train --dev_prefix=./nmt/temp/data20/data20_0.4/val --test_prefix=./nmt/temp/data20/data20_0.4/test --out_dir=./nmt/temp/Benchmark20_bi12_adam_0.0001/model20_0.4 --num_train_steps=30000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type="bi" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001

git pull
git add --all
git commit -m "finished benchmark20_bi12 0.4"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data20/data20_0.5/train --dev_prefix=./nmt/temp/data20/data20_0.5/val --test_prefix=./nmt/temp/data20/data20_0.5/test --out_dir=./nmt/temp/Benchmark20_bi12_adam_0.0001/model20_0.5 --num_train_steps=30000 --steps_per_stats=100 --num_layers=2 --num_units=128 --dropout=0.2 --metrics=edit_distance --share_vocab=True --encoder_type="bi" --steps_per_external_eval=500 --optimizer=adam --learning_rate=0.0001

git pull
git add --all
git commit -m "finished benchmark20_bi12 0.5"
git push

echo "Benchmark20_bi12_adam_0.0001 complete"
