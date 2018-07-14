#!/bin/bash

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data100/data100_0.1/train --dev_prefix=./nmt/t$

git pull
git add --all
git commit -m "completed benchmark100_uni2_adam_0.0001 0.1"
git push

python -m nmt.nmt --src=trace --tgt=label --vocab_prefix=./nmt/temp/binary_seq_vocab/vocab --train_prefix=./nmt/temp/data100/data100_0.3/train --dev_prefix=./nmt/t$

git pull
git add --all
git commit -m "completed benchmark100_uni2_adam_0.0001 0.3"
git push
