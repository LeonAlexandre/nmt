# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ["BatchedInput", "BatchedInput2t", "BatchedInputNt",
           "get_iterator", "get_iterator2t", "get_iteratorNt",
           "get_infer_iterator", "get_infer_iterator2t", "get_infer_iteratorNt"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
  pass

class BatchedInput2t(
    collections.namedtuple("BatchedInput2t",
                           ("initializer", "trace0", "trace1", "target_input",
                            "target_output", "trace0_sequence_length", "trace1_sequence_length",
                            "target_sequence_length"))):
  pass

class BatchedInputNt(
    # "traces" is a tuple of batched inputs "trace0", "trace1", ..., "traceN"
    # "traces_sequence_length" is a tuple of batched input sequence lengths
    collections.namedtuple("BatchedInputNt",
                           ("initializer", "traces", "target_input",
                            "target_output", "traces_sequence_length",
                            "target_sequence_length"))):
  pass


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       src_max_len=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0))  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=None,
      target_output=None,
      source_sequence_length=src_seq_len,
      target_sequence_length=None)


def get_infer_iterator2t(trace0_dataset,
                         trace1_dataset,
                         src_vocab_table,
                         batch_size,
                         eos,
                         src_max_len=None,
                         hparams=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  #trace0_dataset = trace0_dataset.map(lambda trace0: tf.string_split([trace0]).values)
  #trace1_dataset = trace1_dataset.map(lambda trace1: tf.string_split([trace1]).values)

  traces_dataset = tuple([trace0_dataset,trace1_dataset])

  src_dataset = tf.data.Dataset.zip(traces_dataset)
  #print("Infer dataset after zip: " + str(src_dataset))

  def varg_string_split(*x):
    strings = []
    #print(x[0])  # 3 arguments
    for i in x[0]:
      #print("The individual arguments are: " + str(i))
      strings.append(tf.string_split([i]).values)
    return tuple(strings)

  src_dataset = src_dataset.map(lambda *x: varg_string_split(x))

  def varg_len_cutoff(*x, max_len):
    results = []
    for i in x[0]:
      results.append(i[:max_len])
    return tuple(results)

  if src_max_len:
    src_dataset = src_dataset.map(lambda *x: varg_len_cutoff(x, max_len=src_max_len))
    #src_dataset = src_dataset.map(lambda trace0, trace1: (trace0[:src_max_len], trace1[:src_max_len]))
    #print("Infer dataset after len cutoff")

  # Convert the word strings to ids
  #src_dataset = src_dataset.map(
  #    lambda trace0, trace1: (tf.cast(src_vocab_table.lookup(trace0), tf.int32), tf.cast(src_vocab_table.lookup(trace1), tf.int32)))
  
  
  def varg_vocab_lookup(*x):
    lookup_results = []
    for i in x[0]:
      lookup_results.append(tf.cast(src_vocab_table.lookup(i), tf.int32))
    return tuple(lookup_results)
  src_dataset = src_dataset.map(lambda *x: varg_vocab_lookup(x))
  #print("Infer dataset after vocab lookup: " + str(src_dataset))
  # Add in the word counts.
  #src_dataset = src_dataset.map(lambda trace0, trace1: (trace0, tf.size(trace0), trace1, tf.size(trace1)))
  
  
  def varg_get_size(*x):
    sizes = []
    for i in x[0]:
      sizes.append(tf.size(i))
    return tuple(sizes)
  src_dataset = src_dataset.map(lambda *x: (*(x),*(varg_get_size(x))))
  #print("Infer dataset after adding in seq length: " + str(src_dataset))

  def batching_func(x):
    
    padded_shape = []
    for i in range(hparams.num_traces):
      padded_shape.append(tf.TensorShape([None]))
    for i in range(hparams.num_traces):
      padded_shape.append(tf.TensorShape([]))
    padded_shape = tuple(padded_shape)

    padding_values = []
    for i in range(hparams.num_traces):
      padding_values.append(src_eos_id)
    for i in range(hparams.num_traces):
      padding_values.append(0)
    padding_values = tuple(padding_values)
    

    '''
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # trace0
            tf.TensorShape([]),  # trace0_len
            tf.TensorShape([None]),  # trace1
            tf.TensorShape([])) ,   # trace1_len  
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # trace0
            0,   # trace0_len -- unused
            src_eos_id,  # trace1
            0))  # trace1_len -- unused
    '''
    return x.padded_batch(
        batch_size,
        padded_shapes=(padded_shape),
        padding_values=(padding_values))
    

  batched_dataset = batching_func(src_dataset)
  #print("Batched infer dataset: " + str(batched_dataset))
  batched_iter = batched_dataset.make_initializable_iterator()
  #print("Batched infer iter: " + str(batched_iter))
  (trace0_ids, trace1_ids, trace0_seq_len, trace1_seq_len) = batched_iter.get_next()
  return BatchedInput2t(
      initializer=batched_iter.initializer,
      trace0=trace0_ids,
      trace1=trace1_ids,
      target_input=None,
      target_output=None,
      trace0_sequence_length=trace0_seq_len,
      trace1_sequence_length=trace1_seq_len,
      target_sequence_length=None)


def get_infer_iteratorNt(traces_dataset,
                         src_vocab_table,
                         batch_size,
                         eos,
                         src_max_len=None,
                         hparams=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_dataset = tf.data.Dataset.zip(traces_dataset)
  
  #print("Infer dataset after zip: " + str(src_dataset))

  def varg_string_split(*x):
    strings = []
    #print(x[0])  # 3 arguments
    for i in x[0]:
      #print("The individual arguments are: " + str(i))
      strings.append(tf.string_split([i]).values)
    return tuple(strings)

  src_dataset = src_dataset.map(lambda *x: varg_string_split(x))
  #print("Infer dataset after split: " + str(src_dataset))

  def varg_len_cutoff(*x, max_len):
    results = []
    for i in x[0]:
      results.append(i[:max_len])
    return tuple(results)

  if src_max_len:
    src_dataset = src_dataset.map(lambda *x: varg_len_cutoff(x, max_len=src_max_len))
    #print("Infer dataset after len cutoff")

  # Convert the word strings to ids  
  def varg_vocab_lookup(*x):
    lookup_results = []
    for i in x[0]:
      lookup_results.append(tf.cast(src_vocab_table.lookup(i), tf.int32))
    return tuple(lookup_results)

  src_dataset = src_dataset.map(lambda *x: varg_vocab_lookup(x))
  #print("Infer dataset after vocab lookup: " + str(src_dataset))

  # Add in the word counts.
  def varg_get_size(*x):
    sizes = []
    for i in x[0]:
      sizes.append(tf.size(i))
    return tuple(sizes)

  src_dataset = src_dataset.map(lambda *x: (*(x),*(varg_get_size(x))))
  #print("Infer dataset after adding in seq length: " + str(src_dataset))

  def batching_func(x):
    
    padded_shape = []
    for i in range(hparams.num_traces):
      padded_shape.append(tf.TensorShape([None]))
    for i in range(hparams.num_traces):
      padded_shape.append(tf.TensorShape([]))
    padded_shape = tuple(padded_shape)

    padding_values = []
    for i in range(hparams.num_traces):
      padding_values.append(src_eos_id)
    for i in range(hparams.num_traces):
      padding_values.append(0)
    padding_values = tuple(padding_values)
    
    return x.padded_batch(
        batch_size,
        padded_shapes=(padded_shape),
        padding_values=(padding_values))
    

  batched_dataset = batching_func(src_dataset)
  #print("Batched infer dataset: " + str(batched_dataset))
  batched_iter = batched_dataset.make_initializable_iterator()
  #print("Batched infer iter: " + str(batched_iter))

  traces_and_lens = [1] * 2 * hparams.num_traces
  
  (*traces_and_lens,) = batched_iter.get_next()

  traces = tuple(traces_and_lens[0:hparams.num_traces])
  trace_lens = tuple(traces_and_lens[hparams.num_traces:])

  return BatchedInputNt(
      initializer=batched_iter.initializer,
      traces=traces,
      target_input=None,
      target_output=None,
      traces_sequence_length=trace_lens,
      target_sequence_length=None)


def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (src,
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt_in, tgt_out: (
          src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)

  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
   tgt_seq_len) = (batched_iter.get_next())
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len)

def get_iterator2t(trace0_dataset,
                   trace1_dataset,
                   tgt_dataset,
                   src_vocab_table,
                   tgt_vocab_table,
                   batch_size,
                   sos,
                   eos,
                   random_seed,
                   num_buckets,
                   src_max_len=None,
                   tgt_max_len=None,
                   num_parallel_calls=4,
                   output_buffer_size=None,
                   skip_count=None,
                   num_shards=1,
                   shard_index=0,
                   reshuffle_each_iteration=True,
                   hparams=None):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  traces_dataset = tuple([trace0_dataset,trace1_dataset])
  #src_tgt_dataset = tf.data.Dataset.zip((trace0_dataset, trace1_dataset, tgt_dataset))
  src_tgt_dataset = tf.data.Dataset.zip((*traces_dataset, tgt_dataset))
  print("src_tgt_dataset after zip: " + str(src_tgt_dataset))
  #src_tgt_dataset = tf.data.Dataset.zip((*traces_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  print("src_tgt_dataset before split: " + str(src_tgt_dataset))
  #src_tgt_dataset = src_tgt_dataset.map(
  #    lambda trace0, trace1, tgt: (
  #        tf.string_split([trace0]).values, tf.string_split([trace1]).values, tf.string_split([tgt]).values),
  #    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)  
  def varg_string_split(*x):
    strings = []
    #print(x[0])  # 3 arguments
    for i in x[0]:
      #print("The individual arguments are: " + str(i))
      strings.append(tf.string_split([i]).values)
    return tuple(strings)
  src_tgt_dataset = src_tgt_dataset.map(
      lambda *x: varg_string_split(x), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  print("src_tgt_dataset after split: " + str(src_tgt_dataset))

  # Filter zero length input sequences.
  #src_tgt_dataset = src_tgt_dataset.filter(
  #    lambda trace0, trace1, tgt: tf.logical_and(tf.logical_and(tf.size(trace0) > 0, tf.size(trace1) > 0),  tf.size(tgt) > 0))
  def varg_logical_and(*x):
    predicates = tf.size(x[0][0]) > 0
    for i in x[0]:
      predicates = tf.logical_and(predicates, tf.size(i) > 0)
    return predicates
  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda *x: varg_logical_and(x) )
  print("src_tgt_dataset after zero len filter: " + str(src_tgt_dataset))

  def varg_len_cutoff(*x, max_len):
    results = []
    for i in x[0]:
      results.append(i[:max_len])
    return tuple(results)

  if src_max_len:
    #src_tgt_dataset = src_tgt_dataset.map(
    #    lambda trace0, trace1, tgt: (trace0[:src_max_len], trace1[:src_max_len], tgt),
    #    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.map(
        lambda *x: varg_len_cutoff(x,max_len=src_max_len),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    print("src_tgt_dataset after src_max_len cutoff: " + str(src_tgt_dataset))
  if tgt_max_len:
    #src_tgt_dataset = src_tgt_dataset.map(
    #    lambda trace0, trace1, tgt: (trace0, trace1, tgt[:tgt_max_len]),
    #    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    src_tgt_dataset = src_tgt_dataset.map(
        lambda *x: varg_len_cutoff(x,max_len=tgt_max_len),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    print("src_tgt_dataset after tgt_max_len cutoff: " + str(src_tgt_dataset))
  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  #src_tgt_dataset = src_tgt_dataset.map(
  #    lambda trace0, trace1, tgt: (tf.cast(src_vocab_table.lookup(trace0), tf.int32),
  #                                 tf.cast(src_vocab_table.lookup(trace1), tf.int32),
  #                                 tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
  #    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  def varg_vocab_lookup(*x):
    lookup_results = []
    n_trace = 0
    for i in x[0]:
      if n_trace < hparams.num_traces:
        lookup_results.append(tf.cast(src_vocab_table.lookup(i), tf.int32))
        n_trace += 1
    lookup_results.append(tf.cast(tgt_vocab_table.lookup(i), tf.int32))
    return tuple(lookup_results)
  src_tgt_dataset = src_tgt_dataset.map(
      lambda *x: varg_vocab_lookup(x),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  print("src_tgt_dataset after vocab lookup: " + str(src_tgt_dataset))
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  #src_tgt_dataset = src_tgt_dataset.map(
  #    lambda trace0, trace1, tgt: (trace0, trace1, 
  #                      tf.concat(([tgt_sos_id], tgt), 0),
  #                      tf.concat((tgt, [tgt_eos_id]), 0)),
  #    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  def varg_append_token(*x):
    appended = []
    reverse_x = reversed(x[0])
    first = True
    for i in reverse_x:
      if first:
        prefix_sos = tf.concat(([tgt_sos_id], i), 0)
        suffix_eos = tf.concat((i, [tgt_eos_id]), 0)
      else:
        break
    n_trace = 0
    for i in x[0]:
      if n_trace < hparams.num_traces:
        appended.append(i)
        n_trace += 1
      else:
        break
    appended.append(prefix_sos)
    appended.append(suffix_eos)
    return tuple(appended)
  src_tgt_dataset = src_tgt_dataset.map(
      lambda *x: (varg_append_token(x)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  print("src_tgt_dataset after tgt prefix/suffix: " + str(src_tgt_dataset))
  # Add in sequence lengths.
  #src_tgt_dataset = src_tgt_dataset.map(
  #    lambda trace0, trace1, tgt_in, tgt_out: (
  #        trace0, trace1, tgt_in, tgt_out, tf.size(trace0), tf.size(trace1), tf.size(tgt_in)),
  #    num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  def varg_get_size(*x):
    sizes = []
    n_trace = 0
    for i in x[0]:
      if n_trace < hparams.num_traces+1:
        sizes.append(tf.size(i))
        n_trace += 1
      else:
        break
    return tuple(sizes)
  src_tgt_dataset = src_tgt_dataset.map(
      lambda *x: (*(x), *(varg_get_size(x))),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  print("src_tgt_dataset after adding seq lens: " + str(src_tgt_dataset))

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    padded_shape = []
    for i in range(hparams.num_traces+2):
      padded_shape.append(tf.TensorShape([None]))
    for i in range(hparams.num_traces+1):
      padded_shape.append(tf.TensorShape([]))
    padded_shape = tuple(padded_shape)

    padding_values = []
    for i in range(hparams.num_traces):
      padding_values.append(src_eos_id)
    for i in range(2):
      padding_values.append(tgt_eos_id)
    for i in range(hparams.num_traces+1):
      padding_values.append(0)
    padding_values = tuple(padding_values)
    '''
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.

        padded_shapes=(
            tf.TensorShape([None]),  # trace0
            tf.TensorShape([None]),  # trace1
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # trace0_len
            tf.TensorShape([]),  # trace1_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # trace0
            src_eos_id,  # trace1
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # trace0_len -- unused
            0,  # trace1_len -- unused
            0))  # tgt_len -- unused
      '''  
    return x.padded_batch(
        batch_size,
        padded_shapes=(padded_shape),
        padding_values=(padding_values))

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, unused4, unused5, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)
    print("batched_dataset: " + str(batched_dataset))

  batched_iter = batched_dataset.make_initializable_iterator()
  print("batched_iter: " + str(batched_iter))
  (trace0_ids, trace1_ids, tgt_input_ids, tgt_output_ids, trace0_seq_len, trace1_seq_len,
   tgt_seq_len) = (batched_iter.get_next())

  BatchedInputIter = BatchedInput2t(
      initializer=batched_iter.initializer,
      trace0=trace0_ids,
      trace1=trace1_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      trace0_sequence_length=trace0_seq_len,
      trace1_sequence_length=trace1_seq_len,
      target_sequence_length=tgt_seq_len)
  print("BatchedInputIter: " + str(BatchedInputIter))

  return BatchedInputIter


def get_iteratorNt(traces_dataset,
                   tgt_dataset,
                   src_vocab_table,
                   tgt_vocab_table,
                   batch_size,
                   sos,
                   eos,
                   random_seed,
                   num_buckets,
                   src_max_len=None,
                   tgt_max_len=None,
                   num_parallel_calls=4,
                   output_buffer_size=None,
                   skip_count=None,
                   num_shards=1,
                   shard_index=0,
                   reshuffle_each_iteration=True,
                   hparams=None):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((*traces_dataset, tgt_dataset))
  #print("src_tgt_dataset after zip: " + str(src_tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  #print("src_tgt_dataset before split: " + str(src_tgt_dataset))
  
  # Split dataset into sentences
  def varg_string_split(*x):
    strings = []
    #print(x[0])  # 3 arguments
    for i in x[0]:
      #print("The individual arguments are: " + str(i))
      strings.append(tf.string_split([i]).values)
    return tuple(strings)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda *x: varg_string_split(x), num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  #print("src_tgt_dataset after split: " + str(src_tgt_dataset))

  # Filter zero length input sequences.
  def varg_logical_and(*x):
    predicates = tf.size(x[0][0]) > 0
    for i in x[0]:
      predicates = tf.logical_and(predicates, tf.size(i) > 0)
    return predicates

  src_tgt_dataset = src_tgt_dataset.filter(
      lambda *x: varg_logical_and(x) )
  #print("src_tgt_dataset after zero len filter: " + str(src_tgt_dataset))

  def varg_len_cutoff(*x, max_len):
    results = []
    for i in x[0]:
      results.append(i[:max_len])
    return tuple(results)

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda *x: varg_len_cutoff(x,max_len=src_max_len),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    #print("src_tgt_dataset after src_max_len cutoff: " + str(src_tgt_dataset))
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda *x: varg_len_cutoff(x,max_len=tgt_max_len),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
    #print("src_tgt_dataset after tgt_max_len cutoff: " + str(src_tgt_dataset))

  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  def varg_vocab_lookup(*x):
    lookup_results = []
    n_trace = 0
    for i in x[0]:
      if n_trace < hparams.num_traces:
        lookup_results.append(tf.cast(src_vocab_table.lookup(i), tf.int32))
        n_trace += 1
    lookup_results.append(tf.cast(tgt_vocab_table.lookup(i), tf.int32))
    return tuple(lookup_results)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda *x: varg_vocab_lookup(x),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  #print("src_tgt_dataset after vocab lookup: " + str(src_tgt_dataset))

  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  def varg_append_token(*x):
    appended = []
    reverse_x = reversed(x[0])
    first = True
    for i in reverse_x:
      if first:
        prefix_sos = tf.concat(([tgt_sos_id], i), 0)
        suffix_eos = tf.concat((i, [tgt_eos_id]), 0)
      else:
        break
    n_trace = 0
    for i in x[0]:
      if n_trace < hparams.num_traces:
        appended.append(i)
        n_trace += 1
      else:
        break
    appended.append(prefix_sos)
    appended.append(suffix_eos)
    return tuple(appended)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda *x: (varg_append_token(x)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  #print("src_tgt_dataset after tgt prefix/suffix: " + str(src_tgt_dataset))

  # Add in sequence lengths.
  def varg_get_size(*x):
    sizes = []
    n_trace = 0
    for i in x[0]:
      if n_trace < hparams.num_traces+1:
        sizes.append(tf.size(i))
        n_trace += 1
      else:
        break
    return tuple(sizes)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda *x: (*(x), *(varg_get_size(x))),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  #print("src_tgt_dataset after adding seq lens: " + str(src_tgt_dataset))

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    padded_shape = []
    for i in range(hparams.num_traces+2):
      padded_shape.append(tf.TensorShape([None]))
    for i in range(hparams.num_traces+1):
      padded_shape.append(tf.TensorShape([]))
    padded_shape = tuple(padded_shape)

    padding_values = []
    for i in range(hparams.num_traces):
      padding_values.append(src_eos_id)
    for i in range(2):
      padding_values.append(tgt_eos_id)
    for i in range(hparams.num_traces+1):
      padding_values.append(0)
    padding_values = tuple(padding_values)

    return x.padded_batch(
        batch_size,
        padded_shapes=(padded_shape),
        padding_values=(padding_values))

  '''
  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, unused4, unused5, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
  '''
  batched_dataset = batching_func(src_tgt_dataset)
  #print("batched_dataset: " + str(batched_dataset))

  batched_iter = batched_dataset.make_initializable_iterator()
  #print("batched_iter: " + str(batched_iter))

  unpacked_iter = []

  #(*(traces), tgt_input_ids, tgt_output_ids, *(trace_lens), tgt_seq_len) = (batched_iter.get_next())
  (*unpacked_iter,) = (batched_iter.get_next())

  traces = tuple(unpacked_iter[0:hparams.num_traces])
  tgt_input_ids = unpacked_iter[hparams.num_traces]
  tgt_output_ids = unpacked_iter[hparams.num_traces+1]
  trace_lens = tuple(unpacked_iter[hparams.num_traces+2:2*hparams.num_traces+2])
  tgt_seq_len = unpacked_iter[-1]

  BatchedInputIter = BatchedInputNt(
      initializer=batched_iter.initializer,
      traces=traces,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      traces_sequence_length=trace_lens,
      target_sequence_length=tgt_seq_len)
  #print("BatchedInputIter: " + str(BatchedInputIter))

  return BatchedInputIter
