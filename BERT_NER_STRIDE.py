#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
# Copyright 2018 The Google AI Language Team Authors.
# Copyright 2019 The BioNLP-HZAU Kaiyin Zhou
# Time:2019/04/08
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import collections
import os
import pickle
from absl import flags,logging
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf
import metrics
import numpy as np
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

# if you download cased checkpoint you should use "False",if uncased you should use
# "True"
# if we used in bio-medical field，don't do lower case would be better!

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string("middle_output", "middle_data", "Dir was used to store middle data!")
flags.DEFINE_string("crf", "True", "use crf!")
flags.DEFINE_integer("doc_stride", 128, "doc_stride default 128")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 mask,
                 segment_ids,
                 label_ids,
                 token_is_max_context,
                 doc_span_index,
                 example_index,
                 is_real_example=True):
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.token_is_max_context = token_is_max_context
        self.doc_span_index = doc_span_index
        self.example_index = example_index
        self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls,input_file):
        """Read a BIO data!"""
        rf = open(input_file,'r')
        lines = [];words = [];labels = []
        # pdb.set_trace()
        for line in rf:
            word = line.strip().split(' ')[0]
            print("word > ",word)
            label = line.strip().split(' ')[-1]
            print("label > ",label)
            # here we dont do "DOCSTART" check
            if len(line.strip())==0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l,w))
                words=[]
                labels = []
            words.append(word)
            labels.append(label)
        rf.close()
        return lines

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test"
        )


    def get_labels(self):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        # return ["[PAD]","B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
        return ['[PAD]','B-AE','B-Age','B-Concomitant','B-Concomitant-Disease','B-Country-Of-Incidence','B-Date-of-Initial-Receipt','B-Dosage','B-Drug','B-Family-History','B-Gender','B-Indication','B-Lab-Class','B-Lab-Data','B-Lab-Test-Value','B-Medical-Drug-History','B-Medical-History','B-ROA','B-Weight','B-lab-date','I-AE','I-Age','I-Concomitant','I-Concomitant-Disease','I-Country-Of-Incidence','I-Date-of-Initial-Receipt','I-Dosage','I-Drug','I-Family-History','I-Indication','I-Lab-Class','I-Lab-Data','I-Lab-Test-Value','I-Medical-Drug-History','I-Medical-History','I-ROA','I-Weight','I-lab-date','O','X', '[CLS]', '[SEP]','I-Gender']

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, doc_stride ,mode):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i
    with open(FLAGS.middle_output+"/label2id.pkl",'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")
    print(">>>>>>>>>>>>>>>>", len(tokens))
    max_tokens_for_doc = max_seq_length - 1
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(tokens):
        length = len(tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(tokens):
            break
        start_offset += min(length, doc_stride)

    feature_list, ntokens_list, label_ids_list = [], [], []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        # token_to_orig_map = {}
        token_is_max_context = {}
        ntokens = []
        label_ids = []
        segment_ids = []
        ntokens.append("[CLS]")
        label_ids.append(label_map["[CLS]"])
        segment_ids.append(0)
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            # token_to_orig_map[len(ntokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[split_token_index] = is_max_context
            ntokens.append(tokens[split_token_index])
            label_ids.append(label_map[labels[split_token_index]])
            segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("[PAD]")
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(ntokens) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            mask=mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            token_is_max_context=token_is_max_context,
            doc_span_index=doc_span_index,
            example_index=ex_index
        )
        feature_list.append(feature)
        ntokens_list.append(ntokens)
        label_ids_list.append(label_ids)

    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        # print(tokens)
        logging.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    return feature_list,ntokens_list,label_ids_list

def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file,doc_stride,mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    batch_index = []
    feature_list_total = []
    # print(len(examples))
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature_list,ntokens_list,label_ids_list = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, doc_stride , mode)
        feature_list_total.extend(feature_list)
        for feature, ntokens, label_ids in zip(feature_list,ntokens_list,label_ids_list):
            batch_tokens.extend(ntokens)
            batch_labels.extend(label_ids)
            batch_index.extend([ex_index]*len(ntokens))
            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["mask"] = create_int_feature(feature.mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens,batch_labels,batch_index,feature_list_total

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn

# all above are related to data preprocess
# Following i about the model

def hidden2tag(hiddenlayer,numclass):
    linear = tf.keras.layers.Dense(numclass,activation=None)
    return linear(hiddenlayer)

def crf_loss(logits,labels,mask,num_labels,mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    #TODO
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transition",
            shape=[num_labels,num_labels],
            initializer=tf.contrib.layers.xavier_initializer()
        )

    log_likelihood,transition = tf.contrib.crf.crf_log_likelihood(logits,labels,transition_params =trans ,sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)

    return loss,transition

def softmax_layer(logits,labels,num_labels,mask):
    logits = tf.reshape(logits, [-1, num_labels])
    labels = tf.reshape(labels, [-1])
    mask = tf.cast(mask,dtype=tf.float32)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=one_hot_labels)
    loss *= tf.reshape(mask, [-1])
    loss = tf.reduce_sum(loss)
    total_size = tf.reduce_sum(mask)
    total_size += 1e-12 # to avoid division by 0 for all-0 weights
    loss /= total_size
    # predict not mask we could filtered it in the prediction part.
    probabilities = tf.math.softmax(logits, axis=-1)
    predict = tf.math.argmax(probabilities, axis=-1)
    return loss, predict


def create_model(bert_config, is_training, input_ids, mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config = bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()
    #output_layer shape is
    if is_training:
        output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
    logits = hidden2tag(output_layer,num_labels)
    # TODO test shape
    logits = tf.reshape(logits,[-1,FLAGS.max_seq_length,num_labels])
    if FLAGS.crf:
        mask2len = tf.reduce_sum(mask,axis=1)
        loss, trans = crf_loss(logits,labels,mask,num_labels,mask2len)
        predict,viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
        return (loss, logits,predict)

    else:
        loss,predict  = softmax_layer(logits, labels, num_labels, mask)

        return (loss, logits, predict)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        mask = features["mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if FLAGS.crf:
            (total_loss, logits,predicts) = create_model(bert_config, is_training, input_ids,
                                                         mask, segment_ids, label_ids,num_labels,
                                                         use_one_hot_embeddings)

        else:
            (total_loss, logits, predicts) = create_model(bert_config, is_training, input_ids,
                                                          mask, segment_ids, label_ids,num_labels,
                                                          use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        initialized_variable_names=None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:

                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                         init_string)



        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(label_ids, logits,num_labels,mask):
                predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels-1, weights=mask)
                return {
                    "confusion_matrix":cm
                }
                #
            eval_metrics = (metric_fn, [label_ids, logits, num_labels, mask])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=predicts, scaffold_fn=scaffold_fn
            )
        return output_spec

    return model_fn


def _write_base(batch_tokens,id2label,prediction,batch_labels,wf,i):
    token = batch_tokens[i]
    predict = id2label[prediction]
    true_l = id2label[batch_labels[i]]
    if token!="[PAD]" and token!="[CLS]" and true_l!="X":
        #
        if predict=="X" and not predict.startswith("##"):
            predict="O"
        line = "{}\t{}\t{}\n".format(token,true_l,predict)
        wf.write(line)

def Writer(output_predict_file,result,batch_tokens,batch_labels,batch_indexes,id2label,all_features,max_seq_length,doc_stride):
    # max_seq_length = 512
    # doc_stride = 128
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)


    with open(output_predict_file,'w') as wf:

        if  FLAGS.crf:
            predictions  = []
            # current_features = None
            for m,pred in enumerate(result):
                predictions.extend(pred)
                # current_features = all_features[m]

            batch_labels_copy = []
            batch_tokens_copy = []
            predictions_copy = []
            batch_indexes_copy = []

            # remove [CLS]
            for i, prediction in enumerate(predictions):
                if prediction!=40:
                    predictions_copy.append(prediction)
                    batch_labels_copy.append(batch_labels[i])
                    batch_tokens_copy.append(batch_tokens[i])
                    batch_indexes_copy.append(batch_indexes[i])

            batch_labels_copy_1 = []
            batch_tokens_copy_1 = []
            predictions_copy_1 = []
            batch_indexes_copy_1 = []

            # to concatenate various doc_spans of a single example in one
            e_index = 0
            for i, prediction in enumerate(predictions_copy):
                ex_index = batch_indexes_copy[i]
                if i>0 and batch_indexes_copy[i] != batch_indexes_copy[i-1]:
                    e_index = 0

                current_features = example_index_to_features[ex_index]
                for feature in current_features:
                    if not feature.token_is_max_context.get(e_index, False):
                        # e_index += 1
                        continue
                    else:
                        batch_labels_copy_1.append(batch_labels_copy[feature.doc_span_index*(max_seq_length-1) + (e_index-(feature.doc_span_index*doc_stride))])
                        batch_tokens_copy_1.append(batch_tokens_copy[feature.doc_span_index*(max_seq_length-1) + (e_index-(feature.doc_span_index*doc_stride))])
                        predictions_copy_1.append(predictions_copy[feature.doc_span_index*(max_seq_length-1) + (e_index-(feature.doc_span_index*doc_stride))])
                        batch_indexes_copy_1.append(batch_indexes_copy[feature.doc_span_index*(max_seq_length-1) + (e_index-(feature.doc_span_index*doc_stride))])
                        e_index += 1
                        break

            batch_labels_copy_2 = []
            batch_tokens_copy_2 = []
            predictions_copy_2 = []
            batch_indexes_copy_2 = []

            # to concatenate tokens with '##' (reverse tokenization)
            j = 0
            for i, (il,it) in enumerate(zip(batch_labels_copy_1, batch_tokens_copy_1)):
                if i==0 or (il!=39 and it!='[CLS]'):
                    batch_labels_copy_2.append(il)
                    batch_tokens_copy_2.append(it)
                    predictions_copy_2.append(predictions_copy_1[i])
                    batch_indexes_copy_2.append(batch_indexes_copy_1[i])
                    j+=1
                else:
                    if it == '[CLS]':
                        continue
                    if it.startswith('##'):
                        it = it[2:]
                    batch_tokens_copy_2[j-1]+=it
            
            

            # write examples to file
            for i,prediction in enumerate(predictions_copy_2):
                if i>0 and batch_indexes_copy_2[i] != batch_indexes_copy_2[i-1]:
                    wf.write('\n')
                    _write_base(batch_tokens_copy_2, id2label, prediction, batch_labels_copy_2, wf, i)
                else:
                    _write_base(batch_tokens_copy_2,id2label,prediction,batch_labels_copy_2,wf,i)

        else:
            for i,prediction in enumerate(result):
                _write_base(batch_tokens,id2label,prediction,batch_labels,wf,i)



def main(_):
    logging.set_verbosity(logging.INFO)
    processors = {"ner": NerProcessor}
    #if not FLAGS.do_train and not FLAGS.do_eval:
    #    raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)


    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        _,_,_,_ = filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file, FLAGS.doc_stride)
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", FLAGS.train_batch_size)
        logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        _,_,_,_ = filed_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file, FLAGS.doc_stride)

        logging.info("***** Running evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        # if FLAGS.use_tpu:
        #     eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)
        # eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        result = estimator.evaluate(input_fn=eval_input_fn)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with open(output_eval_file,"w") as wf:
            logging.info("***** Eval results *****")
            confusion_matrix = result["confusion_matrix"]
            p,r,f = metrics.calculate(confusion_matrix,len(label_list)-1)
            logging.info("***********************************************")
            logging.info("********************P = %s*********************",  str(p))
            logging.info("********************R = %s*********************",  str(r))
            logging.info("********************F = %s*********************",  str(f))
            logging.info("***********************************************")


    if FLAGS.do_predict:

        with open(FLAGS.middle_output+'/label2id.pkl', 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}

        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        batch_tokens,batch_labels,batch_indexes,all_features = filed_based_convert_examples_to_features(predict_examples, label_list,
                                                                             FLAGS.max_seq_length, tokenizer,
                                                                             predict_file, FLAGS.doc_stride)

        logging.info("***** Running prediction*****")
        logging.info("  Num examples = %d", len(predict_examples))
        logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")
        #here if the tag is "X" means it belong to its before token, here for convenient evaluate use
        # conlleval.pl we  discarding it directly
        Writer(output_predict_file,result,batch_tokens,batch_labels,batch_indexes,id2label,all_features,FLAGS.max_seq_length,FLAGS.doc_stride)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

