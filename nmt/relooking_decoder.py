from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.layers import core as layers_core

from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils

class RelookingTrainingHelper(tf.contrib.seq2seq.TrainingHelper):

	def __init__(self, )