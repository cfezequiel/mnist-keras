"""Cloud ML Engine deployment script."""

from __future__ import print_function

import argparse
import sys

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import model


DEFAULT_STEPS = 10


def run_experiment(hparams):
  # Build training spec
  train_input_fn = model.make_train_input_fn(hparams.shuffle)
  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=hparams.train_steps)

  # Build evaluation/serving spec
  eval_input_fn = model.make_eval_input_fn()
  exporter = tf.estimator.FinalExporter('mnist', model.json_serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn, steps=hparams.eval_steps, exporters=[exporter])

  # Build estimator config
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=hparams.job_dir)

  # Train and evaluate model
  estimator = model.build_estimator(run_config,
                                    model.INPUT_SHAPE,
                                    model.NUM_CLASSES)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def parse_arguments(argv):
  parser = argparse.ArgumentParser(description='Parse MNIST ML task arguments.')
  parser.add_argument('--train-steps', type=int, default=DEFAULT_STEPS,
                      help='Number of training steps.')
  parser.add_argument('--eval-steps', type=int, default=DEFAULT_STEPS,
                      help='Number of evaluation steps.')
  parser.add_argument('--job-dir', required=True,
                      help='GCS location to write checkpoints and export model')
  return parser.parse_args()


if __name__ == '__main__':
  args = parse_arguments(sys.argv[1:])
  hparams = hparam.HParams(**args.__dict__)
  run_experiment(hparams)
