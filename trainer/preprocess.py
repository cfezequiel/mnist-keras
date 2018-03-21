"""Preprocess data using DataFlow."""

import argparse
from datetime import datetime
import sys

import apache_beam as beam
from tensorflow_transform.beam import impl as tft_beam

from keras.datasets import mnist


def _get_job_name():
  return "flowers-pipeline-{}".format(datetime.now().strftime("%Y%m%d%H%M%S"))


def _configure_pipeline(p, pargs):
  _ = (p |
       'ReadInput' >> '')


def _parse_arguments(argv):
  """Parses command line arguments."""

  parser = argparse.ArgumentParser(
      description='Runs preprocessing on the MNIST dataset.')

  parser.add_argument(
      '--output_dir', required=True,
      help='GCS or local directory to store the outputs.')

  parser.add_argument(
      '--job_name', default=_get_job_name(),
      help='Unique name for the preprocessing job.')

  parser.add_argument(
      '--num_workers', default=_DEFAULT_NUM_WORKERS, type=int,
      help='Number of workers to process the job.')

  parser.add_argument(
      '--cloud', default=False, action='store_true',
      help='Run preprocessing job in DataFlow')


def _get_pipeline_config(args):
  if args.cloud:
    runner = 'DataflowRunner'
    options = dict(job_name=args.job_name, max_num_workers=args.num_workers)
  else:
    runner = 'DirectRunner'
    options = None

  return dict(runner=runner, options=options)


def main():
  args = _parse_arguments(sys.argv[1:])
  config = _get_pipeline_config(args)
  options = beam.pipeline.PipelineOptions(flags=[], **config['options'])
  with beam.Pipeline(config['runner'], options=options) as p:
    with tft_beam.Context(temp_dir=args.temp_dir):
      _configure_pipeline(p, args)


if __name__ == '__main__':
  main()