from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import math
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import cv2
import fnmatch
import matplotlib.pyplot as plt

from depth_from_video_in_the_wild import model

gfile = tf.gfile
MAX_TO_KEEP = 1000000  # Maximum number of checkpoints to keep.

flags.DEFINE_string('data_dir', None, 'Preprocessed data.')

flags.DEFINE_string('file_extension', 'png', 'Image data file extension.')

flags.DEFINE_float('learning_rate', 1e-4, 'Adam learning rate.')

flags.DEFINE_float('reconstr_weight', 0.85, 'Frame reconstruction loss weight.')

flags.DEFINE_float('ssim_weight', 3.0, 'SSIM loss weight.')

flags.DEFINE_float('smooth_weight', 1e-2, 'Smoothness loss weight.')

flags.DEFINE_float('depth_consistency_loss_weight', 0.01,
                   'Depth consistency loss weight')

flags.DEFINE_integer('batch_size', 1, 'The size of a sample batch')

flags.DEFINE_integer('img_height', 128, 'Input frame height.')

flags.DEFINE_integer('img_width', 416, 'Input frame width.')

flags.DEFINE_integer('queue_size', 2000,
                     'Items in queue. Use smaller number for local debugging.')

flags.DEFINE_integer('seed', 8964, 'Seed for random number generators.')

flags.DEFINE_float('weight_reg', 1e-2, 'The amount of weight regularization to '
                   'apply. This has no effect on the ResNet-based encoder '
                   'architecture.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory to save model '
                    'checkpoints.')

flags.DEFINE_integer('train_steps', int(1e6), 'Number of training steps.')

flags.DEFINE_integer('summary_freq', 100, 'Save summaries every N steps.')

flags.DEFINE_bool('debug', False, 'If true, one training step is performed and '
                  'the results are dumped to a folder for debugging.')

flags.DEFINE_string('input_file', 'train', 'Input file name')

flags.DEFINE_float('rotation_consistency_weight', 1e-3, 'Weight of rotation '
                   'cycle consistency loss.')

flags.DEFINE_float('translation_consistency_weight', 1e-2, 'Weight of '
                   'thanslation consistency loss.')

flags.DEFINE_integer('foreground_dilation', 8, 'Dilation of the foreground '
                     'mask (in pixels).')

flags.DEFINE_boolean('learn_intrinsics', True, 'Whether to learn camera '
                     'intrinsics.')

flags.DEFINE_boolean('boxify', True, 'Whether to convert segmentation masks to '
                     'bounding boxes.')

flags.DEFINE_string('imagenet_ckpt', None, 'Path to an imagenet checkpoint to '
                    'intialize from.')


FLAGS = flags.FLAGS
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('checkpoint_dir')


def load(filename):
  with gfile.Open(filename) as f:
    return np.load(io.BytesIO(f.read()))


def _print_losses(dir1):
  for f in gfile.ListDirectory(dir1):
    if 'loss' in f:
      print ('----------', f, end=' ')
      f1 = os.path.join(dir1, f)
      t1 = load(f1).astype(float)
      print (t1)


def main(_):
  print("\ntest\n")
  inference_model = model.Model(
      boxify=FLAGS.boxify,
      data_dir=FLAGS.data_dir,
      file_extension=FLAGS.file_extension,
      is_training=False,
      foreground_dilation=FLAGS.foreground_dilation,
      learn_intrinsics=FLAGS.learn_intrinsics,
      learning_rate=FLAGS.learning_rate,
      reconstr_weight=FLAGS.reconstr_weight,
      smooth_weight=FLAGS.smooth_weight,
      ssim_weight=FLAGS.ssim_weight,
      translation_consistency_weight=FLAGS.translation_consistency_weight,
      rotation_consistency_weight=FLAGS.rotation_consistency_weight,
      batch_size=FLAGS.batch_size,
      img_height=FLAGS.img_height,
      img_width=FLAGS.img_width,
      weight_reg=FLAGS.weight_reg,
      depth_consistency_loss_weight=FLAGS.depth_consistency_loss_weight,
      queue_size=FLAGS.queue_size,
      input_file=FLAGS.input_file)
  print("\ntest1\n")

  _test(inference_model, FLAGS.checkpoint_dir, FLAGS.train_steps,
         FLAGS.summary_freq)

#  if FLAGS.debug:
#    _print_losses(os.path.join(FLAGS.checkpoint_dir, 'debug'))


def _test(inference_model, checkpoint_dir, train_steps, summary_freq):
  """Runs a trainig loop."""
  saver = tf.train.Saver()

  sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0,
                           saver=None)
  config = tf.ConfigProto()
  # config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction=0.8
  with sv.managed_session(config=config) as sess:
    logging.info('Attempting to resume training from %s...', checkpoint_dir)
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    logging.info('Last checkpoint found: %s', checkpoint)
    if checkpoint:
      print("checkpoint?")
      saver.restore(sess, checkpoint)
    elif FLAGS.imagenet_ckpt:
      logging.info('Restoring pretrained weights from %s', FLAGS.imagenet_ckpt)
      print("imagenet_ckpt?")
      saver.restore(sess, FLAGS.imagenet_ckpt)

    print("\ntest2\n")

    im_files, basepath_in = collect_input_images("/work/google-research/depth_from_video_in_the_wild/input",
                                                 None, "jpg")
    print(im_files)
    print("\ntest3\n")
    output_dirs = create_output_dirs(im_files, basepath_in, "/work/google-research/depth_from_video_in_the_wild/output")
    im_batch = []
    for i in range(len(im_files)):
      if True:
        logging.info('%s of %s files processed.', i, len(im_files))

        # Read image and run inference.
        print(im_files[i])
        im = load_image(im_files[i], resize=(416, 128))
        im_batch.append(im)
        print("\ntest4\n")
        est_depth = inference_model.inference_depth(im_batch, sess)
        print("\ntest5\n")
        for j in range(len(im_batch)):
          color_map = normalize_depth_for_display(
              np.squeeze(est_depth[j]))
          print("\ntest6\n")
          visualization = np.concatenate((im_batch[j], color_map), axis=0)
          # Save raw prediction and color visualization. Extract filename
          # without extension from full path: e.g. path/to/input_dir/folder1/
          # file1.png -> file1
          k = i - len(im_batch) + 1 + j
          filename_root = os.path.splitext(os.path.basename(im_files[k]))[0]
          pref = ''
          output_raw = os.path.join(
              output_dirs[k], filename_root + pref + '.npy')
          output_vis = os.path.join(
              output_dirs[k], filename_root + pref + '.png')
          with gfile.Open(output_raw, 'wb') as f:
            np.save(f, est_depth[j])
          save_image(output_vis, visualization, "png")
        im_batch = []


def load_image(img_file, resize=None, interpolation='linear'):
  """Load image from disk. Output value range: [0,1]."""
  im_data = np.fromstring(tf.io.gfile.GFile(img_file, 'rb').read(), np.uint8)
  im = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  if resize and resize != im.shape[:2]:
    ip = cv2.INTER_LINEAR if interpolation == 'linear' else cv2.INTER_NEAREST
    im = cv2.resize(im, resize, interpolation=ip)
  return np.array(im, dtype=np.float32) / 255.0

def save_image(img_file, im, file_extension):
  """Save image from disk. Expected input value range: [0,1]."""
  im = (im * 255.0).astype(np.uint8)
  with gfile.Open(img_file, 'w') as f:
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    _, im_data = cv2.imencode('.%s' % file_extension, im)
    f.write(im_data.tostring())

def collect_input_images(input_dir, input_list_file, file_extension):
  """Collects all input images that are to be processed."""
  if input_dir is not None:
    im_files = _recursive_glob(input_dir, '*.' + file_extension)
    basepath_in = os.path.normpath(input_dir)
  elif input_list_file is not None:
    im_files = read_text_lines(input_list_file)
    basepath_in = os.path.dirname(input_list_file)
    im_files = [os.path.join(basepath_in, f) for f in im_files]
  im_files = [f for f in im_files if 'disp' not in f and '-seg' not in f and
              '-fseg' not in f and '-flip' not in f]
  return sorted(im_files), basepath_in

def read_text_lines(filepath):
  with tf.gfile.Open(filepath, 'r') as f:
    lines = f.readlines()
  lines = [l.rstrip() for l in lines]
  return lines

def _recursive_glob(treeroot, pattern):
  results = []
  for base, _, files in os.walk(treeroot):
    files = fnmatch.filter(files, pattern)
    results.extend(os.path.join(base, f) for f in files)
  return results

def normalize_depth_for_display(depth, pc=95, crop_percent=0, normalizer=None,
                                cmap='plasma'):
  """Converts a depth map to an RGB image."""
  # Convert to disparity.

  disp = 1.0 / (depth + 1e-6)
  if normalizer is not None:
    disp /= normalizer
  else:
    disp /= (np.percentile(disp, pc) + 1e-6)
  disp = np.clip(disp, 0, 1)
  # disp = gray2rgb(disp, cmap=cmap)
  disp = gray2rgb(disp, cmap='gray')
  keep_h = int(disp.shape[0] * (1 - crop_percent))
  disp = disp[:keep_h]
  return disp

def gray2rgb(im, cmap='plasma'):
  cmap = plt.get_cmap(cmap)
  result_img = cmap(im.astype(np.float32))
  if result_img.shape[2] > 3:
    result_img = np.delete(result_img, 3, 2)
  return result_img

def create_output_dirs(im_files, basepath_in, output_dir):
  """Creates required directories, and returns output dir for each file."""
  output_dirs = []
  for i in range(len(im_files)):
    relative_folder_in = os.path.relpath(
        os.path.dirname(im_files[i]), basepath_in)
    absolute_folder_out = os.path.join(output_dir, relative_folder_in)
    if not gfile.IsDirectory(absolute_folder_out):
      gfile.MakeDirs(absolute_folder_out)
    output_dirs.append(absolute_folder_out)
  return output_dirs

if __name__ == '__main__':
  app.run(main)
