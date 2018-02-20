import os
import time
import numpy as np
import tensorflow as tf
import pickle
import yaml
from multiprocessing.pool import ThreadPool

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

class BestModelStats:
    def __init__(self,filename,**kwargs):
        self.filename = filename
        self.checkpoints = kwargs.get('checkpoints',{})
        self.point_min = tuple(kwargs.get('point_min',[-1,float("inf")]))

    def loss_min(self):
        return self.point_min[1]

    def add_checkpoint(self,step,data):
        self.checkpoints[step]=data
        assert data['loss']<self.loss_min()
        self.point_min = (step,data['loss'])

    def save(self):
        d = {'checkpoints':self.checkpoints,'point_min':list(self.point_min)}
        with open(self.filename,'w') as f:
            yaml.dump(d,f)

    @classmethod
    def load(cls,yaml_file):
        if os.path.isfile(yaml_file):
            with open(yaml_file,'r') as f:
                yaml_data = yaml.load(f)
            return cls(yaml_file,**yaml_data)
        return cls(yaml_file)

def _save_best_ckpt(self, step, loss_profile, loss_mva):
    file = '{}-best{}'
    model = self.meta['name']
    profile = file.format(model, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt:
        pickle.dump(loss_profile, profile_ckpt)
    
    ckpt = file.format(model,'')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint for best model at step {}'.format(step))
    self.saver_best.save(self.sess, ckpt)

    # save progress file that I can inspect
    self.best_stats.add_checkpoint(step,{'loss':float(loss_mva),'lr':self.FLAGS.lr})
    self.best_stats.save()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)


def train(self):
    loss_ph = self.framework.placeholders
    loss_mva = None; profile = list()
    loss_mva_longer = None

    batches = self.framework.shuffle()
    loss_op = self.framework.loss

    load_number = self.FLAGS.load
    if load_number=='best':
        load_number = self.best_stats.point_min[0]

    for i, (x_batch, datum) in enumerate(batches):
        if not i: self.say(train_stats.format(
            self.FLAGS.lr, self.FLAGS.batch,
            self.FLAGS.epoch, self.FLAGS.save
        ))

        feed_dict = {
            loss_ph[key]: datum[key] 
                for key in loss_ph }
        feed_dict[self.inp] = x_batch
        feed_dict.update(self.feed)

        fetches = [self.train_op, loss_op]

        if self.FLAGS.summary:
            fetches.append(self.summary_op)

        fetched = self.sess.run(fetches, feed_dict)
        loss = fetched[1]

        if loss_mva is None: loss_mva = loss
        loss_mva = .9 * loss_mva + .1 * loss
        step_now = load_number + i + 1
        if loss_mva_longer is None: loss_mva_longer = loss
        loss_mva_longer = .95 * loss_mva_longer + .05 * loss

        if self.FLAGS.summary:
            self.writer.add_summary(fetched[2], step_now)

        form = 'step {} - loss {} - moving ave loss {}'
        self.say(form.format(step_now, loss, loss_mva))
        profile += [(loss, loss_mva)]

        ckpt = (i+1) % (self.FLAGS.save // self.FLAGS.batch)
        args = [step_now, profile]
        if not ckpt: _save_ckpt(self, *args)
        if loss_mva_longer < self.best_stats.loss_min() and loss_mva_longer<5.0:
            _save_best_ckpt(self, step_now, profile, loss_mva_longer)

    if ckpt: _save_ckpt(self, *args)

def return_predict(self, im):
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    this_inp = np.expand_dims(im, 0)
    feed_dict = {self.inp : this_inp}

    out = self.sess.run(self.out, feed_dict)[0]
    boxes = self.framework.findboxes(out)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        })
    return boxesInfo

import math

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        this_batch = all_inps[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(self.framework.preprocess(
                os.path.join(inp_path, inp)), 0)), this_batch)

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

def return_predict_batch(self, im_list):
    assert isinstance(im_list[0], np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im_list[0].shape

    # batch size
    n_imgs = len(im_list)
    batch = min(self.FLAGS.batch, n_imgs)
    n_batch = int(math.ceil(n_imgs / batch))

    # predict in batches
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, n_imgs)

        # collect images input in the batch
        this_batch = im_list[from_idx:to_idx]
        inp_feed = pool.map(lambda inp: (
            np.expand_dims(inp,0)), this_batch)
        #im = self.framework.resize_input(im)
        
        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        out = self.sess.run(self.out, feed_dict)
        
        # Post processing
        def post_process(prediction):
            boxes = self.framework.findboxes(prediction)
            final_boxes = []
            for box in boxes:
                tmpBox = self.framework.process_box(box,h,w,self.FLAGS.threshold)
                if tmpBox is None:
                    continue
                final_boxes.append({
                    "label": tmpBox[4],
                    "confidence": tmpBox[6],
                    "topleft": {
                        "x": tmpBox[0],
                        "y": tmpBox[2]},
                    "bottomright": {
                        "x": tmpBox[1],
                        "y": tmpBox[3]}
                })
            return final_boxes
        boxes_list = pool.map(lambda prediction: post_process(prediction), out)

        return boxes_list