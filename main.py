# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import os
import time
from tqdm import trange

from model import GAN_g, GAN_d
from config import config as conf
from loss import infer_g_valid
from utils import crop_sub_imgs_fn, eval_H_dist, eval_tfpn, eval_dice_hard, eval_IoU

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def train_gan():
    ## create folders to save result images and trained model
    checkpoint_dir = conf.TRAIN.ckpt_dir
    tl.files.exists_or_mkdir(checkpoint_dir)
    output_dir = conf.TRAIN.output_dir
    tl.files.exists_or_mkdir(output_dir)
    samples_dir = conf.TRAIN.gan_samples_dir
    tl.files.exists_or_mkdir(samples_dir)
    logs_dir = conf.TRAIN.gan_log
    tl.files.exists_or_mkdir(logs_dir)
    EPS = 1e-12

    ## Adam
    lr_init = conf.TRAIN.lr_init * 0.1
    beta1 = conf.TRAIN.beta1
    batch_size = conf.TRAIN.batch_size
    ni = int(np.ceil(np.sqrt(batch_size)))

    # load data

    train_img_list = sorted(tl.files.load_file_list(path=conf.TRAIN.img_list_path, regx='.*.jpg', printable=False))
    train_img_list2 = sorted(tl.files.load_file_list(path=conf.TRAIN.img_list_path2, regx='.*.jpg', printable=False))
    valid_img_list = sorted(tl.files.load_file_list(path=conf.VALID.img_list_path, regx='.*.jpg', printable=False))

    train_imgs = np.expand_dims(np.array(tl.vis.read_images(train_img_list, path=conf.TRAIN.img_path, n_threads=64)), axis=3) / 255.0
    train_segs = np.expand_dims(np.array(tl.vis.read_images(train_img_list, path=conf.TRAIN.seg_path, n_threads=64)), axis=3) / 255.0
    train_imgs2 = np.expand_dims(np.array(tl.vis.read_images(train_img_list2, path=conf.TRAIN.img_path, n_threads=64)), axis=3) / 255.0
    train_segs2 = np.expand_dims(np.array(tl.vis.read_images(train_img_list2, path=conf.TRAIN.seg_path, n_threads=64)), axis=3) / 255.0
    valid_imgs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.img_path, n_threads=64)), axis=3) / 255.0
    valid_segs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.seg_path, n_threads=64)), axis=3) / 255.0

    train_data = np.concatenate((train_imgs, train_segs), axis=3)

    # vis data
    vidx = 0
    train_vis_img = train_imgs[vidx:vidx+batch_size,:,:,:]
    train_vis_seg = train_segs[vidx:vidx+batch_size,:,:,:]
    valid_vis_img = valid_imgs[vidx:vidx+batch_size,:,:,:]
    valid_vis_seg = valid_segs[vidx:vidx+batch_size,:,:,:]
    tl.vis.save_images(train_vis_img, [ni,ni], os.path.join(samples_dir, '_train_img.jpg'))
    tl.vis.save_images(train_vis_seg, [ni,ni], os.path.join(samples_dir, '_train_seg.jpg'))
    tl.vis.save_images(valid_vis_img, [ni,ni], os.path.join(samples_dir, '_valid_img.jpg'))
    tl.vis.save_images(valid_vis_seg, [ni,ni], os.path.join(samples_dir, '_valid_seg.jpg'))

    # define network

    x_m = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    y_m = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    x_n = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    y_n = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    x_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    y_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])

    output_channels = 1
    gm_tanh, gm_logit = GAN_g(x_m, output_channels)
    gn_tanh, gn_logit = GAN_g(x_n, output_channels)
    gen_loss_L1 = tf.reduce_mean(tf.abs(x_m - gm_tanh))
    v_g, v_dice = infer_g_valid(x_valid, y_valid)
    #g_vars = g_logit.all_params

    true_pos = tf.reduce_sum(y_m * gm_tanh)
    false_neg = tf.reduce_sum(y_m * (1 - gm_tanh))
    false_pos = tf.reduce_sum((1 - y_m) * gm_tanh)
    alpha = 0.9


    d_logit1_real = GAN_d(x_m, y_m)
    d_logit1_fake0 = GAN_d(x_m, gm_tanh)
    d_logit1_fake = GAN_d(x_n, gn_tanh)
    #d_vars = d_logit1_real.all_params

    lambda_adv = 0.02
    lambda_a = 0.5
    lambda_u = 1 - lambda_a

    d_l1_loss1 = tl.cost.sigmoid_cross_entropy(d_logit1_real, tf.ones_like(d_logit1_real), name="d_l1_1")
    d_l1_loss2 = lambda_a*tl.cost.sigmoid_cross_entropy(d_logit1_fake0, tf.zeros_like(d_logit1_fake0), name="d_l1_2")
    d_l1_loss3 = lambda_u*tl.cost.sigmoid_cross_entropy(d_logit1_fake, tf.zeros_like(d_logit1_fake), name="d_l1_3")
    
    d_loss = d_l1_loss1 + d_l1_loss2 + d_l1_loss3

    g_seg_loss = 1 - ((true_pos + EPS) / (true_pos + alpha*false_neg + (1 - alpha)*false_pos + EPS))
    g_gan_loss1 = lambda_adv*lambda_a * tl.cost.sigmoid_cross_entropy(d_logit1_fake0, tf.ones_like(d_logit1_fake0), name="g_gan1")
    g_gan_loss2 = lambda_adv*lambda_u * tl.cost.sigmoid_cross_entropy(d_logit1_fake, tf.ones_like(d_logit1_fake), name="g_gan2")
    g_loss = g_seg_loss + g_gan_loss1 + g_gan_loss2


    #Train Operation
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## Pretrain
    g_optim_1 = tf.train.AdamOptimizer(lr_v*10, beta1=beta1).minimize(g_loss)
    g_optim_2 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss)
    g_optim = tf.group(g_optim_1, g_optim_2)
    d_optim = tf.train.GradientDescentOptimizer(lr_v*10).minimize(d_loss)


    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.import_meta_graph('checkpoint/1.ckpt.data-00000-of-00001')
        # ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        # saver.restore(sess, ckpt)

        ## summary
        # use tensorboard --logdir="logs/log_gan"
        # http://localhost:6006/
        tb_writer = tf.summary.FileWriter(logs_dir, sess.graph)
        tf.summary.scalar('loss_d/loss_d', d_loss)
        tf.summary.scalar('loss_d/loss_d_l1r', d_l1_loss1)
        tf.summary.scalar('loss_d/loss_d_l1f', d_l1_loss2)
        tf.summary.scalar('loss_d/loss_d_l1f0', d_l1_loss3)
        tf.summary.scalar('loss_g/loss_g', g_loss)
        tf.summary.scalar('loss_g/loss_gan1', g_gan_loss1)
        tf.summary.scalar('loss_g/loss_gan2', g_gan_loss2)
        tf.summary.scalar('loss_g/loss_seg', g_seg_loss)
        tf.summary.scalar('gen_loss_L1', gen_loss_L1)
        tf.summary.scalar('learning_rate', lr_v)               #运行时显示
        tb_merge = tf.summary.merge_all()                      #以便tensorboard显示





        # datasets information
        n_epoch = conf.TRAIN.n_epoch
        lr_decay = conf.TRAIN.lr_decay
        decay_every = conf.TRAIN.decay_every
        n_step_epoch = np.int(len(train_imgs)/batch_size)
        n_step = n_epoch * n_step_epoch
        #val_step_epoch = np.int(val_fX.shape[0]/FLAGS.batch_size)

        print('\nInput Data Info:')
        print('   train_file_num:', len(train_imgs), '\tval_file_num:', len(valid_imgs))
        print('\nTrain Params Info:')
        print('   learning_rate:', lr_init)
        print('   batch_size:', batch_size)
        print('   n_epoch:', n_epoch, '\tstep in an epoch:', n_step_epoch, '\ttotal n_step:', n_step)
        print('\nBegin Training ...')

        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        max_dice = 0
        tb_train_idx = 0
        for epoch in range(n_epoch):
            ## update learning rate
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay**(epoch // decay_every)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                sess.run(tf.assign(lr_v, lr_init))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
                print(log)

            #time_start = time.time()
            t_batch = [x for x in tl.iterate.minibatches(inputs=train_data, targets=train_data, batch_size=batch_size, shuffle=True)]
            t_batch2 = [x for x in tl.iterate.minibatches(inputs=train_imgs2, targets=train_segs2, batch_size=batch_size, shuffle=True)]
            tbar = trange(min(len(t_batch), len(t_batch2)), unit='batch', ncols=100)
            train_err_d, train_err_g, train_dice, n_batch = 0, 0, 0, 0
            for i in tbar:
                ## You can also use placeholder to feed_dict in data after using
                #img_seg = np.concatenate((batch[i][0], batch[i][1]), axis=3)
                img_seg = tl.prepro.threading_data(t_batch[i][0], fn=crop_sub_imgs_fn, is_random=True)
                img_feed = np.expand_dims(img_seg[:,:,:,0], axis=3)
                seg_feed = np.expand_dims(img_seg[:,:,:,1], axis=3)
                xn_img_feed = tl.prepro.threading_data(t_batch2[i][0], fn=crop_sub_imgs_fn, is_random=False)
                yn_img_feed = tl.prepro.threading_data(t_batch2[i][1], fn=crop_sub_imgs_fn, is_random=False)

                feed_dict = {x_m: img_feed, y_m: seg_feed, x_n: xn_img_feed, y_n: yn_img_feed}

                # update D
                #sess.run(d_optim, feed_dict=feed_dict)
                _errD, _errDl11, _errDl12, _errDl13, _ = sess.run([d_loss, d_l1_loss1, d_l1_loss2, d_l1_loss3, d_optim], feed_dict=feed_dict)

                # update G
                _tbres, _dice, _errG, _errSeg, _errGAN1, _errGAN2, _ = sess.run([tb_merge, gen_loss_L1, g_loss, g_seg_loss, g_gan_loss1, g_gan_loss2, g_optim], feed_dict=feed_dict)

                train_err_g += _errG; train_err_d += _errD; train_dice += _dice; n_batch += 1
                tbar.set_description('Epoch %d/%d ### step %i' % (epoch+1, n_epoch, i))
                tbar.set_postfix(dice=train_dice/n_batch, g=train_err_g/n_batch, d=train_err_d/n_batch, g_seg=_errSeg, g_gan=_errGAN1+_errGAN2, d_11=_errDl11, d_12=_errDl12+_errDl13)

                tb_writer.add_summary(_tbres, tb_train_idx)
                tb_train_idx += 1

            if np.mod(epoch, conf.VALID.per_print) == 0:
                # vis image
                feed_dict = {x_valid: train_vis_img, y_valid: train_vis_seg}
                #feed_dict.update(v_g.all_drop)
                _output = sess.run(v_g, feed_dict=feed_dict)
                tl.vis.save_images(_output, [ni,ni], os.path.join(samples_dir, 'train_pred_{}.jpg'.format(epoch)))
                feed_dict = {x_valid: valid_vis_img, y_valid: valid_vis_seg}
                #feed_dict.update(v_g.all_drop)
                _output = sess.run(v_g, feed_dict=feed_dict)
                tl.vis.save_images(_output, [ni,ni], os.path.join(samples_dir, 'valid_pred_{}.jpg'.format(epoch)))
                print('Validation ...')
                time_start = time.time()
                val_acc, n_batch = 0, 0
                for batch in tl.iterate.minibatches(inputs=valid_imgs, targets=valid_segs, batch_size=1, shuffle=True):
                    img_feed, seg_feed = batch
                    feed_dict = {x_valid: img_feed, y_valid: seg_feed}
                    #feed_dict.update(v_g.all_drop)
                    _dice = sess.run(v_dice, feed_dict=feed_dict)
                    val_acc += _dice; n_batch += 1
                print('   Time:{}\tDice:{}'.format(time.time()-time_start, val_acc/n_batch))

                if val_acc/n_batch > max_dice:
                    max_dice = val_acc/n_batch

                print('[!] Max dice:', max_dice)
            saver = tf.train.Saver()
            saver.save(sess, "./checkpoint/1.ckpt")

def evaluate():
    ## create folders to save result images and trained model
    result_dir = conf.VALID.result_dir
    tl.files.exists_or_mkdir(result_dir)
    checkpoint_dir = conf.TRAIN.ckpt_dir
    model_path = checkpoint_dir
    # load data
    valid_img_list = sorted(tl.files.load_file_list(path=conf.VALID.img_list_path, regx='.*.jpg', printable=False))
    valid_imgs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.img_path, n_threads=1)), axis=3) / 255.0
    valid_segs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.seg_path, n_threads=1)), axis=3) / 255.0
    print("valid_img_list", valid_img_list)
    # define model
    x_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    y_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])

    gm_tanh, gm_logit = GAN_g(x_valid, 1)

    batch_size = conf.TRAIN.batch_size
    ni = int(np.ceil(np.sqrt(batch_size)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('checkpoint/1.ckpt.data-00000-of-00001')
        ckpt = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, ckpt)
    # valid
        j = 0
        pre = 0
        acc = 0
        rec = 0
        for batch in tl.iterate.minibatches(inputs=valid_imgs, targets=valid_segs, batch_size=1, shuffle=False):
            #img_feed = tl.prepro.threading_data(batch[batch][0], fn=crop_sub_imgs_fn, is_random=False)
            #seg_feed = tl.prepro.threading_data(batch[batch][1], fn=crop_sub_imgs_fn, is_random=False)
            img_feed, seg_feed = batch
            feed_dict = {x_valid: img_feed, y_valid: seg_feed}
            # 转化为numpy数组
            _out = sess.run(gm_logit, feed_dict=feed_dict)
            #feed_dict.update(gm_tanh)
            print("Finish!")
            tl.vis.save_images(_out, [ni, ni], os.path.join(result_dir, 'test{}.jpg'.format(valid_img_list[j])))
            print("list:", valid_img_list[j])
            j += 1
            #save_image(_out, 1)
            ACC, precision, recall, _ = eval_tfpn(_out, seg_feed)
            pre = pre + precision
            acc = ACC + acc
            rec = rec + recall

        print("ACC:", acc / 75)
        print("precision:", pre / 75)
        print("recall:", rec / 75)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_gan', help='train_gan, evaluate')
    args = parser.parse_args()
    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'train_gan':
        train_gan()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
