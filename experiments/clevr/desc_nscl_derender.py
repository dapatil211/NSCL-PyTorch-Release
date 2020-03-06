#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : desc_nscl_derender.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/10/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

"""
Derendering model for the Neuro-Symbolic Concept Learner.

Unlike the model in NS-VQA, the model receives only ground-truth programs and needs to execute the program
to get the supervision for the VSE modules. This model tests the implementation of the differentiable
(or the so-called quasi-symbolic) reasoning process.

Note that, in order to train this model, one must use the curriculum learning.
"""

from jacinle.utils.container import GView
from nscl.models.reasoning_v1 import make_reasoning_v1_configs, ReasoningV1Model
from nscl.models.utils import canonize_monitors, update_from_loss_module
import torch.nn.functional as F
import torch
from ipdb import set_trace as st

configs = make_reasoning_v1_configs()
configs.model.vse_known_belong = False
configs.train.scene_add_supervision = True
configs.train.qa_add_supervision = True
configs.train.full_scene_supervision = True


class Model(ReasoningV1Model):
    def __init__(self, vocab):
        super().__init__(vocab, configs)

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)
        monitors, outputs = {}, {}

        depth = feed_dict.depth
        depth = F.tanh(depth) * 0.5
        inp = torch.cat((feed_dict.image, depth.unsqueeze(1)), axis=1)

        f_scene = self.resnet(inp)
        f_sng = self.scene_graph(f_scene, feed_dict.objects, feed_dict.objects_length)

        programs = feed_dict.program_qsseq
        programs, buffers, answers = self.reasoning(f_sng, programs, fd=feed_dict)
        outputs["buffers"] = buffers
        outputs["answer"] = answers

        update_from_loss_module(
            monitors,
            outputs,
            self.scene_loss(
                feed_dict,
                f_sng,
                self.reasoning.embedding_attribute,
                self.reasoning.embedding_relation,
            ),
        )
        update_from_loss_module(monitors, outputs, self.qa_loss(feed_dict, answers))

        canonize_monitors(monitors)

        if self.training:
            loss = monitors["loss/qa"]
            if configs.train.full_scene_supervision:
                loss = loss + monitors["loss/scene"]
            return loss, monitors, outputs
        else:
            outputs["monitors"] = monitors
            outputs["buffers"] = buffers
            return outputs


def make_model(args, vocab):
    return Model(vocab)
