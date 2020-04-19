#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : datasets.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/02/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

import copy
import json
import os.path as osp

import imageio
import nltk
import numpy as np
import torch
from PIL import Image

import jacinle.io as io
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from nscl.datasets.common.filterable import (
    FilterableDatasetUnwrapped,
    FilterableDatasetView,
)
from nscl.datasets.common.program_translator import (
    gen_vocab,
    nsclseq_to_nsclqsseq,
    nsclseq_to_nscltree,
    nscltree_to_nsclqstree,
)
from nscl.datasets.common.vocab import Vocab
from nscl.datasets.definition import gdef

logger = get_logger(__file__)

__all__ = [
    "NSCLDataset",
    "NSCLMultiviewDataset",
    "ConceptRetrievalDataset",
    "ConceptQuantizationDataset",
    "create_prototype_dataset",
]


class ProtoDataset(FilterableDatasetUnwrapped):
    def __init__(self, scenes_path, image_root, depth_root):

        self.scenes_path = scenes_path
        self.image_root = image_root
        self.depth_root = depth_root

        self.scenes = np.load(scenes_path, allow_pickle=True)
        self.scenes = {key: self.scenes[key] for key in self.scenes.files}
        self.im_id_to_scene_id = {
            self.scenes["image_index"][idx]: idx
            for idx in range(len(self.scenes["image_index"]))
        }
        self.metainfo_cache = dict()

    def get_metainfo(self, index):
        # index = self.data_idxs[index]
        if index not in self.metainfo_cache:
            question = {"image_index": index}
            scene = {
                key: self.scenes[key][self.im_id_to_scene_id[question["image_index"]]]
                for key in self.scenes
            }
            question["scene"] = scene
            self.metainfo_cache[index] = question

        return self.metainfo_cache[index]

    def __len__(self):
        return len(self.im_id_to_scene_id)

    def __getitem__(self, index):
        # index = index % 200
        metainfo = GView(self.get_metainfo(index))
        metainfo.view_id = 1
        feed_dict = GView()
        feed_dict.scene = metainfo.scene
        feed_dict.update(gdef.annotate_objects(metainfo.scene))
        # feed_dict.objects_raw = feed_dict.objects.copy()
        feed_dict.update(gdef.annotate_scene(metainfo.scene))
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.scene["image_filename"]
        # image_base_name = feed_dict.image_filename.split(".")[0]
        image_base_name = f"{feed_dict.image_filename}/{VIEWS[metainfo.view_id]}"

        feed_dict.image = Image.open(
            osp.join(self.image_root, f"{image_base_name}.png")
        ).convert("RGB")
        feed_dict.image = self.image_transform(feed_dict.image).unsqueeze(0)
        feed_dict.depth = imageio.imread(
            osp.join(self.depth_root, f"{image_base_name}.exr"), format="EXR-FI"
        )[:, :, 0]
        feed_dict.depth = self.depth_transform(feed_dict.depth)
        feed_dict.attribute_name = "shape"
        feed_dict.concept_name = metainfo.scene["shape"][0]
        # program

        # Scene
        feed_dict.bboxes = torch.tensor(feed_dict.scene["obj_bboxes"][0]).reshape(-1, 9)
        # feed_dict.bboxes_len = torch.tensor(feed_dict.bboxes.size(0))
        feed_dict.pix_T_cam = torch.tensor(metainfo.scene["pix_T_cams"]).float()
        feed_dict.origin_T_cam = torch.tensor(
            metainfo.scene["origin_T_cams"][metainfo.view_id]
        ).float()
        return feed_dict.toDict()

    # def __len__(self):
    #     return self.nRecords

    def depth_transform(self, depth):
        transform = T.Compose(
            [
                # T.Resize(self.image_size),
                T.ToTensor(),
                T.Lambda(lambda x: x * 100.0),
            ]
        )

        return transform(depth)

    def image_transform(self, image):
        image_transform = T.Compose(
            [
                # T.Resize(self.image_size),
                T.ToTensor(),
                T.Lambda(lambda x: x - 0.5)
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return image_transform(image)


def create_prototype_dataset(prototype_path):
    # Create Prototype dataset
    dataset_path = prototype_path
    processed_data = osp.join(dataset_path, "npzs")
    scenes_path = osp.join(processed_data, "scenes.npz")
    image_root = osp.join(dataset_path, "images")
    depth_root = osp.join(dataset_path, "depth")
    prototpye_dataset = ProtoDataset(scenes_path, image_root, depth_root)

    # Create one-shot dataset
    # dataset_path = one_shot_path
    # processed_data = os.path.join(dataset_path, "npzs")
    # questions_path = os.path.join(processed_data, "questions.npz")
    # scenes_path = os.path.join(processed_data, "scenes.npz")
    # image_root = os.path.join(dataset_path, "images")
    # depth_root = os.path.join(dataset_path, "depth")
    # # split = os.path.join(dataset_path, "splits", 'one_shot.json')
    # # shuffle = hyp.shuffles[set_name]
    # # collate_guide = create_collate_guide()
    # # with open(split) as f:
    # #     scene_idxs = set(json.load(f))
    # full_dataset = FilterableDataset(
    #     NSCLMultiviewDatasetUnwrapped(
    #         None,
    #         questions_path,
    #         scenes_path,
    #         processed_data,
    #         image_root,
    #         depth_root,
    #         (hyp.image_size),
    #     )
    # )
    # full_dataset = full_dataset.filter(
    #     lambda metainfo: metainfo["image_index"] in scene_idxs,
    #     "data_regime_filter",
    # )
    return prototpye_dataset  # , full_dataset


class NSCLDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(
        self,
        scenes_json,
        questions_json,
        image_root,
        image_transform,
        depth_root,
        vocab_json,
        question_transform=None,
        incl_scene=True,
    ):
        super().__init__()

        self.scenes_json = scenes_json
        self.questions_json = questions_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.depth_root = depth_root
        self.vocab_json = vocab_json
        self.question_transform = question_transform

        self.incl_scene = incl_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)["scenes"]
        if isinstance(self.questions_json, (tuple, list)):
            self.questions = list()
            for filename in self.questions_json:
                logger.info('Loading questions from: "{}".'.format(filename))
                self.questions.extend(io.load_json(filename)["questions"])
        else:
            logger.info('Loading questions from: "{}".'.format(self.questions_json))
            self.questions = io.load_json(self.questions_json)["questions"]

        if self.vocab_json is not None:
            logger.info('Loading vocab from: "{}".'.format(self.vocab_json))
            self.vocab = Vocab.from_json(self.vocab_json)
        else:
            logger.info("Building the vocab.")
            self.vocab = gen_vocab(self)
            vocab_file = questions_json.split("/")
            vocab_file[-1] = "vocab.json"
            vocab_file = "/".join(vocab_file)
            self.vocab.dump_json(vocab_file)

    def _get_metainfo(self, index):
        question = gdef.translate_question(self.questions[index])
        scene = gdef.translate_scene(self.scenes[question["image_index"]])
        question["scene"] = scene

        question["image_index"] = question["image_index"]
        question["image_filename"] = gdef.get_image_filename(scene)
        question["question_index"] = index
        question["question_tokenized"] = nltk.word_tokenize(question["question"])

        # program section
        has_program = False
        if "program_nsclseq" in question:
            question["program_raw"] = question["program_nsclseq"]
            question["program_seq"] = question["program_nsclseq"]
            has_program = True
        elif "program" in question:
            question["program_raw"] = question["program"]
            question["program_seq"] = gdef.program_to_nsclseq(
                question["program"], question
            )
            has_program = True

        if has_program:
            question["program_tree"] = nsclseq_to_nscltree(question["program_seq"])
            question["program_qsseq"] = nsclseq_to_nsclqsseq(question["program_seq"])
            question["program_qstree"] = nscltree_to_nsclqstree(
                question["program_tree"]
            )
            question["question_type"] = question["program_seq"][-1]["op"]
        else:
            question["question_type"] = None

        return question

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # metainfo annotations
        if self.incl_scene:
            feed_dict.scene = metainfo.scene
            feed_dict.update(gdef.annotate_objects(metainfo.scene))
            if "objects" in feed_dict:
                # NB(Jiayuan Mao): in some datasets, object information might be completely unavailable.
                feed_dict.objects_raw = feed_dict.objects.copy()
            feed_dict.update(gdef.annotate_scene(metainfo.scene))

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None and feed_dict.image_filename is not None:
            feed_dict.image = Image.open(
                osp.join(self.image_root, feed_dict.image_filename)
            ).convert("RGB")
            feed_dict.image, feed_dict.objects = self.image_transform(
                feed_dict.image, feed_dict.objects
            )
        if self.depth_root is not None and feed_dict.image_filename is not None:
            depth_filename = feed_dict.image_filename.split(".")[0] + ".exr"
            feed_dict.depth = torch.tensor(
                load_depth(osp.join(self.depth_root, depth_filename))
            )

        # program
        if "program_raw" in metainfo:
            feed_dict.program_raw = metainfo.program_raw
            feed_dict.program_seq = metainfo.program_seq
            feed_dict.program_tree = metainfo.program_tree
            feed_dict.program_qsseq = metainfo.program_qsseq
            feed_dict.program_qstree = metainfo.program_qstree
        feed_dict.question_type = metainfo.question_type

        # question
        feed_dict.question_index = metainfo.question_index
        feed_dict.question_raw = metainfo.question
        feed_dict.question_raw_tokenized = metainfo.question_tokenized
        feed_dict.question_metainfo = gdef.annotate_question_metainfo(metainfo)
        feed_dict.question = metainfo.question_tokenized
        feed_dict.answer = gdef.canonize_answer(metainfo.answer, metainfo.question_type)
        feed_dict.update(gdef.annotate_question(metainfo))

        if self.question_transform is not None:
            self.question_transform(feed_dict)
        feed_dict.question = np.array(
            self.vocab.map_sequence(feed_dict.question), dtype="int64"
        )

        return feed_dict.raw()

    def __len__(self):
        return len(self.questions)


def load_depth(depth_file):
    depth = np.array(imageio.imread(depth_file, format="EXR-FI"))[:, :, 0]
    depth = depth.astype(np.float32) * 2.0
    return depth


VIEWS = ["319_20", "319_33", "319_46", "319_59"]


def copydict(d):
    return json.loads(json.dumps(d))


class NSCLMultiviewDatasetUnwrapped(NSCLDatasetUnwrapped):
    def _get_metainfo(self, index):
        qid = index // len(VIEWS)
        view_id = index % len(VIEWS)
        question = gdef.translate_question(self.questions[qid])
        question = copydict(question)
        question["index"] = index
        scene = gdef.translate_scene(self.scenes[question["image_index"]])
        scene = copydict(scene)
        scene["objects_detection"] = scene["objects_detection"][VIEWS[view_id]]
        for obj in scene["objects"]:
            obj["mask"] = obj["mask"][VIEWS[view_id]]
        question["scene"] = scene
        # question["image_index"] = question["image_index"]
        question["view_id"] = view_id
        question["image_filename"] = osp.join(
            gdef.get_image_filename(scene), VIEWS[view_id] + ".png"
        )
        question["question_index"] = qid
        question["question_tokenized"] = nltk.word_tokenize(question["question"])
        # program section
        has_program = False
        if "program_nsclseq" in question:
            question["program_raw"] = question["program_nsclseq"]
            question["program_seq"] = question["program_nsclseq"]
            has_program = True
        elif "program" in question:
            question["program_raw"] = question["program"]
            question["program_seq"] = gdef.program_to_nsclseq(
                question["program"], question
            )
            has_program = True

        if has_program:
            question["program_tree"] = nsclseq_to_nscltree(question["program_seq"])
            question["program_qsseq"] = nsclseq_to_nsclqsseq(question["program_seq"])
            question["program_qstree"] = nscltree_to_nsclqstree(
                question["program_tree"]
            )
            question["question_type"] = question["program_seq"][-1]["op"]
        else:
            question["question_type"] = None
        return question

    def __len__(self):
        return len(VIEWS) * len(self.questions)


class NSCLDatsetFilterableView(FilterableDatasetView):
    def filter_program_size_raw(self, max_length):
        def filt(question):
            return len(question["program"]) <= max_length

        return self.filter(filt, "filter-program-size-clevr[{}]".format(max_length))

    def filter_scene_size(self, max_scene_size):
        def filt(question):
            return len(question["scene"]["objects"]) <= max_scene_size

        return self.filter(filt, "filter-scene-size[{}]".format(max_scene_size))

    def filter_question_type(self, *, allowed=None, disallowed=None):
        def filt(question):
            if allowed is not None:
                return question["question_type"] in allowed
            elif disallowed is not None:
                return question["question_type"] not in disallowed

        if allowed is not None:
            return self.filter(
                filt,
                "filter-question-type[allowed={{{{}}}]".format(",".join(list(allowed))),
            )
        elif disallowed is not None:
            return self.filter(
                filt,
                "filter-question-type[disallowed={{{}}}]".format(
                    ",".join(list(disallowed))
                ),
            )
        else:
            raise ValueError("Must provide either allowed={...} or disallowed={...}.")

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            "scene": "skip",
            "objects_raw": "skip",
            "objects": "concat",
            "image_index": "skip",
            "image_filename": "skip",
            "question_index": "skip",
            "question_metainfo": "skip",
            "question_raw": "skip",
            "question_raw_tokenized": "skip",
            "question": "pad",
            "program_raw": "skip",
            "program_seq": "skip",
            "program_tree": "skip",
            "program_qsseq": "skip",
            "program_qstree": "skip",
            "question_type": "skip",
            "answer": "skip",
        }

        gdef.update_collate_guide(collate_guide)

        return JacDataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=nr_workers,
            pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide),
        )


def NSCLDataset(*args, **kwargs):
    return NSCLDatsetFilterableView(NSCLDatasetUnwrapped(*args, **kwargs))


def NSCLMultiviewDataset(*args, **kwargs):
    return NSCLDatsetFilterableView(NSCLMultiviewDatasetUnwrapped(*args, **kwargs))


class ConceptRetrievalDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(
        self, program, scenes_json, image_root, image_transform, incl_scene=True
    ):
        super().__init__()

        self.program = program
        self.scenes_json = scenes_json
        self.image_root = image_root
        self.image_transform = image_transform

        self.incl_scene = incl_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)["scenes"]

    def _get_metainfo(self, index):
        question = dict()
        question["scene"] = gdef.translate_scene(self.scenes[index])

        question["image_index"] = index
        question["image_filename"] = gdef.get_image_filename(question["scene"])

        # program section
        question["program_raw"] = self.program
        question["program_seq"] = self.program
        question["program_tree"] = nsclseq_to_nscltree(question["program_seq"])
        question["program_qsseq"] = nsclseq_to_nsclqsseq(question["program_seq"])
        question["program_qstree"] = nscltree_to_nsclqstree(question["program_tree"])
        question["question_type"] = question["program_seq"][-1]["op"]

        return question

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # scene annotations
        if self.incl_scene:
            feed_dict.scene = metainfo.scene
            feed_dict.update(gdef.annotate_objects(metainfo.scene))
            feed_dict.objects_raw = feed_dict.objects.copy()
            feed_dict.update(gdef.annotate_scene(metainfo.scene))

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None:
            feed_dict.image = Image.open(
                osp.join(self.image_root, feed_dict.image_filename)
            ).convert("RGB")
            feed_dict.image, feed_dict.objects = self.image_transform(
                feed_dict.image, feed_dict.objects
            )

        # program
        feed_dict.program_raw = metainfo.program_raw
        feed_dict.program_seq = metainfo.program_seq
        feed_dict.program_tree = metainfo.program_tree
        feed_dict.program_qsseq = metainfo.program_qsseq
        feed_dict.program_qstree = metainfo.program_qstree
        feed_dict.question_type = metainfo.question_type

        # question
        feed_dict.answer = True

        return feed_dict.raw()

    def __len__(self):
        return len(self.scenes)


class ConceptRetrievalDatsetFilterableView(FilterableDatasetView):
    def filter_scene_size(self, max_scene_size):
        def filt(question):
            return len(question["scene"]["objects"]) <= max_scene_size

        return self.filter(filt, "filter-scene-size[{}]".format(max_scene_size))

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            "scene": "skip",
            "objects_raw": "skip",
            "objects": "concat",
            "image_index": "skip",
            "image_filename": "skip",
            "program_raw": "skip",
            "program_seq": "skip",
            "program_tree": "skip",
            "program_qsseq": "skip",
            "program_qstree": "skip",
            "question_type": "skip",
            "answer": "skip",
        }

        gdef.update_collate_guide(collate_guide)

        return JacDataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=nr_workers,
            pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide),
        )


def ConceptRetrievalDataset(*args, **kwargs):
    return ConceptRetrievalDatsetFilterableView(
        ConceptRetrievalDatasetUnwrapped(*args, **kwargs)
    )


class ConceptQuantizationDatasetUnwrapped(FilterableDatasetUnwrapped):
    def __init__(self, scenes_json, image_root, image_transform, incl_scene=True):
        super().__init__()

        self.scenes_json = scenes_json
        self.image_root = image_root
        self.image_transform = image_transform
        self.incl_scene = incl_scene

        logger.info('Loading scenes from: "{}".'.format(self.scenes_json))
        self.scenes = io.load_json(self.scenes_json)["scenes"]

    def _get_metainfo(self, index):
        question = dict()
        question["scene"] = gdef.translate_scene(self.scenes[index])

        question["image_index"] = index
        question["image_filename"] = gdef.get_image_filename(question["scene"])

        return question

    def __getitem__(self, index):
        metainfo = GView(self.get_metainfo(index))
        feed_dict = GView()

        # scene annotations
        if self.incl_scene:
            feed_dict.scene = metainfo.scene
            feed_dict.update(gdef.annotate_objects(metainfo.scene))
            feed_dict.objects_raw = feed_dict.objects.copy()
            feed_dict.update(gdef.annotate_scene(metainfo.scene))

        # image
        feed_dict.image_index = metainfo.image_index
        feed_dict.image_filename = metainfo.image_filename
        if self.image_root is not None:
            feed_dict.image = Image.open(
                osp.join(self.image_root, feed_dict.image_filename)
            ).convert("RGB")
            feed_dict.image, feed_dict.objects = self.image_transform(
                feed_dict.image, feed_dict.objects
            )

        return feed_dict.raw()

    def __len__(self):
        return len(self.scenes)


class ConceptQuantizationDatasetFilterableView(FilterableDatasetView):
    def filter_scene_size(self, max_scene_size):
        def filt(question):
            return len(question["scene"]["objects"]) <= max_scene_size

        return self.filter(filt, "filter-scene-size[{}]".format(max_scene_size))

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            "scene": "skip",
            "objects_raw": "skip",
            "objects": "concat",
            "image_index": "skip",
            "image_filename": "skip",
        }

        gdef.update_collate_guide(collate_guide)

        return JacDataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=nr_workers,
            pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide),
        )


def ConceptQuantizationDataset(*args, **kwargs):
    return ConceptQuantizationDatasetFilterableView(
        ConceptQuantizationDatasetUnwrapped(*args, **kwargs)
    )
