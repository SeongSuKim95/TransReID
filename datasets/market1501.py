# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class Market1501(BaseImageDataset): # BaseImageDataset inherit
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)  # root_dir + / + market1501
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train') # root_dir + / + market1501 + /bounding_box_train
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run() # osp.exist 로 directory 가 존재하는지 확인(dataset이 올바르게 구성되었는지 확인)
        self.pid_begin = pid_begin
        # 각 data를 [('image_dir',pid,camid,1),...] 형태로 반환
        train = self._process_dir(self.train_dir, relabel=True) 
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery) # BaseImageDataset method

        self.train = train #self.train에 train data list 할당
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train) # Method from BaseImageDataset
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg')) # path내의 모든 image를 glob하여 모든 directory를 list 형식으로
        # len(img_paths) = 12936
        pattern = re.compile(r'([-\d]+)_c(\d)') # 숫자_c숫자 형식 
        pid_container = set() #set으로 person id set 초기화
        for img_path in sorted(img_paths): #img_paths 정렬 후 각 img_path들에
            pid, _ = map(int, pattern.search(img_path).groups()) # 두 패턴 ([-\d+]),(\d)를 int로 mapping후 반환
            # img_path = '/mnt/hdd_data/Dataset/market1501/bounding_box_train/0002_c1s1_000451_03.jpg'
            # pid = 2, _ = 1
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}  # 751 개의 pid에 numbering
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))
            #[('/mnt/hdd_data/Datase...451_03.jpg', 0, 0, 1),....]
            #[('image_dir',pid,camid,1),...]
        return dataset
