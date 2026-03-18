import os
import logging
import pickle
import json
import numpy as np
import pandas as pd
import torch
import gzip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from modelscope import AutoTokenizer, AutoModel
from operator import itemgetter
from torch.nn.utils.rnn import pad_sequence

__all__ = ['MMDataLoader']

logger = logging.getLogger('MSA')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2': self.__init_simsv2,
            'meld': self.__init_meld,
            'iemocap': self.__init_iemocap,
            'cherma': self.__init_cherma,
        }
        DATA_MAP[args.datasetName]()

    def __init_meld(self):
        data_path = os.path.join(self.args.dataPath, self.args.datasetName + '_' + self.mode + '.pkl')
        label_index_mapping = self.args.label_index_mapping
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.vision = np.array(list(map(lambda item: item['features']['video'], data))).astype(np.float32)
            self.audio = np.array(list(map(lambda item: item['features']['audio'], data))).astype(np.float32)
            self.rawText = np.array(list(map(lambda item: item['features']['text'], data)))

            self.labels = {
                'M': list(map(lambda item: label_index_mapping.get(item['label'],-1), data))
            }
            if self.args.use_PLM:
                self.text = self.PLM_tokenizer(self.rawText)

        if not self.args.need_data_aligned:
            self.audio_lengths = np.array(list(map(lambda item: item['features']['audio_len'], data)))
            self.vision_lengths = np.array(list(map(lambda item: item['features']['video_len'], data)))

    def __init_iemocap(self):
        return self.__init_meld()

    def __init_cherma(self):
        return self.__init_meld()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
            if self.args.use_PLM:
                self.text = data[self.mode]['raw_text']
                self.text = self.PLM_tokenizer(self.text)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']

        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }

        if self.args.need_label_prefix:
            labels = self.labels['M']
            label_prefix = []
            for i in range(len(labels)):
                if labels[i] < 0:
                    label_prefix.append(f'negative,{labels[i].item():.{1}f}')
                elif labels[i] > 0:
                    label_prefix.append(f'positive,{labels[i].item():.{1}f}')
                else:
                    label_prefix.append(f'neutral,{labels[i].item():.{1}f}')
            self.labels_prefix = label_prefix

        if self.args.datasetName == 'sims':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
            self.text_lengths = self.args.seq_lens[0]
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision != self.vision] = 0

        if self.args.need_normalized:
            self.__normalize()
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __init_simsv2(self):
        return self.__init_mosi()

    def __truncated(self):
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.seq_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def PLM_tokenizer(self, rawtexts):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.pretrain_LM,
            padding_side='left',
            trust_remote_code=True
        )
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        self.tokenizer.pad_token_id = self.eos_token_id
        token_list = []
        for text in rawtexts:
            text_tokenizer = self.tokenizer(text,
                                 padding='max_length',
                                 truncation=True,
                                 max_length=self.args.seq_lens[0],
                                 return_tensors='pt',
                                 add_special_tokens=False
                                )

            token_ids = text_tokenizer['input_ids'].squeeze(0)
            attn_masks = text_tokenizer['attention_mask'].squeeze(0)
            token_type_ids = [0] * len(token_ids)

            input_ids = np.expand_dims(token_ids, 1)
            input_mask = np.expand_dims(attn_masks, 1)
            segment_ids = np.expand_dims(token_type_ids, 1)

            text_pretrain = np.concatenate([input_ids, input_mask, segment_ids], axis=1).T
            token_list.append(text_pretrain)

        token_list = np.array(token_list)
        return token_list

    def __getitem__(self, index):
        if self.args.train_mode == 'regression':
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'id': self.ids[index],
                'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
                'labels_prefix': self.labels_prefix[index]
            }
        else:
            sample = {
                'raw_text': self.rawText[index],
                'text': torch.Tensor(self.text[index]),
                'audio': torch.Tensor(self.audio[index]),
                'vision': torch.Tensor(self.vision[index]),
                'index': index,
                'labels': {k: v[index] for k, v in self.labels.items()}
            }

        if not self.args.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
            sample['text_lengths'] = self.args.seq_lens[0]

        return sample


def MMDataLoader(args):
    """创建数据加载器，支持DDP模式"""
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len() 

    # 根据是否使用DDP创建不同的DataLoader
    if getattr(args, 'use_ddp', False):
        # DDP模式：使用DistributedSampler
        train_sampler = DistributedSampler(
            datasets['train'],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            drop_last=False
        )
        
        # 验证集和测试集不需要分布式采样（只在主进程评估）
        dataLoader = {
            'train': DataLoader(
                datasets['train'],
                batch_size=args.batch_size,
                sampler=train_sampler,  # 使用sampler时不能指定shuffle
                num_workers=args.num_workers,
                pin_memory=True
            ),
            'valid': DataLoader(
                datasets['valid'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True
            ),
            'test': DataLoader(
                datasets['test'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True
            )
        }
        # 保存sampler以便在训练时设置epoch
        dataLoader['train_sampler'] = train_sampler
    else:
        # 单卡模式：正常创建DataLoader
        dataLoader = {
            ds: DataLoader(
                datasets[ds],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=(ds == 'train')
            )
            for ds in datasets.keys()
        }
    
    return dataLoader