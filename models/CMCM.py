# self supervised multimodal multi-task learning network
import math
import os
import sys
import collections
from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.Textmodel import Language_model

__all__ = ['CMCM']

class CMCM(nn.Module):
    def __init__(self, args):
        super(CMCM, self).__init__()
        # text enocding
        self.LLM = Language_model(args)
        # strategy and head switches
        self.fusion_strategy = getattr(args, 'fusion_strategy', 'text_guided')
        self.output_head = getattr(args, 'output_head', 'llm')
        self.num_classes = getattr(args, 'num_classes', 2)
        # audio and video enocding
        text_in, audio_in, video_in = args.feature_dims[:]
        text_len, audio_len, video_len = args.seq_lens[:]

        self.audio_LSTM = TVA_LSTM(audio_in, args.a_lstm_hidden_size, num_layers=args.a_lstm_layers, dropout=args.a_lstm_dropout)
        self.video_LSTM = TVA_LSTM(video_in, args.v_lstm_hidden_size, num_layers=args.v_lstm_layers, dropout=args.v_lstm_dropout)

        self.text_guide_mixer = Text_guide_mixer()
        # add tri-modal mixer (audio, video, pooled text)
        self.tri_mixer = TriModalMixer()
        #low_rank_fusion
        fusion_input_size = 256
        self.mutli_scale_fusion = mutli_scale_fusion(input_size=fusion_input_size, output_size= text_in, pseudo_tokens= args.pseudo_tokens)
        # heads for mlp output mode
        if self.output_head == 'mlp':
            # 修改点 1: MLP 的输入维度改为 LLM 的隐藏层维度 (text_in, 即 2048)
            # 之前是 256 (fusion_vec维度)，现在我们要用 LLM 输出的特征
            mlp_input_dim = text_in 
            if getattr(args, 'train_mode', 'regression') == 'regression':
                self.reg_head = nn.Sequential(
                    nn.LayerNorm(mlp_input_dim),
                    nn.Linear(mlp_input_dim, 1)
                )
            else:
                self.cls_head = nn.Sequential(
                    nn.LayerNorm(mlp_input_dim),
                    nn.Linear(mlp_input_dim, self.num_classes)
                )
    
    def forward(self, labels, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        text = self.LLM.text_embedding(text[:,0,:].long()) # text 存的是原本的token ids，所以需要转换为 embedding

        video_h = self.video_LSTM(video, video_len)
        audio_h = self.audio_LSTM(audio, audio_len)

        # choose fusion strategy
        if self.fusion_strategy == 'tri_fusion':
            fusion_vec = self.tri_mixer(audio_h, video_h, text)  # [B,256]
        else:
            fusion_vec = self.text_guide_mixer(audio_h, video_h, text)  # [B,256]

        fusion_h= self.mutli_scale_fusion(fusion_vec)  # [B, pseudo_tokens, hidden]

        # 修改点 2: 统一构建 LLM 输入，无论 output_head 是什么
        if self.fusion_strategy == 'tri_fusion':
            LLM_input = fusion_h
        else:
            LLM_input = torch.cat([fusion_h, text], dim=1)

        # output path
        if self.output_head == 'llm':
            LLM_output = self.LLM(LLM_input, labels)

            res = {
                'Loss': LLM_output.loss,
                'Feature_a': audio_h,
                'Feature_v': video_h,
                'Feature_f': fusion_h,
            }
            return res
        else:
            # direct head WITH LLM processing
            # 修改点 3: 将数据喂给 LLM，提取最后一层的 Hidden State
            # output_hidden_states=True 确保返回隐藏层状态
            outputs = self.LLM.model(inputs_embeds=LLM_input, output_hidden_states=True)
            
            # 取最后一层 ([-1]) 的 最后一个 Token ([:, -1, :]) 的特征
            # 维度: [Batch, Hidden_Size(2048)]
            llm_feature = outputs.hidden_states[-1][:, -1, :]

            if getattr(self, 'reg_head', None) is not None:
                preds = self.reg_head(llm_feature).squeeze(-1)
                loss = F.l1_loss(preds.view(-1), labels.view(-1))
            else:
                logits = self.cls_head(llm_feature)
                loss = F.cross_entropy(logits, labels.long().view(-1))
                preds = logits
            res = {
                'Loss': loss,
                'Feature_a': audio_h,
                'Feature_v': video_h,
                'Feature_f': fusion_h,
            }
            return res

    def generate(self, text, audio, video):
        audio, audio_len = audio
        video, video_len = video
        text, text_len = text
        text = self.LLM.text_embedding(text[:,0,:].long())

        audio_h = self.audio_LSTM(audio, audio_len)
        video_h = self.video_LSTM(video, video_len)

        # choose fusion strategy
        if self.fusion_strategy == 'tri_fusion':
            fusion_vec = self.tri_mixer(audio_h, video_h, text)
        else:
            fusion_vec = self.text_guide_mixer(audio_h, video_h, text)

        fusion_h = self.mutli_scale_fusion(fusion_vec)

        # 修改点 4: Generate 阶段同样构建 LLM 输入
        if self.fusion_strategy == 'tri_fusion':
            LLM_input = fusion_h
        else:
            LLM_input = torch.cat([fusion_h, text], dim=1)

        if self.output_head == 'llm':
            LLM_output = self.LLM.generate(LLM_input)
            return LLM_output
        else:
            # mlp head predictions using LLM features
            # 推理时同样过一遍 LLM 获取特征
            outputs = self.LLM.model(inputs_embeds=LLM_input, output_hidden_states=True)
            llm_feature = outputs.hidden_states[-1][:, -1, :]

            if getattr(self, 'reg_head', None) is not None:
                preds = self.reg_head(llm_feature).squeeze(-1)  # [B]
                return preds
            else:
                logits = self.cls_head(llm_feature)  # [B,C]
                pred_labels = torch.argmax(logits, dim=-1).tolist()
                return pred_labels


class TVA_LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(TVA_LSTM, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 256)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False) #这里把length.to cpu是因为pytorch版本问题
        # _, (final_states, _) = self.rnn(packed_sequence)
        # h = self.dropout(final_states[-1])
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        h = self.linear(h)
        return h

class Text_guide_mixer(nn.Module):
    def __init__(self):
        super(Text_guide_mixer, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_mlp = nn.Linear(2048, 256)
    def forward(self, audio, video, text):
        text_GAP = self.GAP(text.permute(0, 2, 1)).squeeze()
        text_knowledge = self.text_mlp(text_GAP)

        audio_mixed = torch.mul(audio, text_knowledge)
        video_mixed = torch.mul(video, text_knowledge)

        fusion = audio_mixed + video_mixed

        return fusion

class TriModalMixer(nn.Module):
    """
    Three mode mixer: simple splicing and fusion of audio, video and text features.
    Compared to Text_guide_mixer, the three modalities are treated equally here.
    """
    def __init__(self):
        super(TriModalMixer, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.text_proj = nn.Linear(2048, 256)
        self.fuse = nn.Sequential(
            nn.Linear(256*3, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
    def forward(self, audio, video, text):
        # audio, video: [B,256]; text: [B, L, 2048]
        text_vec = self.GAP(text.permute(0, 2, 1)).squeeze()
        text_vec = self.text_proj(text_vec)
        x = torch.cat([audio, video, text_vec], dim=1)
        return self.fuse(x)

class mutli_scale_fusion(nn.Module):
    """
    Multi scale fusion: expands the fused single feature vector into multiple 'Pseudo Tokens',
    In order to feed LLM. It extracts multi-scale features through "expert" networks with different scaling ratios.
    """
    def __init__(self, input_size, output_size, pseudo_tokens = 4):
        super(mutli_scale_fusion, self).__init__()
        multi_scale_hidden = 256
        self.scale1 = nn.Sequential(
            nn.Linear(input_size, output_size // 8),
            nn.GELU(),
            nn.Linear(output_size // 8, multi_scale_hidden)
        )
        self.scale2 = nn.Sequential(
            nn.Linear(input_size, output_size // 32),
            nn.GELU(),
            nn.Linear(output_size // 32, multi_scale_hidden)
        )
        self.scale3 = nn.Sequential(
            nn.Linear(input_size, output_size // 16),
            nn.GELU(),
            nn.Linear(output_size // 16, multi_scale_hidden)
        )

        self.integrating = Integrating(scales = 3)
        self.multi_scale_projector =  nn.Linear(multi_scale_hidden, output_size)
        self.projector = nn.Linear(1, pseudo_tokens)

    def forward(self,x):
        # 增加样本复制，将单一样本复制一份,避免最后一个batch只有一个数据时的报错
        if x.dim() == 1:
            x = x.unsqueeze(0)
        #compute different scale experts outputs
        scale1 = self.scale1(x)
        scale2 = self.scale2(x)
        scale3 = self.scale3(x)


        # Calculate the expert outputs
        multi_scale_stack = torch.stack([scale1, scale2, scale3], dim=2)
        multi_scale_integrating =  self.integrating(multi_scale_stack)

        multi_scale = self.multi_scale_projector(multi_scale_integrating)
        output = self.projector(multi_scale.unsqueeze(2))
        return output.permute(0, 2, 1)  #[batch,seq_len,hidden_siez]

# Define the gating model
class Integrating(nn.Module):
    def __init__(self,  scales):
        super(Integrating, self).__init__()

    # Layers
        self.Integrating_layer = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(1, scales), stride=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.Integrating_layer(x)
        x = x.squeeze((1, 3))
        return x
