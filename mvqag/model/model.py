import torch
import torch.nn as nn
import torchvision as tv  # eval is using this

from mvqag.model.sysu_vgg_hgap import vgg16HGap
from mvqag.model.q_model import *


__all__ = ['VQANet', 'SANNet', 'MixPool2d', 'AdaptiveMixPool']


class MixPool2d(nn.Module):
    def __init__(self,
                 kernel_size=2,
                 stride=2,
                 padding=0,
                 dilation=1,
                 ceil_mode=False) -> None:
        """Combination of MaxPool and AvgPool"""
        super().__init__()
        self.max = nn.MaxPool2d(kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    ceil_mode=ceil_mode)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    ceil_mode=ceil_mode)
        # Trainable weighing parameters
        self.ga = nn.Parameter(torch.tensor(2.),requires_grad=True)
        self.gm = nn.Parameter(torch.tensor(2.),requires_grad=True)

    def forward(self, x):
        return (self.avg(x)/self.ga)+(self.max(x)/self.gm)


class AdaptiveMixPool(nn.Module):
    def __init__(self, output_size) -> None:
        """Combination of AdaptiveMaxPool2d and AdaptiveAvgPool2d"""
        super().__init__()
        self.max = nn.AdaptiveMaxPool2d(output_size=output_size)
        self.avg = nn.AdaptiveAvgPool2d(output_size=output_size)
        # Trainable weighing parameters
        self.ga = nn.Parameter(torch.tensor(2.),requires_grad=True)
        self.gm = nn.Parameter(torch.tensor(2.),requires_grad=True)

    def forward(self, x):
        return (self.avg(x)/self.ga)+(self.max(x)/self.gm)


class VQANet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 vnet_name: str = 'vgg16',
                 qnet_name: str = 'GRU',
                 qnet_nlayers: int = 2,
                 vocab_dim: int = None,
                 emb_dim: int = None,
                 hid_dim: int = 1024,
                 bidirect: bool = False,
                 vdp: float = 0.2,
                 qdp: float = 0.2) -> None:
        super().__init__()
        # Visual model
        if vnet_name.lower() == 'vgg16':
            v_dim = 4096
            self.vnet = tv.models.vgg16(pretrained=True)
            self.vnet.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, v_dim),
                nn.Dropout(p=vdp)
            )
        elif vnet_name.lower() == 'vgg16mixpool':
            v_dim = 4096
            self.vnet = tv.models.vgg16(pretrained=True)
            for i, layer in enumerate(self.vnet.features):
                if isinstance(layer, nn.MaxPool2d):
                    self.vnet.features[i] = MixPool2d()
            self.vnet.avgpool = AdaptiveMixPool(output_size=(7,7))
            self.vnet.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, v_dim),
                nn.Dropout(p=vdp)
            )
        else:
            raise NotImplementedError

        # Question model
        self.w_emb = WordEmbedding(vocab_dim, emb_dim, dropout=qdp)
        if qnet_name.lower() in ['gru', 'lstm']:
            self.q_emb = QuestionEmbedding(
                in_dim=emb_dim,
                num_hid=hid_dim,
                nlayers=qnet_nlayers,
                bidirect=bidirect,
                dropout=qdp,
                rnn_type=qnet_name)
            if bidirect:
                hid_dim *= 2
        self.qnet = FCNet([hid_dim, v_dim])

        # Classifier
        self.classifier = SimpleClassifier(v_dim, v_dim, n_classes, vdp)

    def forward(self, V, Q):
        # Visual
        vout = self.vnet(V)
        # Questions
        w_emb = self.w_emb(Q['input_ids'])
        # print(f"w_emb = {w_emb.shape}")
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        # print(f"q_emb = {q_emb.shape}")
        qout = self.qnet(q_emb)
        out = vout * qout  # Elementwise multiplication
        out = self.classifier(out)
        return out


class Attention(nn.Module):
    def __init__(self, d: int, k: int = 256, dp: float = 0.) -> None:
        super().__init__()
        self.v_i = nn.Sequential(
            nn.Linear(d, k, bias=False),
            nn.Dropout(p=dp)
        )  # Image features
        self.v_q = nn.Sequential(
            nn.Linear(d, k),
            nn.Dropout(p=dp)
        )  # Question features
        self.p_i = nn.Sequential(
            nn.Linear(k, 1),
            nn.Dropout(p=dp)
        )  # probability
        self.tanh = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, vI, vQ):
        # vI = (bs, 196, 512), vQ = (bs, 512)
        vQ = vQ.view(vQ.shape[0], 1, -1)
        # vQ = (bs, 1, 512)
        vI, vQ = self.v_i(vI), self.v_q(vQ)
        # vI = (bs, d, k), vQ = (bs, d, k)
        ha = self.tanh(vI + vQ)
        # (bs, 196, 512)
        pi = self.softmax(self.p_i(ha).squeeze(-1))
        # (bs, 196)
        vI_attn = (pi.unsqueeze(2) * vI).sum(dim=1)
        return vI_attn + vQ.squeeze(1)  # u = ̃vI + vQ


class SANNet(nn.Module):
    def __init__(self,
                 n_classes: int,
                 vnet_name: str = 'vgg16',
                 qnet_name: str = 'GRU',
                 qnet_nlayers: int = 2,
                 vocab_dim: int = None,
                 emb_dim: int = None,
                 vdp: float = 0.2,
                 qdp: float = 0.2) -> None:
        """Stacked Attention Network"""
        super().__init__()
        # Visual model
        self.n_v_feats, k = 2048, 1024
        if vnet_name.lower() == 'vgg16':
            self.vnet = tv.models.vgg16(pretrained=True)
            self.vnet = self.vnet.features  # Only take features
            self.vnet[30] = nn.Identity()  # Remove maxpool
            self.vnetfc = nn.Linear(self.n_v_feats, self.n_v_feats)
            self.vnetfc_act = nn.LeakyReLU()
        elif vnet_name.lower() == 'vgg16mixpool':
            self.vnet = tv.models.vgg16(pretrained=True)
            for i, layer in enumerate(self.vnet.features):
                if isinstance(layer, nn.MaxPool2d):
                    self.vnet.features[i] = MixPool2d()
            self.vnet = self.vnet.features  # Only take features
            self.vnet[30] = nn.Identity()  # Remove maxpool
            self.vnetfc = nn.Linear(self.n_v_feats, self.n_v_feats)
            self.vnetfc_act = nn.LeakyReLU()
        else:
            raise NotImplementedError

        # Question model
        self.w_emb = WordEmbedding(vocab_dim, emb_dim, dropout=qdp)
        if qnet_name.lower() in ['gru', 'lstm']:
            self.q_emb = QuestionEmbedding(
                in_dim=emb_dim,
                num_hid=self.n_v_feats,
                nlayers=qnet_nlayers,
                bidirect=False,
                dropout=qdp,
                rnn_type=qnet_name.upper())
            
        # Attention
        self.attn = Attention(d=self.n_v_feats, k=k)
        # Classifier
        self.classifier = nn.Linear(k, n_classes)

    def forward(self, V, Q):
        # Visual
        V = self.vnet(V)
        V = V.view(V.shape[0], self.n_v_feats, -1).permute(0, 2, 1)
        # Questions
        w_emb = self.w_emb(Q['input_ids'])
        vQ = self.q_emb(w_emb)  # [batch, q_dim]
        # print(f"vQ = {vQ.shape}")
        # Attention
        vI = self.vnetfc_act(self.vnetfc(V))
        # print(f"vI = {vI.shape}")
        attn = self.attn(vI, vQ)
        # print(f"Attn = {attn.shape}")
        out = self.classifier(attn)
        return out

# if __name__ == '__main__':
#     import torch
#     from mvqag.utils import load_yaml
#     from mvqag import CNF_PATH
#     CNF = load_yaml(CNF_PATH)
#     net = SANNet(
#         n_classes=330,
#         vnet_name='vgg16',
#         qnet_name='lstm',
#         vocab_dim=31,
#         emb_dim=128,
#     )
#     img = torch.rand((4, 3, 224, 224))
#     Q = torch.randint(0, 12, (4, 12))
#     out = net(img, {'input_ids': Q})
#     print(out.shape)

#     # net = Attention(d=512, k=256)
#     # Vi = torch.rand((4, 196, 512))
#     # Vq = torch.rand((4, 512))
#     # net(Vi, Vq)
