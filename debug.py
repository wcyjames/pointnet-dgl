import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dgl
import dgl.function as fn
from dgl.geometry.pytorch import FarthestPointSampler

from pointnet2 import RelativePositionMessage, PointNetConv

# To profile speed
from pyinstrument import Profiler
profiler = Profiler()

class RPM(nn.Module):
    '''
    Compute the input feature from neighbors
    '''
    def __init__(self, n_neighbor):
        super(RPM, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, edges):
        pos = edges.src['pos'] - edges.dst['pos']
        # if 'feat' in edges.src:
        if True:
            res = torch.cat([pos, edges.src['feat']], 1)
        else:
            res = pos
        import time

        # profiler.start()
        # for i in range(50):
        #     pos = edges.src['pos'] - edges.dst['pos']
        #     if 'feat' in edges.src:
        #         res = torch.cat([pos, edges.src['feat']], 1)
        #     else:
        #         res = pos
        # profiler.stop()
        # print(profiler.output_text(unicode=True, color=True, show_all=True))
        # print(time.time() - s)
        return {'agg_feat': res}


class PNConv(nn.Module):
    '''
    Feature aggregation
    '''
    def __init__(self, sizes, batch_size):
        super(PNConv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(nn.Conv2d(sizes[i-1], sizes[i], 1))
            self.bn.append(nn.BatchNorm2d(sizes[i]))

    def forward(self, nodes):

        shape = nodes.mailbox['agg_feat'].shape
        h = nodes.mailbox['agg_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 1, 2) # torch.Size([16, 6, 512, 32])
        # for conv, bn in zip(self.conv, self.bn):
        #     h = conv(h)
        #     h = bn(h)
        #     h = F.relu(h)

        #profiler.start()
        #for i in range(50):
        #     shape = nodes.mailbox['agg_feat'].shape
        #     h = nodes.mailbox['agg_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 1, 2)
        #     for conv, bn in zip(self.conv, self.bn):
        #         h = conv(h)
        #         h = bn(h)
        #         h = F.relu(h)

        # profiler.stop()
        # print(profiler.output_text(unicode=True, color=True, show_all=True))

        # h = torch.max(h, 3)[0]
        # feat_dim = h.shape[1]
        # h = h.permute(0, 2, 1).reshape(-1, feat_dim)
        h = h.reshape(nodes.data['pos'].shape[0], -1)
        return {'new_feat': h}


# First SA
glist, _ = dgl.load_graphs("./data.bin")
g = glist[0]
n_neighbor = 32
message = RPM(n_neighbor)

mlp_sizes = [6, 64, 64, 128]
batch_size = 16
conv = PNConv(mlp_sizes, batch_size)

message = message.cuda()
conv.cuda()
g = g.to("cuda")

profiler.start()
for i in range(50):
    g.update_all(message, conv)
    # g.update_all(message, fn.mean('agg_feat', 'a'))
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))

#print(g)

# Second SA
