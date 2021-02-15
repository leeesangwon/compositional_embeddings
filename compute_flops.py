from thop import profile
import torch
from g_net import *
from h_net import *
from subset_embeddings_openimage_fh import F_net
from subset_embeddings_libispeech_fg import Lstm

model = Mike_net_linear(128)
input = torch.randn(1, 128)#torch.randn(1, 199, 32)
macs, params = profile(model, inputs=(input, input, ))
print(macs, params)

