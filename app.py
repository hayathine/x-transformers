# x-transformerの実行ファイル

import torch 
from x_transformers.x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 512,
    atten_layers = Decoder(
        dim = 512,
        depth = 12,
        heads = 8,
    )
).cuda()

# randint(0, 256, (1, 1024))で、0から255までの整数をランダムに1024個生成
x = torch.randint(0, 256, (1, 1024)).cuda()

model(x)