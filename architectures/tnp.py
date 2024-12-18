import torch.nn as nn
import torch


class TNP(nn.Module):
    """Transformer neural process.

    Args:
        dim_x (int): Input dimensionality.
        dim_y (int): Output dimensionality.
        dim_embedding (int): Dimensionality of the embedding.
        num_layers (int): Number of layers in the encoder.
        embedding_depth (int): Depth of the encoder.
        num_heads (int): Number of heads.
        dim_feedforward (int): Width of the hidden layers

    """
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        num_layers: int,
        embedding_depth: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super(TNP, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(dim_embedding, num_heads, dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def build_mlp(self, dim_in, dim_hid, dim_out, depth):
        modules = [nn.Linear(dim_in, dim_hid), nn.ReLU(True)]
        for _ in range(depth - 2):
            modules.append(nn.Linear(dim_hid, dim_hid))
            modules.append(nn.ReLU(True))
        modules.append(nn.Linear(dim_hid, dim_out))
        return nn.Sequential(*modules)

    def create_mask(self, xc, xt):
        num_xc = xc.shape[1]
        num_xt = xt.shape[1]
        num_all = num_xc + num_xt

        mask = torch.zeros(num_all, num_all).fill_(float('-inf'))
        mask = mask.type(xc.dtype)
        mask[:, :num_xc] = 0.0

        return mask, num_xt

    def _forward(self, xc, yc, xt):
        # (batch_size, set_size, feature_dim)
        xc = torch.t(xc)
        yc = torch.t(yc)
        xt = torch.t(xt)
        x_y_context = torch.concat((xc, yc), dim=-1)
        x_t_test = torch.concat((xt, torch.zeros(xt.shape[0], xt.shape[1], 1).type(xt.dtype)), dim=-1)
        inp = torch.concat((x_y_context, x_t_test), dim=1)
        mask, num_tar = self.create_mask(xc, xt)
        embeddings = self.embedder(inp)
        out = self.encoder(embeddings, mask=mask)
        out = torch.t(out[:, -num_tar:])

        return out

