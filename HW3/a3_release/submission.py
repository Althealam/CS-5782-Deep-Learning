import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    """
    h: Height of the patch.
    w: Width of the patch.
    dim: The dimension of the model embeddings.
    """

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def triplet_loss(queries, keys, margin=1.0):
    """
    Inputs:
    queries (b x D): A batch of training examples.
    keys (b x D): A batch of training examples. The ith example in keys is a positive
                  example for the ith example in queries.
    margin: The margin, m, in the equation above.

    Outputs:
    The triplet loss, calculated as described above.
    """
    b = queries.shape[0]  # batch size
    device = queries.device
    n = b * 2  # total number of examples

    # TODO1: Implement triplet loss
    # Hint: Whenever you create a new tensor, make sure to send it to the same
    #       location (device) your model and data are on.
    # Hint: How might you use matrices/matrix operations to keep track of distances between
    #       positive and negative pairs? (looking ahead to the instructions in part 1.2 maybe be useful)
    #################
    # 1. L2 Normlization
    queries_norm = F.normalize(queries, p=2, dim=1)
    keys_norm = F.normalize(keys, p=2, dim=1)

    # 2. Similarity Matrix
    # or: similarity_matrix = queries_norm@keys_norm.T, shape = (b, b), which means sim(qi, kj)
    similarity_matrix = torch.matmul(queries_norm, keys_norm.T)

    # 3. Positive Similarity
    positive_sim = similarity_matrix.diag() # sim(q_i, k_i), shape = (b, )
    positive_sim = positive_sim.unsqueeze(1) # shape = (b, 1)

    # 4. triplet loss
    loss_matrix = similarity_matrix-positive_sim+margin # (b, b) - (b, 1)
    loss_matrix = F.relu(loss_matrix)

    # 5. remove diagonal
    mask = ~torch.eye(b, dtype=torch.bool, device = queries.device)
    loss_matrix = loss_matrix[mask]

    return loss_matrix.mean()


def nt_xent_loss(queries, keys, temperature=0.1):
    """
    Inputs:
    queries (b x D): A batch of training examples.
    keys (b x D): A batch of training examples. The ith example in keys is a
                  differently-augmented view of the ith example in queries.
    temperature: The temperature, tau, in the equation above.

    Outputs:
    The SimCLR loss, calculated as described above.

    We do two random data augmentation in the SimCLR
    - Queries identifies the representation after the first augmentation, and batch size is B
    - Keys identifies the representation after the second augmentation, and batch size is B
    - queries[i] and keys[i] are the different variants for the same image 
    """
    b, device = queries.shape[0], queries.device
    n = b * 2

    # TODO2: Implement the SimCLR loss
    # Hint: Whenever you create a new tensor, make sure to send it to the same
    #       location (device) your model and data are on.
    # Hint: Which loss function does the first equation in step 3 remind you of?
    #################
    # 1. L2 Normalization
    queries_norm = F.normalize(queries, p=2, dim=1)
    keys_norm = F.normalize(keys, p=2, dim=1)

    # 2. Put all the samples together
    # we have 2b training samples (as the question said), and each sample's dimension is D
    representation = torch.cat([queries_norm, keys_norm], dim=0) # [2b, D]

    # 3. Similarity Matrix 
    similarity_matrix = torch.matmul(representation, representation.T) # [2b, 2b]
    similarity_matrix = similarity_matrix/temperature

    # 4. Masked self-similarity for the negative samples pair
    mask = torch.eye(n, dtype=torch.bool).to(device) # In
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    # 5. find positive pair labels
    # each image has one positive sample and 2B-2 negative samples
    labels = torch.arange(n).to(device) # Example: [1, 2, 3, 4, 5, 6]
    # [q1, q2, q3, k1, k2, k3] -> [k1, k2, k3, q1, q2, q3]
    # it means that the key for q1 is k1
    labels[:b]+=b 
    labels[b:]-=b

    # 6. cross entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


class ViT(nn.Module):
    def __init__(self, d_model, num_layers, patch_size=4, img_side_length=32, p=0.05):
        """
        Inputs:
        d_model: The dimension of the encoder embeddings.
        num_layers: Number of encoder layers.
        patch_size: Side length of the square image patches.
        img_side_length: The height and width of the images.
        p: Dropout probability.
        """
        super(ViT, self).__init__()

        d_ff = 4 * d_model
        num_heads = d_model // 32

        # TODO3: define the ViT
        #################
        self.d_model = d_model
        self.num_layers = num_layers
        self.p = p

        # number of patches in the image (sequence length/number of tokens)
        self.num_patches = (img_side_length//patch_size)**2
        # size of each patch
        self.patch_size = patch_size
        # patch dim (RGB: 3 channels, and each channel have patch_size*pathc_size)
        # raw input dimension (vocab_size)
        self.patch_dim = 3*self.patch_size*self.patch_size

        # 1. to_patch_embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. pos_embedding
        h_w = img_side_length//patch_size # height and width of the patch
        pos_data = posemb_sincos_2d(h_w, h_w, d_model)
        self.register_buffer("pos_embedding", pos_data.unsqueeze(0))

        # 3. encoder
        self.dropout = nn.Dropout(p)
        encoder_layers = nn.TransformerEncoderLayer(
          d_model = d_model, 
          nhead = num_heads, 
          dim_feedforward=d_ff, 
          dropout = p,
          batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_ln = nn.LayerNorm(d_model)

        # 4. projection head (only be used when return_embedding is False)
        self.projection_head = nn.Sequential(
          nn.Linear(d_model, d_model),
          nn.SiLU(),
          nn.Linear(d_model, d_model)
        )
        ################

    def forward(self, x, return_embedding=False):

        ## TODO4: Write the forward pass for the ViT
        #################
        b, c, h, w = x.shape

        # ==== 1. to_patch_embedding ====
        x = self.to_patch_embedding(x)


        # ==== 2. pos_embedding ====
        # pos_embedding: (num_patches, d_model)
        x = x+self.pos_embedding # (b, num_patches, d_model)

        # ==== 3. encoder ====
        x = self.dropout(x)
        x = self.encoder(x)

        # global average pooling
        # (b, num_patches, d_model) ==> (b, d_model)
        embedding = x.mean(dim=1)
        x = self.output_ln(embedding)

        if return_embedding:
          return embedding
        
        # ==== 4. projection_head ====
        output = self.projection_head(embedding)

        return output
        #################
