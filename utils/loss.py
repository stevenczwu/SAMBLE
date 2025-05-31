import torch


def feature_transform_regularizer_loss(trans):
    # trans: (B,C,C)
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))
    )
    return loss


def token_orthognonal_loss(tokens):
    # tokens.shape=(B,1,N,C)=(16,1,1024,6)
    tokens = torch.squeeze(tokens, dim=1)

    token_orthognonal_loss = torch.matmul(tokens.transpose(1, 2), tokens)
    token_orthognonal_loss = token_orthognonal_loss - torch.diag_embed(
        torch.diagonal(token_orthognonal_loss, dim1=1, dim2=2)
    )  # - torch.diag(token_orthognonal_loss.diagonal())
    token_orthognonal_loss = torch.sum(token_orthognonal_loss)

    return token_orthognonal_loss
