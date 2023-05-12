import torch


def moment_loss(output, target, log=False):
    loss_mean = torch.mean((target - output[:, 0].reshape(-1, 1)) ** 2)
    loss_variance = torch.mean(
        ((target - output[:, 0].reshape(-1, 1)) ** 2 - output[:, 1].reshape(-1, 1) ** 2)
        ** 2
    )
    if log:
        return 0.5 * (torch.log(loss_mean) + torch.log(loss_variance))
    print("Loss mean = ", loss_mean)
    print("Loss variance = ", loss_variance)
    return 0.5 * (loss_mean + loss_variance)
