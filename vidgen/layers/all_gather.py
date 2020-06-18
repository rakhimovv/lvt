import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Function
from torch.distributed import ReduceOp

"""
Based on:
1. https://github.com/ag14774/diffdist
2. https://discuss.pytorch.org/t/how-to-preserve-backward-grad-fn-after-distributed-operations/49343/6
"""

def all_gather(gather_list,
               tensor,
               group=dist.group.WORLD,
               next_backprop=None,
               inplace=True):
    return AllGather(group, next_backprop, inplace)(gather_list, tensor)


class AllGather(nn.Module):
    def __init__(self,
                 group=dist.group.WORLD,
                 next_backprop=None,
                 inplace=True):
        super(AllGather, self).__init__()
        self.group = group
        self.next_backprop = next_backprop
        self.inplace = inplace

        self.consume = None
        if self.next_backprop is not None:
            self.consume = ConsumeVariable()

    def forward(self, gather_list, tensor):
        if self.consume:
            tensor, = self.consume(self.next_backprop, tensor)
        return list(
            AllGatherFunc.apply(tensor, self.group, self.inplace,
                                *gather_list))


class ConsumeVariable(nn.Module):
    def __init__(self, set_ones_grad=True):
        """
        If set_ones_grad=True then the gradient w.r.t tensor_to_consume
        is set to 1 during backprop. Otherwise, it is set to 0.
        """
        super(ConsumeVariable, self).__init__()
        self.set_ones_grad = set_ones_grad

    def forward(self, tensor_to_consume, *tensors_to_return):
        tensors_to_return = ConsumeVariableFunc.apply(
            tensor_to_consume, self.set_ones_grad, *tensors_to_return)
        return tensors_to_return


class ConsumeVariableFunc(Function):
    @staticmethod
    def forward(ctx, tensor_to_consume, set_ones_grad, *tensors_to_return):
        ctx.save_for_backward(tensor_to_consume)
        ctx.set_ones_grad = set_ones_grad
        return tensors_to_return

    @staticmethod
    def backward(ctx, *grad_outputs):
        tensor_to_consume, = ctx.saved_tensors
        if ctx.set_ones_grad:
            fake_grad = torch.ones_like(tensor_to_consume)
        else:
            fake_grad = torch.zeros_like(tensor_to_consume)

        return (fake_grad, None) + grad_outputs


class AllGatherFunc(Function):
    @staticmethod
    def forward(ctx, tensor, group, inplace, *gather_list):
        ctx.save_for_backward(tensor)
        ctx.group = group
        gather_list = list(gather_list)
        if not inplace:
            gather_list = [torch.zeros_like(g) for g in gather_list]
        dist.all_gather(gather_list, tensor, group)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        reduce_scatter(grad_out, list(grads), group=ctx.group)
        return (grad_out, None, None) + grads


def reduce_scatter(tensor,
                   tensor_list,
                   op=ReduceOp.SUM,
                   group=dist.group.WORLD,
                   async_op=False):
    rank = dist.get_rank(group)
    if tensor is None:
        tensor = tensor_list[rank]
    if tensor.dim() == 0:
        tensor = tensor.view(-1)
    tensor[:] = tensor_list[rank]
    ops = []
    for i in range(dist.get_world_size(group)):
        if i == rank:
            tmp = dist.reduce(tensor, rank, op, group, async_op=True)
        else:
            tmp = dist.reduce(tensor_list[i], i, op, group, async_op=True)
        ops.append(tmp)

    oplist = AsyncOpList(ops)
    if async_op:
        return oplist
    else:
        oplist.wait()


class AsyncOpList(object):
    def __init__(self, ops):
        self.ops = ops

    def wait(self):
        for op in self.ops:
            op.wait()

    def is_completed(self):
        for op in self.ops:
            if not op.is_completed():
                return False
        return True
