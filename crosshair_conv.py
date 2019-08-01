import math
import numpy as np

def compute_out_dimension(In_Dim, Fil_Dim, P, S):
    return math.floor((In_Dim + 2 * P - Fil_Dim) / S) + 1


def conv2d_forward(X, F, P, S):
    """
    Performs a 2D convolution on the 4D input tensor X.

    Given 4D input X (NHWC format) and 4D filter F (FCHW format),
    compute the convolution of these two tensors given a stride S
    and using padding P.

    Arguments:
        X (4D tensor-like): input tensor in NHWC format.
        F (4D tensor-like): filter tensor in FCHW format.
        P (int): padding used.
        S (int): padding used.

    Returns:
        4D tensor-like: output tensor in NHWC format.
    """
    N, In_H, In_W, In_C = X.shape
    Out_C, In_C, Fil_H, Fil_W = F.shape

    Out_H = compute_out_dimension(In_H, Fil_H, P, S)
    Out_W = compute_out_dimension(In_W, Fil_W, P, S)

    padded_input = np.pad(X, pad_width=((0, 0), (P, P), (P, P), (0, 0)),
                          mode='constant', constant_values=0)

    out = np.zeros((N, Out_H, Out_W, Out_C))

    for b in range(N):
        for f in range(Out_C):
            for i in range(Out_H):
                for j in range(Out_W):
                    for c in range(In_C):
                        for f_i in range(Fil_H):
                            for f_j in range(Fil_W):
                                out[b, i, j, f] += padded_input[b, i + f_i, j + f_j, c] * F[f, c, f_i, f_j]

    return out


def dispatch(X, F, P, S, TL, BR):
    """
    Dispatches a convolution on a slice of the input tensor.

    Given part of the cross-hair convolution to perform, perform a convolution
    on the slice of the input X given by the coordinates top_left_x,
    top_left_y, bot_right_x, bot_right_y. The slice is purely in the spatial
    dimension - i.e. it is done across all tensors in the batch and all channels
    per tensor.

    Note that coordinates start from 0,0 at the top-left corner of the tensor,
    and the coordinates provided are inclusive (e.g. (1, 1) and (3, 3) as top-
    left and bottom-right respectively will mean a slice of height 3 x 3.

    Arguments:
        X (4D tensor-like): input tensor in NHWC format.
        F (4D tensor-like): filter tensor in FCHW format.
        P (int): padding used.
        S (int): stride used.
        TL (int, int): top-left position in the input tensor to start from.
        BR (int, int): bottom-right position in the input tensor to end at.

    Returns:
        4D tensor-like: output tensor in NHWC format, with N, H, and W the
                        same as in the input.
    """
    top_left_x, top_left_y = TL
    bot_right_x, bot_right_y = BR

    block = X[:, top_left_y:bot_right_y+1, top_left_x:bot_right_x+1, :]

    block_N, block_H, block_W, block_C = block.shape

    # Padding and stride both 1 to ensure same-size output.
    out_h = compute_out_dimension(block_H, F.shape[2], P, S)
    out_w = compute_out_dimension(block_W, F.shape[3], P, S)
    assert out_h == block_H, \
        'Block\'s output height {0} should be equal to input height {1}!'.format(out_h, block_H)
    assert out_w == block_W, \
        'Block\'s output width {0} should be equal to input width {1}!'.format(out_w, block_W)

    out = conv2d_forward(block, F, P, S)

    in_shape = block.shape
    out_shape = out.shape
    shape_match = out_shape == in_shape
    assert shape_match, \
        'Convolution of block not giving expected output size: input shape is {0} but output shape is {1}'.format(in_shape, out_shape)

    return out


def crosshair_conv(X, Cr_F, Cr_P, Cr_S, Cr_TL, Cr_BR, Per_F, Per_P, Per_S):
    """
    Compute the cross-hair convolution of 4D tensor X.

    Given 4D input X (NHWC format) and 4D filter Cr_F (FCHW format),
    compute the convolution of the slice of X specified by Cr_TL, Cr_BR,
    and a weaker convolution of the remaining peripheral block using Per_F.

    For simplicity, assume the cross-hair block's convolution outputs
    a block of the same size. Likewise the peripheral block. Hence the
    output has the same spatial dimensions as the input, although the depth
    may differ.

    Note that the current implementation uses zero-padding for the cross-hair,
    rather than neighbouring pixels in the peripheral block.

    +--------------+
    |              | <-- Peripheral block
    |   +------+   |
    |   |      |<--|---- Cross-hair block, with a given top-left and bottom-right.
    |   +------+   |
    |              |
    +--------------+

    Args:
        X (4D tensor-like): input tensor in NHWC format.
        Cr_F (4D tensor-like): cross-hair block filter tensor in FCHW format.
        Cr_P (int): padding used for cross-hair block convolution.
        Cr_S (int): stride for cross-hair block convolution.
        Cr_TL (int, int): upper-left position to calculate cross-hair block.
                          Assumes x,y coordinates with 0,0 as the top-left cell.
        Cr_BR (int, int): bottom-right position to calculate cross-hair block.
                          Assumes x,y coordinates with 0,0 as the top-left cell.
        Per_F (4D tensor-like): peripheral block filter tensor in FCHW format.
        Per_P (int): padding used for peripheral block convolution.
        Per_S (int): stride for peripheral block convolution.

    Returns:
        out (4D tensor-like): output tensor in NHWC format.
    """
    N, In_H, In_W, In_C = X.shape
    Out_C, _, Cr_F_H, Cr_F_W = Cr_F.shape

    # Acquire output dimensions for cross-hair block and peripheral block.
    top_left_x, top_left_y = Cr_TL
    bot_right_x, bot_right_y = Cr_BR

    # Dispatch convolution for crosshair block, and the 4 sub-blocks making up
    # the peripheral block.
    crosshair_out = dispatch(X, Cr_F, Cr_P, Cr_S,
                             (top_left_x, top_left_y),
                             (bot_right_x, bot_right_y))

    # Note that the peripheral blocks all share the same filter.
    peripheral_block_upper_out = dispatch(X, Per_F, Per_P, Per_S,
                                          (0, 0),
                                          (In_W, top_left_y-1))

    peripheral_block_lower_out = dispatch(X, Per_F, Per_P, Per_S,
                                          (0, bot_right_y+1),
                                          (In_W, In_H))

    peripheral_block_left_out = dispatch(X, Per_F, Per_P, Per_S,
                                         (0, top_left_y),
                                         (top_left_x-1, bot_right_y))

    peripheral_block_right_out = dispatch(X, Per_F, Per_P, Per_S,
                                          (bot_right_x+1, top_left_y),
                                          (In_W, bot_right_y))

    out = np.zeros((N, In_H, In_W, Out_C))

    # Combine the blocks together at the end and return
    out[:, top_left_y:bot_right_y+1, top_left_x:bot_right_x+1, :] = crosshair_out
    out[:, 0:top_left_y, 0:In_W, :] = peripheral_block_upper_out
    out[:, top_left_y:bot_right_y+1, 0:top_left_x, :] = peripheral_block_left_out
    out[:, top_left_y:bot_right_y+1, bot_right_x+1:In_W, :] = peripheral_block_right_out
    out[:, bot_right_y+1:In_H, 0:In_W, :] = peripheral_block_lower_out

    return out


def iota_initialised_input_tensor(N, H, W, C):
    return np.array([i for i in range(N*H*W*C)]).reshape((N, H, W, C))


def iota_initialised_filter_tensor(F, C, H, W):
    return iota_initialised_input_tensor(F, C, H, W)


if __name__ == "__main__":
    X = iota_initialised_input_tensor(1, 9, 9, 1)

    Cr_F, Cr_P, Cr_S = iota_initialised_filter_tensor(1, 1, 3, 3), 1, 1
    Cr_TL, Cr_BR = (3, 3), (5, 5)
    Pe_F, Pe_P, Pe_S = iota_initialised_filter_tensor(1, 1, 3, 3), 1, 1

    crosshair_block_args = (Cr_F, Cr_P, Cr_S, Cr_TL, Cr_BR)
    peripheral_block_args = (Pe_F, Pe_P, Pe_S)

    out = crosshair_conv(X, *crosshair_block_args, *peripheral_block_args)
    print(out.reshape((9, 9)))
