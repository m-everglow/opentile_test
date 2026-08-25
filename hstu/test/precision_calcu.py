#
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#

import torch
import torch_npu

MIN_ERR = 1e-7

def get_err_threshold(dtype:torch.dtype):
    err_threshold = 0
    if dtype in [torch.bfloat16]:
        err_threshold = 2**(-8)
    if dtype in [torch.float16]:
        err_threshold = 2**(-11)
    if dtype in [torch.float32]:
        err_threshold = 2**(-14)
    return err_threshold

#最大相对误差：max relative error，MARE
def get_mare(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mare = torch.max(abs_error.flatten())
    return mare

#平均相对误差：mean relative error，MERE
def get_mere(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mere = torch.mean(abs_error)
    return mere

#均方根误差:Root Mean Squared Error，RMSE
def get_rmse(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse

def compare_cv(golden:torch.Tensor, gpu:torch.Tensor, actual:torch.Tensor,
               mare_threshold=2, mere_threshold=1.2, rmse_threshold=1.2):

    err_threshold = get_err_threshold(actual.dtype)
    print(f"err_threshold:{err_threshold}")
    mare_npu = get_mare(golden, actual)
    mare_gpu = get_mare(golden, gpu)

    mere_npu = get_mere(golden, actual)
    mere_gpu = get_mere(golden, gpu)

    rmse_npu = get_rmse(golden, actual)
    rmse_gpu = get_rmse(golden, gpu)

    mare_rate = mare_npu / max(mare_gpu, err_threshold)
    mere_rate = mere_npu / max(mere_gpu, err_threshold)
    rmse_rate = rmse_npu / max(rmse_gpu, err_threshold)

    result = (mare_rate < mare_threshold) and (mere_rate < mere_threshold) \
             and (rmse_rate < rmse_threshold)

    print(f"mare_npu:{mare_npu} mare_gpu:{mare_gpu}")
    print(f"mere_npu:{mere_npu} mere_gpu:{mere_gpu}")
    print(f"rmse_npu:{rmse_npu} rmse_gpu:{rmse_gpu}")
    print(f"MARE:{mare_rate} MERE:{mere_rate} RMSE:{rmse_rate}")
    print(f"new golden cv result:{result}")
    return result