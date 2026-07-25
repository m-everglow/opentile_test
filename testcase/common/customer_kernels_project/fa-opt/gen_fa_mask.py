import torch
import triton
import triton.language as tl
import numpy as np

import triton.runtime.driver as driver
device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = properties["num_aicore"]
VECTOR_NUM = properties["num_vectorcore"]

def get_fwd_configs():
	configs = [
		triton.Config(
			{
				"BLOCK_M": BM,
				"BLOCK_N": BN,
			}, 
			num_stages=2,
		)
		for BM in [32, 64, 128]
		for BN in [32, 64, 128, 256]
	]
	return configs


@triton.jit
def mask_fn(q_attn_arg, k_attn_arg, q_offset, k_offset, TYPE: tl.constexpr):
	# tril_causal = q_offset[:, None] >= k_offset[None, :]
	# triu_causal = q_offset[:, None] <= k_offset[None, :]
	# attn_arg = 0 代表 sequence，非 0 代表 query，不同 query 用不同的 attn_arg
	if TYPE == 1:
		# return (q_offset[:, None] <= k_offset[None, :])
		triu_causal = (q_offset[:, None] <= k_offset[None, :]).to(tl.int32)
		# tl.device_print("q_offset[:, None] = ", q_offset[:, None])
		# tl.device_print("triu_causal_triton = ",triu_causal)
		# tl.device_print("q_attn_arg_triton = ",q_attn_arg)
		# tl.device_print("k_attn_arg_triton = ",k_attn_arg)
		attn_args_mask = ((q_attn_arg[:, None] == k_attn_arg[None, :]).to(tl.int32) |
				(k_attn_arg[None, :] == 0).to(tl.int32)).to(tl.int32)
		# tl.device_print("attn_args_mask_triton = ",attn_args_mask)
		return (
				(triu_causal &
				attn_args_mask) |
				(q_offset[:, None] == k_offset[None, :]).to(tl.int32))
	if TYPE == 2:
		tril_causal = (q_offset[:, None] >= k_offset[None, :])
		return ((tril_causal & ((q_attn_arg[:, None] == k_attn_arg[None, :]) | (k_attn_arg[None, :] == 0))) | (
					q_offset[:, None] == k_offset[None, :]))

@triton.jit
def load_if(block_ptr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
	if EVEN_M & EVEN_N:
		return tl.load(block_ptr)
	elif EVEN_M:
		return tl.load(block_ptr, boundary_check=(1,), padding_option="zero")
	elif EVEN_N:
		return tl.load(block_ptr, boundary_check=(0,), padding_option="zero")
	else:
		return tl.load(block_ptr, boundary_check=(0, 1), padding_option="zero")

@triton.jit
def store_if(block_ptr, value, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr):
	if EVEN_M & EVEN_N:
		tl.store(block_ptr, value)
	elif EVEN_N:
		tl.store(block_ptr, value, boundary_check=(0,))
	elif EVEN_M:
		tl.store(block_ptr, value, boundary_check=(1,))
	else:
		tl.store(block_ptr, value, boundary_check=(0, 1))

#@triton.autotune(
#    configs=get_fwd_configs(),
#    key=["BATCH_SIZE", "SPARSE_OPT", "MASK_FN"],
#)
@triton.jit
def gen_fa_mask_kernel(
		output_ptr,
		q_attn_arg_ptr, k_attn_arg_ptr,
		cu_seqlens_q, cu_seqlens_k,
		MASK_FN: tl.constexpr,
		BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
		AICORE_NUM: tl.constexpr,
		MAX_Q_LEN: tl.constexpr,
		MAX_K_LEN: tl.constexpr,
		BATCH_SIZE: tl.constexpr,
):
	ORI_MAX_Q_LEN = MAX_Q_LEN
	ORI_MAX_K_LEN = MAX_K_LEN
	pid = tl.program_id(0)
	task_nums = tl.cdiv(MAX_Q_LEN, BLOCK_M)
	# task_nums = task_nums * q_head
	task_nums = task_nums % 7
	MAX_Q_LEN = tl.where(task_nums == 0, MAX_Q_LEN + BLOCK_M, MAX_Q_LEN)
	NUM_BLOCKS_M = tl.cdiv(MAX_Q_LEN, BLOCK_M)  # 非对齐跳过
	NUM_BLOCKS = NUM_BLOCKS_M * BATCH_SIZE 
	zero_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int8)
	start_block, end_block, step = pid, NUM_BLOCKS, AICORE_NUM
	for block_idx in range(start_block, end_block, step):
		task_hz_idx = block_idx // NUM_BLOCKS_M
		start_m = block_idx % NUM_BLOCKS_M
		start_b = task_hz_idx.to(tl.int64)

		q_start1 = tl.load(cu_seqlens_q + start_b)
		q_end = tl.load(cu_seqlens_q + start_b + 1)
		q_len = q_end - q_start1
		# Cannot have `return` statements inside `while` or `for` statements in triton
		# unsupported AST node type: Continue
		# if start_m * BLOCK_M >= q_len:
		#     return
		if start_m * BLOCK_M < q_len:
			k_start1 = tl.load(cu_seqlens_k + start_b)
			k_end = tl.load(cu_seqlens_k + start_b + 1)
			k_len = k_end - k_start1

			begin = start_m * BLOCK_M
			if begin.to(tl.int64) < k_len.to(tl.int64):
				end = k_len

				# log2e: tl.constexpr = 1.4426950408889634
				offset_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

				q_start = q_start1.to(tl.int64)
				k_start = k_start1.to(tl.int64)
				q_attn_arg_block_ptr = tl.make_block_ptr(
					base = q_attn_arg_ptr + q_start,
					shape = (q_len,),
					strides = (1,),
					offsets = (start_m * BLOCK_M,),
					block_shape = (BLOCK_M,),
					order = (0,)
				)
				
				k_attn_arg_block_ptr = tl.make_block_ptr(
					base = k_attn_arg_ptr + k_start,
					shape = (k_len,),
					strides = (1,),
					offsets = (begin,),
					block_shape = (BLOCK_N,),
					order = (0,)
				)
				mask_out_ptr = tl.make_block_ptr(
					base=output_ptr + start_b * ORI_MAX_Q_LEN * ORI_MAX_K_LEN,
					shape=(q_len, k_len),
					strides=(ORI_MAX_K_LEN, 1),
					offsets=(start_m * BLOCK_M, begin),
					block_shape=(BLOCK_M, BLOCK_N),
					order=(1, 0)
				)
				if begin > 0:
					zero_mask_out_ptr = tl.make_block_ptr(
						base=output_ptr + start_b * ORI_MAX_Q_LEN * ORI_MAX_K_LEN,
						shape=(q_len, begin),
						strides=(ORI_MAX_K_LEN, 1),
						offsets=(start_m * BLOCK_M, 0),
						block_shape=(BLOCK_M, BLOCK_N),
						order=(1, 0)
					)
					for start_n in range(0, begin, BLOCK_N):
						store_if(zero_mask_out_ptr, zero_block, False, False)
						zero_mask_out_ptr = tl.advance(zero_mask_out_ptr, (0, BLOCK_N))
				q_attn_arg = load_if(q_attn_arg_block_ptr, False, True)

				for start_n in range(begin, end, BLOCK_N):
					start_n = tl.multiple_of(start_n, BLOCK_N)
					k_attn_arg = load_if(k_attn_arg_block_ptr, False, True)
					offset_n = start_n + tl.arange(0, BLOCK_N)
					mask = mask_fn(q_attn_arg, k_attn_arg, offset_m, offset_n, MASK_FN).to(tl.int8)

					# tl.device_print("mask   ", mask)
					store_if(mask_out_ptr, mask, False, False)
					# tl.store(mask_out_ptr, mask, mask=q_mask & k_mask)
					mask_out_ptr = tl.advance(mask_out_ptr, (0, BLOCK_N))

def generate_mask_fn_vectorized(cu_seqlens_q, cu_seqlens_k, bs, max_q_len, max_k_len, q_attn_arg, k_attn_arg):
	device = "npu"

	# 创建结果张量
	mask_fn = torch.zeros((bs, max_q_len, max_k_len), dtype=torch.bool, device=device)
	q_seq_list = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
	k_seq_list = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
	# 为每个batch独立处理
	for b_i in range(bs):
		cur_q_len = q_seq_list[b_i]
		cur_k_len = k_seq_list[b_i]

		# 创建位置索引
		q_positions = torch.arange(cur_q_len, device=device).view(-1, 1)
		k_positions = torch.arange(cur_k_len, device=device).view(1, -1)

		# 计算 causal mask: q_offset <= k_offset (注意这里是 <=, 不是 <)
		# 原始代码使用的是 triu_causal = (q_offset[:, None] <= k_offset[None, :])
		causal_mask = (q_positions <= k_positions)
		# print(f"{causal_mask=}")

		# 计算 attention args mask
		# 确保数据类型一致，原始代码使用的是 .bool()
		q_attn_slice = torch.tensor(q_attn_arg[cu_seqlens_q[b_i]:cu_seqlens_q[b_i + 1]], device=device, dtype=torch.int32).view(-1, 1)
		k_attn_slice = torch.tensor(k_attn_arg[cu_seqlens_k[b_i]:cu_seqlens_k[b_i + 1]], device=device, dtype=torch.int32).view(1, -1)

		# 原始逻辑: (cur_q_attn_args[:, None] == cur_k_attn_args[None, :]) | (cur_k_attn_args[None, :] == 0)
		# print(f"{q_attn_slice=}")
		# print(f"{k_attn_slice=}")
		attn_args_mask = (q_attn_slice == k_attn_slice) | (k_attn_slice == 0)
		# print(f"{attn_args_mask=}")

		# 计算 q offset mask: q_offset == k_offset
		q_offset_mask = (q_positions == k_positions)
		# print(f"{q_offset_mask=}")

		# 组合所有mask，保持与原始代码相同的布尔运算顺序
		# 原始: ((triu_causal.bool() & attn_args_mask.bool()) | q_offset_mask.bool())
		result_mask = ((causal_mask.bool() & attn_args_mask.bool()) | q_offset_mask.bool()).to(torch.bool)

		# 存储结果，不需要额外的valid_mask，因为我们只处理有效范围
		mask_fn[b_i, :cur_q_len, :cur_k_len] = result_mask

	return mask_fn

def test_mask_fn():
	dtype = torch.bfloat16
	DEVICE = torch.device("npu")
	
	q_cumsum = np.array([     0,    856,    879,    900,   1093,   2912,   3016,   4569,   5999,
		6384,   7961,   9233,  10667,  11006,  11972,  12000,  13436,  14675,
		14790,  16597,  18588,  19847,  21766,  24162,  25817,  27342,  28678,
		29266,  30446,  31398,  33586,  34596,  35653,  35887,  38168,  39758,
		40029,  40311,  42155,  42202,  43291,  43853,  44179,  45726,  46452,
		48794,  49105,  50795,  51637,  53378,  54993,  55394,  55587,  57620,
		59536,  61283,  63182,  63512,  65452,  65893,  67187,  67825,  68011,
		68336,  70046,  71568,  72851,  74855,  75578,  76079,  77199,  77692,
		78582,  79742,  81670,  83082,  83927,  85238,  87365,  87871,  89364,
		89885,  90358,  92610,  94055,  94362,  95971,  96161,  97401,  99217,
		99268, 100330, 101601, 103571, 103625, 105593, 107450, 109650, 110659,
		112381, 114410, 114446, 115427, 117345, 118386, 119475, 120617, 121652,
		123996, 124751, 125013, 126082, 127651, 130004, 132352, 132767, 132847,
		133609, 134267, 134488, 135708, 138031, 139296, 140622, 141561, 142308,
		142713, 143886, 143931, 144960, 146407, 146809, 147175, 148152, 149172,
		150348, 152693, 153556, 154122, 155658, 157245, 157955, 159130, 159534,
		159816, 161075, 162644, 162990, 165221, 165845, 166845, 168731, 170287,
		170351, 171934, 172121, 173460, 174002, 175866, 176838, 178065, 178722,
		180749, 181912, 183399, 183594, 183722, 184132, 184825, 185276, 187100,
		187645, 190004, 190676, 191492, 193630, 194512, 195604, 196546, 196651,
		198266, 198990, 200550, 202261, 204337, 206736, 207433, 209172, 211224,
		212827, 213752, 213866, 215875, 216534, 218151, 220069, 220527, 222176,
		223997, 225973, 227021, 227928, 230264, 231197, 231988, 232091, 233374,
		234415, 234883, 236873, 239190, 241027, 242356, 243989, 245793, 246326,
		246661, 246898, 248464, 248872, 250293, 250587, 252758, 253477, 254503,
		255231, 257250, 259242, 260156, 260542, 261212, 262637, 263555, 263772,
		265998, 266051, 266234, 266966, 267718, 268964, 271004, 273031, 273235,
		274588, 275102, 275680, 277695, 278585, 278978, 280224, 282023, 283694,
		285586, 287605, 288770, 289261, 291526])
	
	q_cumsum = np.array([     0,    463,   2330,   3521,   4960,   5918,   6973,   9215,  10091,
         10529,  10591,  12058,  13186,  14492,  15484,  15875,  17326,  19330,
         20832,  21249,  21467,  23749,  25115,  27081,  28221,  28419,  29676,
         31842,  32608,  33117,  34406,  35596,  37527,  38160,  39257,  40427,
         40595,  42973,  43952,  46268,  46563,  48286,  48420,  48449,  49116,
         51453,  53598,  53704,  54189,  55904,  56680,  58662,  59044,  59484,
         59925,  60020,  60084,  61369,  62624,  63784,  65843,  67178,  68680,
         70634,  71083,  72637,  72999,  73838,  74906,  75554,  77535,  79793,
         80965,  82341,  83166,  83315,  83337,  85564,  86374,  86389,  86568,
         88143,  88876,  89035,  89742,  91843,  93824,  94282,  95991,  97067,
         97253,  99064, 101312, 103402, 103986, 104500, 105580, 107597, 109583,
        111512, 111809, 114173, 115579, 116445, 118602, 120831, 122151, 122776,
        123673, 124803, 125001, 125133, 125544, 126841, 128670, 129087, 130045,
        130812, 131235, 133636, 135457, 135737, 136830, 138444, 138994, 139160,
        141542, 142724, 144386, 145051, 146032, 147311, 147653, 149451, 149772,
        150505, 150648, 153029, 154995, 155380, 157645, 159785, 160403, 162297,
        163216, 164985, 166990, 167340, 168905, 169671, 171720, 172251, 173404,
        175525, 175670, 176255, 176733, 178871, 180524, 181613, 181749, 182301,
        182902, 183092, 184914, 185380, 187647, 188084, 189903, 192252, 193518,
        194474, 194565, 196420, 197195, 198007, 198029, 199099, 200951, 202150,
        202506, 204479, 204759, 206946, 208447, 208690, 209367, 210233, 212356,
        213698, 214041, 214505, 215606, 217607, 219068, 221204, 221702, 222240,
        222539, 222947, 225062, 226600, 227443, 227505, 227927, 228475, 228490,
        230634, 232442, 232856, 232914, 234978, 236459, 238584, 240928, 243036,
        245280, 247259, 247844, 248147, 248581, 249515, 250101, 252259, 253760,
        254190, 255907, 256717, 256922, 258810, 259963, 260523, 262468, 263924,
        265354, 267747, 268063, 269916, 271456, 273288, 274241, 276418, 277536,
        278313, 280244, 282231, 283291, 283735, 284067, 284194, 286084, 287745,
        289456, 291706, 292411, 293000, 293184, 293327, 295004, 295447, 295526,
        296361, 296813, 298525, 299087, 299466, 300069, 301955, 303595, 305992,
        307603, 308094, 310357, 312556, 314217, 316088, 317117, 317474, 319502,
        321301, 321578, 322370, 322515, 322602, 324726, 325420, 325746, 327805,
        329506, 330738, 332235, 332834, 335009, 337086, 337529, 338437, 340546,
        342562, 343291, 344652, 346188, 347172, 348582, 349625, 351583, 353805,
        353995, 356053, 356834, 358907, 359468, 361654, 364010, 365536, 366122,
        368205, 368770, 369116, 371382, 372651, 372953, 374551, 375996, 378241,
        379173, 381310, 382405, 384486, 386102, 387346, 388844, 390533, 391513,
        393573, 395090, 396129, 396468, 397371, 397833, 399245, 400677, 401541,
        403635, 405160, 406704, 406956, 407558, 407989, 408307, 409934, 410992,
        411395, 412034, 413932, 414735, 415085, 415560, 417252, 418984, 420673,
        421634, 422256, 422779, 422911, 424077, 424097, 424881, 426901, 428885,
        430090, 432234, 432891, 434918, 435131, 437478, 439320, 439887, 441737,
        443693, 444527, 446877, 447110, 448802, 448832, 451214, 452381, 454386,
        456304, 457686, 459952, 462249, 463796, 466141, 468073, 468977, 471245,
        471281, 472362, 473375, 475340, 476750, 478092, 479184, 480956, 482220,
        484072, 485521, 486135, 487428, 489687, 491265, 491723, 492399, 493437,
        495763, 497595, 499419, 501156, 502045, 503660, 504996, 505493, 506261,
        507170, 507916, 508462, 510304, 510456, 512598, 512992, 513797, 516096,
        517678, 520065, 521646, 524009, 526098, 527493, 527576, 529747, 531846,
        533178, 534865, 536770, 538022, 540012, 540708, 542807, 543146, 544925,
        545962, 546337, 547456, 547851, 548607, 549603, 551907, 553559, 555540,
        556443, 558367, 560209, 560526, 561208, 562439, 562592, 563067, 564527,
        566022, 567570, 569039, 569513, 570053, 572386, 573525, 573803, 574044,
        575621, 576032, 577942, 579055, 579241, 579749, 581756, 582228, 584530,
        584707, 584740, 585941, 588195, 588570, 590862, 593024, 594399, 596000,
        597253, 598783, 599633, 601678, 603865, 605887, 607801, 609762, 612004,
        612206, 613203, 613307, 615509, 616675, 617653, 619195, 620467, 620479])

	# q_cumsum = np.array([     0,    1024])
		
	q_seq_list = q_cumsum[1:] - q_cumsum[:-1] # [0] + [320] * 8  # 0 for cumsum
	k_seq_list = q_seq_list # [0] + [320] * 8  # 0 for cumsum

	bs = len(q_seq_list)
	q_len = sum(q_seq_list)
	k_len = sum(k_seq_list)
	max_seqlen_q = int(np.max(q_seq_list))
	max_seqlen_k = int(np.max(k_seq_list))
	print(f"===>load {max_seqlen_q=}, {max_seqlen_q=}")
	# qkv
	
	q_attn_arg = torch.zeros(q_len, dtype=torch.int32, device="cpu")
	k_attn_arg = torch.zeros(k_len, dtype=torch.int32, device="cpu")

	cu_seqlens_q = q_cumsum.tolist()
	cu_seqlens_k = q_cumsum.tolist()
	# print(f"{cu_seqlens_q=}")
	cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cpu")
	cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, device="cpu")

	# BLOCK_M, BLOCK_N = 128, 128
	q_attn_arg = q_attn_arg.npu()
	k_attn_arg = k_attn_arg.npu()
	cu_seqlens_q = cu_seqlens_q.npu()
	cu_seqlens_k = cu_seqlens_k.npu()

	###################################
	# 分配输出内存
	mask_tensor = torch.empty((bs, max_seqlen_q, max_seqlen_k), dtype=torch.bool, device=device)
	q_attn_arg, k_attn_arg = q_attn_arg.to(torch.int32), k_attn_arg.to(torch.int32),
	cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.to(torch.int32), cu_seqlens_k.to(torch.int32),
	print(f"{mask_tensor.shape=}")
	NUM_CORES = VECTOR_NUM

	grid = (NUM_CORES,)
	# 启动 kernel
	gen_fa_mask_kernel[grid](
		mask_tensor,
		q_attn_arg, k_attn_arg,
		cu_seqlens_q, cu_seqlens_k,
		1,
		128,
		128,
		VECTOR_NUM,
		max_seqlen_q,
		max_seqlen_k,
		bs,
		multibuffer=True,
	)
	# torch.save({"mask_tensor": mask_tensor, "q_attn_arg": q_attn_arg, \
	#			  "k_attn_arg": k_attn_arg, "cu_seqlens_q": cu_seqlens_q, \
	#			  "cu_seqlens_k": cu_seqlens_k, "max_seqlen_q": max_seqlen_q, "max_seqlen_k": max_seqlen_k, "bs": bs}, "fa_gen_mask.pt")
	torch.npu.synchronize()
	mask_tensor_torch = generate_mask_fn_vectorized(cu_seqlens_q, cu_seqlens_k, bs, max_seqlen_q, max_seqlen_k, q_attn_arg, k_attn_arg)
	rtol = 0.0
	atol = 1e-2
	print(f"{mask_tensor=}")
	print(f"{mask_tensor_torch=}")

	assert torch.allclose(mask_tensor_torch, mask_tensor, atol=atol, rtol=rtol)
	

def tset_simulator():
	data = torch.load("fa_gen_mask.pt")
	mask_tensor = data["mask_tensor"]
	q_attn_arg = data["q_attn_arg"]
	k_attn_arg = data["k_attn_arg"]
	cu_seqlens_q = data["cu_seqlens_q"]
	cu_seqlens_k = data["cu_seqlens_k"]
	max_seqlen_q = data["max_seqlen_q"]
	max_seqlen_k = data["max_seqlen_k"]
	bs = data["bs"]
	NUM_CORES = VECTOR_NUM

	grid = (NUM_CORES,)
	# 启动 kernel
	gen_fa_mask_kernel[grid](
		mask_tensor,
		q_attn_arg, k_attn_arg,
		cu_seqlens_q, cu_seqlens_k,
		1,
		128,
		128,
		VECTOR_NUM,
		max_seqlen_q,
		max_seqlen_k,
		bs,
		multibuffer=True,
	)

if __name__ == "__main__":
	test_mask_fn()
	# tset_simulator()
