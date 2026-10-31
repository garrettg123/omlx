"""UNEXECUTED STATIC SPECIFICATION

Not measured numerical/FP32 equivalence, environment provenance, or B-N1 PASS.
Never imported or run in STATIC-B-N1-PREP. No fixtures or expected results.
Independent scalar equation transcription, not copied production helpers.

Text custody: /tmp/elephant-numerical-text-drive-48212-Mac/source-sha256.txt.
Publisher baselabs/elephant-1 b3f8a93462fb1051a9de031e94a7ff380e254072,
serve.patch:248-268 defines full-attention block visibility, causal GDN.
mlx-lm 0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d source-text contracts:
  qwen3_5.py:134-206: conv, Q/K normalization, recurrent cache, gated norm;
  gated_delta.py:18-20,34-45,385-426,557-647: decay, repeats, state update;
  qwen3_next.py:55-151: RMS Q/K, per-head Q/gate split, partial RoPE, GQA;
  base.py:110-139: unquantized SDPA forwards mask and scale unchanged;
  models_rope_utils.py:294-310: default RoPE, traditional=False;
  models_activations.py:16-20: gated norm multiplication in FP32.

All arrays are nested scalar sequences, batch B=1 omitted. Inputs must be
finite, rectangular, unpadded, nonempty, correctly shaped and source-rounded.
No models, imports, assets, I/O, fixtures, decorators, defaults, or annotations
are evaluated at module definition time. Standard math imports occur ONLY
inside unexecuted function bodies. No entrypoint or module-level calls.

Dtype notation: source dtype is explicitly fp32, bf16, or fp16. FP32 SSM
state is [Hv,Dv,Dk], NOT [Hv,Dk,Dv]. Round-to-nearest-even scalar boundaries
are spelled out below; host transcendental accuracy, MLX fused operations,
normalization kernels and parallel reduction order are NOT emulated/proved.
Norm/softmax formulas use host intermediates then stated output rounding;
sequential FP32 sums are a specification choice, NOT a kernel-equivalence
claim. Neither source-dtype rounding nor FP32 operands prove native parity.
This is not a full hybrid model, projection/FFN/readout, cache-isolation test,
vector-gate/lower-bound/negative-eigenvalue GDN, scaled RoPE, or Metal oracle.
"""


def round_source(x, dtype):
    """Finite scalar IEEE binary rounding; fail rather than invent overflow.

    Supports subnormals; zero results retain the input sign, including
    underflow. Host round uses ties-to-even; later audit of boundaries owed.
    """
    from math import copysign, frexp, isfinite, ldexp

    formats = {"fp32": (24, -126, 127), "bf16": (8, -126, 127),
               "fp16": (11, -14, 15)}
    precision, emin, emax = formats[dtype]
    if not isfinite(x):
        raise ValueError("finite scalar inputs required")
    if x == 0:
        return x
    _, exponent = frexp(abs(x))
    quantum_exponent = max(exponent - precision, emin - precision + 1)
    rounded = ldexp(round(ldexp(x, -quantum_exponent)), quantum_exponent)
    if not isfinite(rounded) or abs(rounded) >= ldexp(1.0, emax + 1):
        raise ValueError("rounded overflow outside specification")
    if rounded == 0:
        return copysign(0.0, x)
    return rounded


def fp32(x):
    """Explicit FP32 boundary, not merely Python float(x)."""
    return round_source(x, "fp32")


def dot32(x, y):
    """Equal-length vectors -> scalar, sequential FP32 products/additions."""
    if len(x) != len(y):
        raise ValueError("dot length mismatch")
    total = fp32(0.0)
    for left, right in zip(x, y):
        total = fp32(total + fp32(fp32(left) * fp32(right)))
    return total


def sigmoid(x):
    """Stable real sigmoid; caller specifies source rounding boundary."""
    from math import exp

    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    ex = exp(x)
    return ex / (1.0 + ex)


def softplus(x):
    """Positive softplus, log(1+exp(x)), NOT its negative."""
    from math import exp, log1p

    return max(x, 0.0) + log1p(exp(-abs(x)))


def rms(x, weight, epsilon, dtype):
    """[D],[D] -> [D]: gamma*x/sqrt(mean(x*x)+epsilon).

    Attention Q/K and gated output use this RMS norm, unlike GDN L2 norm.
    Ideal host norm followed by source rounding; no fused RMS proof.
    """
    from math import sqrt

    if not x or len(x) != len(weight) or epsilon <= 0:
        raise ValueError("RMS geometry/epsilon")
    denominator = sqrt(sum(v * v for v in x) / len(x) + epsilon)
    return [round_source(w * v / denominator, dtype)
            for v, w in zip(x, weight)]


def normalize_gdn(rawq, rawk, dtype):
    """One key head [Dk],[Dk] -> q,k [Dk], rounded to source dtype.

    eps=1e-6 belongs to SUM of squares, not mean. q alone has 1/sqrt(Dk).
    Pinned RMS implementation uses rms_eps=eps/Dk, q multiplier=1/Dk,
    k multiplier=1/sqrt(Dk). These are algebraically the L2 equations here.
    """
    from math import sqrt

    dk = len(rawq)
    if dk == 0 or len(rawk) != dk:
        raise ValueError("Q/K geometry")
    qden = sqrt(sum(v * v for v in rawq) + 1e-6) * sqrt(dk)
    kden = sqrt(sum(v * v for v in rawk) + 1e-6)
    return ([round_source(v / qden, dtype) for v in rawq],
            [round_source(v / kden, dtype) for v in rawk])


def causal_conv_silu(qkv, history, kernel, dtype):
    """[T,C], [K-1,C], [C,K] -> activated [T,C], new history [K-1,C].

    Inputs already projected/source-rounded. Kernel is oldest-to-current
    depthwise cross-correlation, not reversed convolution. History contains
    pre-conv rows; callers explicitly supply zero history or real prefix.
    No padding mask: this unit specifies unpadded sequences only.
    """
    ksize = len(kernel[0])
    if ksize < 2 or len(history) != ksize - 1 or not qkv:
        raise ValueError("conv geometry")
    joined = [list(row) for row in history] + [list(row) for row in qkv]
    result = []
    for t in range(len(qkv)):
        row = []
        for c in range(len(kernel)):
            pre = round_source(dot32([joined[t + j][c]
                                      for j in range(ksize)], kernel[c]), dtype)
            row.append(round_source(pre * sigmoid(pre), dtype))
        result.append(row)
    return result, [list(row) for row in joined[-(ksize - 1):]]


def gdn_step(rawq, rawk, v, a, b, a_log, dt_bias, state, dtype):
    """One unpadded token; raw q/k [Hk,Dk], v [Hv,Dv].

    a,b,A_log,dt_bias [Hv]; explicit incoming state [Hv,Dv,Dk].
    Sequence entry requires NONZERO state; a later state may become zero.
    a,b,dt_bias share dtype here; mixed-parameter promotion is not modeled.
    q/k/v are post-conv SiLU split; C=2*Hk*Dk+Hv*Dv, z is separate.
    Returns y [Hv,Dv] in q/source dtype, fresh S [Hv,Dv,Dk] FP32.
    No input mutation; each head owns its own state matrix. Ascending time:
      D = g*S_old; delta = beta*(v-D*k); S_new = D+delta outer k;
      y = S_new*q (not S_old*q, not q outer delta).
    Scalar decay only. No masked tokens; no extra output 1/sqrt(Dk).
    """
    from math import exp

    hk, hv = len(rawq), len(v)
    if hk == 0 or hv % hk or len(rawk) != hk or len(state) != hv:
        raise ValueError("GDN heads")
    qk = [normalize_gdn(q, k, dtype) for q, k in zip(rawq, rawk)]
    repeats = hv // hk
    outputs, new_state = [], []
    for h in range(hv):
        q, k = qk[h // repeats]  # contiguous Hk->Hv repeat, NOT h % Hk
        # Source a+dt_bias and softplus; A_log explicitly cast before exp.
        sp = round_source(softplus(round_source(a[h] + dt_bias[h], dtype)), dtype)
        g = fp32(exp(-fp32(fp32(exp(fp32(a_log[h]))) * fp32(sp))))
        beta = round_source(sigmoid(b[h]), dtype)  # no factor two
        decayed = [[fp32(g * fp32(value)) for value in row]
                   for row in state[h]]
        delta = [fp32(fp32(fp32(v[h][j]) - dot32(row, k)) * fp32(beta))
                 for j, row in enumerate(decayed)]
        head = [[fp32(value + fp32(delta[j] * fp32(k[i])))
                 for i, value in enumerate(row)]
                for j, row in enumerate(decayed)]
        outputs.append([round_source(dot32(row, q), dtype) for row in head])
        new_state.append(head)
    return outputs, new_state


def gdn_sequence(rawq, rawk, v, a, b, a_log, dt_bias, initial_state, dtype):
    """Time-leading Q/K [T,Hk,Dk], V [T,Hv,Dv], a/b [T,Hv].

    Returns [T,Hv,Dv], final FP32 state. Prefix then suffix continues the
    returned state, NEVER resets it; no implicit global/cache ownership.
    Initial state is required, nonzero and nonsymmetric in later fixtures.
    No fixture is constructed here. No future timestep is read by a step.
    """
    count = len(rawq)
    if not count or any(len(x) != count for x in (rawk, v, a, b)):
        raise ValueError("sequence lengths")
    if not any(value != 0 for head in initial_state
               for row in head for value in row):
        raise ValueError("nonzero initial state required by this specification")
    state = [[[fp32(value) for value in row] for row in head]
             for head in initial_state]
    outputs = []
    for t in range(count):
        y, state = gdn_step(rawq[t], rawk[t], v[t], a[t], b[t],
                            a_log, dt_bias, state, dtype)
        outputs.append(y)
    return outputs, state


def gated_rms_output(y, z, weight, epsilon, dtype):
    """One head [Dv] -> [Dv], before flattening Hv and output projection.

    RMS result rounded to source, converted to FP32, multiplied by SiLU of
    z converted to FP32, then cast to y/source dtype (precise_swiglu).
    """
    normed = rms(y, weight, epsilon, dtype)
    return [round_source(fp32(fp32(n) * fp32(fp32(g) * sigmoid(fp32(g)))), dtype)
            for n, g in zip(normed, z)]


def partial_rope(x, position, rotary_dims, theta, dtype):
    """[D] -> [D], default absolute RoPE, traditional=False.

    r=int(D*partial_rotary_factor), even 0<=r<=D. Pair i with i+r/2,
    NOT adjacent pairs; angle=position/theta**(2*i/r). Unrotated tail is
    preserved. No YaRN/mRoPE/custom scaling. Suffix starts at prefix P.
    """
    from math import cos, sin

    if rotary_dims < 0 or rotary_dims > len(x) or rotary_dims % 2 or theta <= 0:
        raise ValueError("partial RoPE geometry")
    out = list(x)
    half = rotary_dims // 2
    for i in range(half):
        angle = position / theta ** (2.0 * i / rotary_dims)
        c, s = cos(angle), sin(angle)
        out[i] = round_source(x[i] * c - x[i + half] * s, dtype)
        out[i + half] = round_source(x[i + half] * c + x[i] * s, dtype)
    return out


def attention_rows(qgate, keys, values, query_positions, key_positions,
                   prefix_length, qweight, kweight, epsilon,
                   rotary_dims, theta, dtype):
    """Projected qgate [Q,Hq,2D], K/V [K,Hkv,D] -> gated [Q,Hq,D].

    qgate is split WITHIN each head into Q then sigmoid gate; q/k RMS
    weights [D] are per-head-dimension, shared across heads. No biases or
    projections evaluated here. Projected inputs must be source-rounded.
    Absolute positions select block 0 (<P) or 1 (>=P). No causal fallback.
    Stable softmax is ideal host FP64 then source-rounded; native fused
    SDPA intermediate precision/reduction order remains unproved.
    """
    from math import exp, sqrt

    hq, hkv, dim = len(qgate[0]), len(keys[0]), len(keys[0][0])
    if hkv == 0 or hq % hkv or len(query_positions) != len(qgate):
        raise ValueError("attention heads/positions")
    if len(keys) != len(values) or len(keys) != len(key_positions):
        raise ValueError("KV positions")
    repeated = hq // hkv
    rotated_k = [[partial_rope(rms(head, kweight, epsilon, dtype), pos,
                               rotary_dims, theta, dtype) for head in row]
                 for pos, row in zip(key_positions, keys)]
    output = []
    for pos, row in zip(query_positions, qgate):
        token = []
        for h in range(hq):
            if len(row[h]) != 2 * dim:
                raise ValueError("per-head Q/gate width")
            query = partial_rope(rms(row[h][:dim], qweight, epsilon, dtype),
                                 pos, rotary_dims, theta, dtype)
            kvh = h // repeated  # GQA contiguous repeats
            # block(k)<=block(q): prefix NEVER reads suffix; both blocks
            # see their own future tokens. Suffix can read all prefix.
            allowed = [j for j, kp in enumerate(key_positions)
                       if int(kp >= prefix_length) <= int(pos >= prefix_length)]
            if not allowed:
                raise ValueError("empty visibility")
            scores = [dot32(query, rotated_k[j][kvh]) / sqrt(dim)
                      for j in allowed]  # attention scale exactly once
            peak = max(scores)
            masses = [exp(score - peak) for score in scores]
            total = sum(masses)
            probabilities = [round_source(mass / total, dtype) for mass in masses]
            context = [round_source(dot32(probabilities,
                                          [values[j][kvh][d] for j in allowed]), dtype)
                       for d in range(dim)]
            token.append([round_source(c * round_source(sigmoid(g), dtype), dtype)
                          for c, g in zip(context, row[h][dim:])])
        output.append(token)
    return output


def two_block_attention(qgate, keys, values, prefix_length, qweight, kweight,
                        epsilon, rotary_dims, theta, dtype):
    """Whole [P+S,...] -> [P+S,Hq,D], P>0,S>0, absolute positions 0..T-1.

    This is ONE full-attention layer over two prompt blocks, not two causal
    layers or a fully unmasked prompt. Output is before flatten/o_proj.
    """
    if not 0 < prefix_length < len(qgate) or len(keys) != len(qgate):
        raise ValueError("two nonempty blocks")
    positions = list(range(len(qgate)))
    return attention_rows(qgate, keys, values, positions, positions,
                          prefix_length, qweight, kweight, epsilon,
                          rotary_dims, theta, dtype)


def suffix_attention(qgate_suffix, prefix_keys, prefix_values, suffix_keys,
                     suffix_values, qweight, kweight, epsilon,
                     rotary_dims, theta, dtype):
    """Suffix query calculation -> [S,Hq,D], absolute query offset=P.

    Prefix/suffix K here are UNROTATED projected source rows, not cached
    already-rotated keys. Each key rotates exactly once at its absolute
    position. Production cached prefix keys are already rotated; this
    separate interface avoids pretending to verify actual cache copying.
    No input mutation; sibling ownership/MLX buffer isolation is unproved.
    """
    p, s = len(prefix_keys), len(qgate_suffix)
    if p == 0 or s == 0 or len(suffix_keys) != s:
        raise ValueError("two nonempty blocks")
    return attention_rows(qgate_suffix, list(prefix_keys) + list(suffix_keys),
                          list(prefix_values) + list(suffix_values),
                          list(range(p, p + s)), list(range(p + s)), p,
                          qweight, kweight, epsilon, rotary_dims, theta, dtype)
