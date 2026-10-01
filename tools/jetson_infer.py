"""Run a TensorRT engine over a captured scene on the Jetson and save patch features.

Device-side, Python 3.6, numpy 1.13, JetPack OpenCV. Nothing is installed: the TensorRT
Python bindings ship with JetPack, and device memory is handled by calling libcudart
through ctypes rather than pulling in pycuda. Twenty lines against one less thing to
install on a board that cannot run a modern pip.

    python3 tools/jetson_infer.py \
        --engine engines/dinov3_vits16_384x672_fp16.plan \
        --scene scenes/occlusion --out feats/occlusion_384x672

Writes feat_%04d.npy (fp16, CxHxW, already L2-normalised in the graph) and timings.csv.
"""

import argparse
import ctypes
import glob
import os
import time

import cv2
import numpy as np
import tensorrt as trt

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
H2D, D2H = 1, 2  # cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost

_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cudart.cudaDeviceSynchronize.argtypes = []


def _check(code, what):
    if code != 0:
        raise RuntimeError(f"{what} failed with CUDA error {code}")


def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    _check(_cudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return ptr


def copy_to_device(dst, host_array):
    _check(
        _cudart.cudaMemcpy(dst, host_array.ctypes.data_as(ctypes.c_void_p), host_array.nbytes, H2D),
        "cudaMemcpy H2D",
    )


def copy_to_host(host_array, src):
    _check(
        _cudart.cudaMemcpy(host_array.ctypes.data_as(ctypes.c_void_p), src, host_array.nbytes, D2H),
        "cudaMemcpy D2H",
    )


def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("could not deserialise " + path + " (built by a different TensorRT?)")
    return engine


def preprocess(bgr, h, w):
    """BGR uint8 -> 1x3xHxW float32, matching dvos.backbone.preprocess."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != h or rgb.shape[1] != w:
        # INTER_AREA on the way down is the honest match for torch's antialiased bilinear;
        # the residual difference is reported in the write-up, not hidden.
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--scene", required=True, help="directory of frame_*.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.scene, "frame_*.png")))
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit("no frames in " + args.scene)
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    engine = load_engine(args.engine)
    context = engine.create_execution_context()

    in_idx = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)][0]
    out_idx = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)][0]
    in_shape = tuple(engine.get_binding_shape(in_idx))
    out_shape = tuple(engine.get_binding_shape(out_idx))
    _, _, h, w = in_shape
    print(f"engine {os.path.basename(args.engine)}: in {in_shape} out {out_shape}")

    host_in = np.zeros(in_shape, dtype=np.float32)
    host_out = np.zeros(out_shape, dtype=np.float32)
    d_in = cuda_malloc(host_in.nbytes)
    d_out = cuda_malloc(host_out.nbytes)
    bindings = [0] * engine.num_bindings
    bindings[in_idx] = int(d_in.value)
    bindings[out_idx] = int(d_out.value)

    rows = [("frame", "read_ms", "pre_ms", "h2d_ms", "infer_ms", "d2h_ms", "total_ms")]
    n_bad = 0
    try:
        for n, path in enumerate(frames):
            t0 = time.time()
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            t1 = time.time()
            host_in[...] = preprocess(bgr, h, w)
            t2 = time.time()
            copy_to_device(d_in, host_in)
            _cudart.cudaDeviceSynchronize()
            t3 = time.time()
            if not context.execute_v2(bindings):
                raise RuntimeError(f"execute_v2 returned False on frame {n}")
            _cudart.cudaDeviceSynchronize()
            t4 = time.time()
            copy_to_host(host_out, d_out)
            _cudart.cudaDeviceSynchronize()
            t5 = time.time()

            # An engine that silently emits NaN still benchmarks beautifully: trtexec feeds
            # random input and never looks at the output, which is how a whole FP16 sweep
            # got timed before anyone noticed it computed nothing. Check the first frame
            # loudly, and count the rest.
            finite = int(np.isfinite(host_out).sum())
            if finite != host_out.size:
                n_bad += 1
                if n == 0:
                    raise SystemExit(
                        f"frame 0 output is not finite ({finite}/{host_out.size} finite). The engine runs and times "
                        "fine but computes nothing. On TensorRT 8.2 this is usually fp16 overflow "
                        "in the decomposed LayerNorm; rebuild the engine without --fp16."
                    )
            np.save(os.path.join(args.out, f"feat_{n:04d}.npy"), host_out[0].astype(np.float16))
            ms = lambda a, b: round((b - a) * 1000.0, 2)  # noqa: E731
            rows.append((n, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t0, t5)))
            if n == 0 or (n + 1) % 10 == 0:
                print(f"  frame {n:3d}  infer {rows[-1][4]:6.1f} ms  total {rows[-1][6]:6.1f} ms")
    finally:
        _cudart.cudaFree(d_in)
        _cudart.cudaFree(d_out)

    csv_path = os.path.join(args.out, "timings.csv")
    with open(csv_path, "w") as f:
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    infer = [r[4] for r in rows[1:]]
    infer_sorted = sorted(infer)
    median = infer_sorted[len(infer_sorted) // 2]
    print(
        f"\n{len(infer)} frames, infer median {median:.1f} ms ({1000.0 / median:.2f} fps), features -> {args.out}"
    )
    print("timings -> " + csv_path)
    if n_bad:
        print(f"WARNING: {n_bad} of {len(frames)} frames had non-finite features")


if __name__ == "__main__":
    main()
