"""
deployment/inference_engine.py
──────────────────────────────
Phase 7: ONNX export and inference engine for DOCR-MARL.

Provides:
1. Model export to ONNX format for cross-platform deployment
2. Latency benchmarking (<100 ms target)
3. Batch inference support
4. Memory usage profiling
5. Quantization support (INT8) for edge devices

Usage:
    from deployment.inference_engine import ONNXExporter, InferenceEngine

    exporter = ONNXExporter(actor_network)
    exporter.export("relay_model.onnx")

    engine = InferenceEngine("relay_model.onnx")
    result = engine.infer(observation)
    print(f"Latency: {engine.latency_ms:.2f} ms")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Optional, Callable

__all__ = [
    "InferenceConfig",
    "ONNXExporter",
    "InferenceEngine",
    "RelayDeploymentPack",
    "export_marl_deployment",
]

# ── Logging ─────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    """Configuration for the inference engine."""

    batch_size: int = 1
    warmup_runs: int = 10
    benchmark_runs: int = 100
    target_latency_ms: float = 100.0
    use_quantization: bool = False
    quantization_bits: int = 8  # 8 or 16
    threads: int = 1
    input_shape: tuple = (10,)


# ── Quantization ─────────────────────────────────────────────────────────
#
# Symmetric per-tensor "fake quantization": round weights to the nearest of
# 2**quantization_bits representable levels, then dequantize back to float
# so the numpy/onnxruntime forward pass can run unchanged. This mirrors
# quantize_int8() in experiments/exp_fig53_model_compression.py (the
# reference implementation used for Fig 53's model-compression figure),
# generalized here from a hardcoded 8-bit/127-level scale to an arbitrary
# ``quantization_bits`` so InferenceConfig.quantization_bits actually has
# an effect. Without this, use_quantization=True was a no-op: the "INT8
# model" was bit-identical to the full-precision one and any reported
# accuracy/size deltas were fabricated by construction.


def fake_quantize(weights: np.ndarray, bits: int) -> np.ndarray:
    """Simulate fixed-point quantization of ``weights`` at ``bits`` bit-width.

    Symmetric signed quantization: scale = max(|w|) / (2**(bits-1) - 1),
    round(w / scale) clipped to the representable signed integer range,
    then dequantized (q * scale) back to float. Bit widths >= 32 are a
    no-op (nothing coarser than float32 to simulate).

    Parameters
    ----------
    weights : np.ndarray
        Full-precision weight tensor.
    bits : int
        Target bit-width (e.g. 8, 4, 16).

    Returns
    -------
    np.ndarray
        Dequantized weights, same shape/dtype as the input, perturbed by
        the quantization round-trip (identical to the input only in the
        degenerate all-zero case).
    """
    weights = np.asarray(weights)
    if bits >= 32:
        return weights.copy()

    qmax = 2 ** (bits - 1) - 1  # e.g. 127 for 8 bits, 7 for 4 bits
    max_abs = np.max(np.abs(weights)) if weights.size else 0.0
    if max_abs < 1e-9:
        return weights.copy()

    scale = max_abs / qmax
    quantized = np.clip(np.round(weights / scale), -qmax - 1, qmax)
    dequantized = quantized * scale
    return dequantized.astype(weights.dtype)


# ── ONNX Exporter ───────────────────────────────────────────────────────────

class ONNXExporter:
    """Export DOCR-MARL models to ONNX format.

    Supports:
    - Actor network (policy) export
    - Critic network (value) export
    - Dynamic batch sizes
    - Opset version selection
    """

    def __init__(
        self,
        model: Callable,
        input_shape: tuple = (10,),
    ) -> None:
        """
        Parameters
        ----------
        model : Callable
            The trained actor to export. Expected to be either:
            - a ``SharedGaussianActor`` (marl/marl_agents.py) instance,
              exposing ``.net`` (an ``MLP`` with real ``W1/b1/W2/b2/W3/b3``
              weight matrices) and a ``get_mean(obs)`` method (which applies
              a final ``tanh`` on top of the raw MLP output), or
            - a raw ``MLP`` instance itself (``W1/b1/W2/b2/W3/b3`` directly,
              no final ``tanh``), or
            - any other object exposing ``W1/b1/W2/b2/W3/b3`` attributes.
            The object must also be callable, or expose ``get_mean``/
            ``forward``, so a forward pass can be run for verification.
        input_shape : tuple
            Expected input shape (excluding batch dimension).
        """
        self.model = model
        self.input_shape = input_shape
        self.metadata = {
            "framework": "numpy",
            "model_type": "DOCR-MARL",
            "input_shape": input_shape,
            "output_description": "[delta_TMS, delta_PS] per relay",
        }

    # ── Real-weight introspection ───────────────────────────────────────────

    def _get_architecture(self) -> dict:
        """Extract the real trained weight matrices from ``self.model``.

        Returns a dict with keys ``W1, b1, W2, b2, W3, b3, apply_tanh`` —
        the actual learned parameters, never fabricated/random ones. Raises
        ``TypeError`` if ``self.model`` doesn't expose a recognizable
        network structure (a ``SharedGaussianActor``-like wrapper with a
        ``.net``, or a raw MLP-like object with ``W1..b3`` directly).

        2026-10-22: the old hand-rolled ``MLP`` (``W1/b1/W2/b2/W3/b3``
        NumPy arrays) was replaced by a real ``torch.nn.Module``
        (``_TorchMLP``, with ``fc1/fc2/fc3`` ``nn.Linear`` layers) as part
        of the ES-to-backprop rewrite -- this broke ONNX export outright
        (confirmed: raised TypeError on any post-rewrite actor) until this
        branch was added. ``nn.Linear(in, out).weight`` has shape
        ``(out, in)`` and ``.bias`` has shape ``(out,)``, exactly matching
        the old ``W1``/``b1`` convention, so no transpose is needed -- this
        is a direct, shape-compatible extraction, not a reinterpretation.
        """
        model = self.model
        # A SharedGaussianActor-style wrapper stores the real network on
        # `.net` and applies tanh() on top of the raw output in get_mean().
        if hasattr(model, "net") and all(
            hasattr(model.net, attr) for attr in ("fc1", "fc2", "fc3")
        ):
            torch_net = model.net
            apply_tanh = hasattr(model, "get_mean")
            import torch as _torch
            with _torch.no_grad():
                return {
                    "W1": torch_net.fc1.weight.detach().cpu().numpy().astype(np.float32),
                    "b1": torch_net.fc1.bias.detach().cpu().numpy().astype(np.float32),
                    "W2": torch_net.fc2.weight.detach().cpu().numpy().astype(np.float32),
                    "b2": torch_net.fc2.bias.detach().cpu().numpy().astype(np.float32),
                    "W3": torch_net.fc3.weight.detach().cpu().numpy().astype(np.float32),
                    "b3": torch_net.fc3.bias.detach().cpu().numpy().astype(np.float32),
                    "apply_tanh": apply_tanh,
                }
        if hasattr(model, "net") and all(
            hasattr(model.net, attr) for attr in ("W1", "b1", "W2", "b2", "W3", "b3")
        ):
            net = model.net
            apply_tanh = hasattr(model, "get_mean")
        elif all(hasattr(model, attr) for attr in ("W1", "b1", "W2", "b2", "W3", "b3")):
            # A raw MLP (or MLP-compatible) object passed directly.
            net = model
            apply_tanh = False
        else:
            raise TypeError(
                "ONNXExporter requires a model exposing real weight matrices "
                "(a SharedGaussianActor with a `.net` MLP/torch module, or "
                "an MLP-like object with W1/b1/W2/b2/W3/b3 attributes "
                f"directly). Got {type(model)!r} which has neither."
            )

        return {
            "W1": np.asarray(net.W1, dtype=np.float32),
            "b1": np.asarray(net.b1, dtype=np.float32),
            "W2": np.asarray(net.W2, dtype=np.float32),
            "b2": np.asarray(net.b2, dtype=np.float32),
            "W3": np.asarray(net.W3, dtype=np.float32),
            "b3": np.asarray(net.b3, dtype=np.float32),
            "apply_tanh": apply_tanh,
        }

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Run a real forward pass through ``self.model`` for a single input.

        Prefers calling the model directly (it may be a plain callable);
        falls back to ``get_mean``/``forward`` for objects that aren't
        directly callable (e.g. a bare ``SharedGaussianActor``, which has
        no ``__call__``).
        """
        model = self.model
        if callable(model):
            return model(x)
        if hasattr(model, "get_mean"):
            return model.get_mean(x)
        if hasattr(model, "forward"):
            return model.forward(x)
        raise TypeError(
            f"Model {type(model)!r} is not callable and has neither "
            "`get_mean` nor `forward`; cannot run a verification pass."
        )

    def export(
        self,
        output_path: str,
        opset_version: int = 11,
        verify: bool = True,
    ) -> dict:
        """Export model to ONNX or numpy-npz fallback.

        Parameters
        ----------
        output_path : str
            Destination file path.
        opset_version : int
            ONNX opset version (used when onnx is available).
        verify : bool
            Run a forward-pass verification after export.

        Returns
        -------
        dict
            Export metadata and verification results.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Pull the model's REAL trained weights (never fabricated/random ones).
        arch = self._get_architecture()
        W1, b1 = arch["W1"], arch["b1"]
        W2, b2 = arch["W2"], arch["b2"]
        W3, b3 = arch["W3"], arch["b3"]
        apply_tanh = arch["apply_tanh"]

        # ── Attempt real ONNX export if library is available ───────────────
        try:
            import onnx  # noqa: F401
            import onnx.helper as oh
            import onnx.numpy_helper as onh
            from onnx import TensorProto as tp

            out_features = int(W3.shape[0])

            flat_node = oh.make_node(
                "Flatten", inputs=["obs"], outputs=["obs_flat"]
            )
            node_gemm1 = oh.make_node(
                "Gemm", inputs=["obs_flat", "W1", "b1"], outputs=["h1_pre"], transB=1
            )
            node_relu1 = oh.make_node("Relu", inputs=["h1_pre"], outputs=["h1"])
            node_gemm2 = oh.make_node(
                "Gemm", inputs=["h1", "W2", "b2"], outputs=["h2_pre"], transB=1
            )
            node_relu2 = oh.make_node("Relu", inputs=["h2_pre"], outputs=["h2"])
            final_output_name = "action_pre" if apply_tanh else "action"
            node_gemm3 = oh.make_node(
                "Gemm", inputs=["h2", "W3", "b3"], outputs=[final_output_name], transB=1
            )
            nodes = [flat_node, node_gemm1, node_relu1, node_gemm2, node_relu2, node_gemm3]
            if apply_tanh:
                nodes.append(
                    oh.make_node("Tanh", inputs=["action_pre"], outputs=["action"])
                )

            graph = oh.make_graph(
                nodes=nodes,
                name="docr_marl_actor",
                inputs=[
                    oh.make_tensor_value_info(
                        "obs", tp.FLOAT, [None] + list(self.input_shape)
                    )
                ],
                outputs=[
                    oh.make_tensor_value_info(
                        "action", tp.FLOAT, [None, out_features]
                    )
                ],
                initializer=[
                    onh.from_array(W1, name="W1"),
                    onh.from_array(b1, name="b1"),
                    onh.from_array(W2, name="W2"),
                    onh.from_array(b2, name="b2"),
                    onh.from_array(W3, name="W3"),
                    onh.from_array(b3, name="b3"),
                ],
            )
            model_proto = oh.make_model(
                graph,
                opset_imports=[oh.make_opsetid("", opset_version)],
            )
            model_proto.doc_string = str(self.metadata)
            onnx.save(model_proto, str(output_path))

            export_info = {
                "path": str(output_path),
                "opset_version": opset_version,
                "input_shape": self.input_shape,
                "metadata": self.metadata,
                "status": "onnx_exported",
                "backend": "onnx",
            }
            logger.info("Exported model to ONNX: %s", output_path)

        except ImportError:
            # ── Fallback: portable numpy .npz archive ──────────────────────
            # np.savez() silently appends a ".npz" suffix when the given
            # path doesn't already end in one (e.g. "model.onnx" is actually
            # written to "model.onnx.npz") — normalize output_path up front
            # so what we report/verify is the file that was actually written.
            npz_path = output_path if output_path.suffix == ".npz" else output_path.with_suffix(output_path.suffix + ".npz")
            np.savez(
                str(npz_path),
                W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3,
                apply_tanh=np.array(apply_tanh),
                input_shape=np.array(self.input_shape),
                metadata=np.array([str(self.metadata)]),
            )
            output_path = npz_path
            export_info = {
                "path": str(output_path),
                "opset_version": opset_version,
                "input_shape": self.input_shape,
                "metadata": self.metadata,
                "status": "npz_exported",
                "backend": "numpy",
                "note": (
                    "Install onnx+onnxruntime for true ONNX export; "
                    "InferenceEngine supports this npz format natively."
                ),
            }
            logger.info("Exported model to npz fallback: %s", output_path)

        if verify:
            export_info["verification"] = self._verify_export(output_path)

        return export_info

    def _verify_export(self, path: Path) -> dict:
        """Verify the exported artifact's output matches the real model's.

        Runs the same test input through (a) ``self.model`` directly and
        (b) the just-exported artifact (ONNX via onnxruntime, or the npz
        fallback reconstructed by hand), and confirms they agree — this is
        what would have caught weights being replaced by random noise.
        """
        test_input = np.random.default_rng(0).standard_normal((1, *self.input_shape)).astype(np.float32)
        original_output = np.asarray(self._forward(test_input[0])).astype(np.float64).ravel()

        finite = bool(np.all(np.isfinite(original_output)))

        exported_output = None
        matches_exported = None
        compare_error = None
        try:
            exported_output = self._run_exported_artifact(path, test_input[0])
            matches_exported = bool(
                np.allclose(original_output, exported_output, rtol=1e-4, atol=1e-5)
            )
        except Exception as exc:  # pragma: no cover - defensive
            compare_error = str(exc)

        status = "verified" if (finite and matches_exported) else "verification_failed"
        if compare_error is not None:
            status = "verification_error"

        result = {
            "test_input_shape": test_input.shape,
            "original_output": original_output.tolist(),
            "outputs_finite": finite,
            "exported_output": (
                exported_output.tolist() if exported_output is not None else None
            ),
            "matches_exported_artifact": matches_exported,
            "status": status,
        }
        if compare_error is not None:
            result["compare_error"] = compare_error
        return result

    def _run_exported_artifact(self, path: Path, x: np.ndarray) -> np.ndarray:
        """Load back the just-written artifact and run it on ``x``."""
        path = Path(path)
        # Dispatch on the artifact's actual format instead of trying
        # onnxruntime unconditionally and hoping for ImportError to signal
        # "this is really an npz" -- if onnxruntime is installed but onnx
        # (the authoring package) isn't, export() writes an .npz fallback
        # (see export()) and InferenceSession(".npz path") raises a
        # non-ImportError exception trying to parse it as a protobuf. That
        # only got caught by _verify_export's broad `except Exception`,
        # producing "verification_error" and never actually running the
        # comparison this method exists to make -- silently defeating
        # export verification under that (plausible) dependency combination.
        if path.suffix == ".onnx":
            import onnxruntime as ort

            sess = ort.InferenceSession(str(path))
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            x_in = np.asarray(x, dtype=np.float32).reshape(1, -1)
            out = sess.run([out_name], {in_name: x_in})[0][0]
            return np.asarray(out, dtype=np.float64).ravel()

        # npz fallback artifact: reconstruct the forward pass by hand.
        data = np.load(str(path), allow_pickle=True)
        W1, b1 = data["W1"].astype(np.float64), data["b1"].astype(np.float64)
        W2, b2 = data["W2"].astype(np.float64), data["b2"].astype(np.float64)
        W3, b3 = data["W3"].astype(np.float64), data["b3"].astype(np.float64)
        apply_tanh = bool(data["apply_tanh"])

        xv = np.asarray(x, dtype=np.float64).ravel()
        h1 = np.maximum(0.0, W1 @ xv + b1)
        h2 = np.maximum(0.0, W2 @ h1 + b2)
        out = W3 @ h2 + b3
        if apply_tanh:
            out = np.tanh(out)
        return out

    def get_model_summary(self) -> dict:
        """Return the real model's architecture summary."""
        arch = self._get_architecture()
        n_parameters = sum(
            arch[k].size for k in ("W1", "b1", "W2", "b2", "W3", "b3")
        )
        return {
            "input_shape": self.input_shape,
            "output_shape": (int(arch["W3"].shape[0]),),
            "hidden_sizes": (int(arch["W1"].shape[0]), int(arch["W2"].shape[0])),
            "n_parameters": int(n_parameters),
            "metadata": self.metadata,
            "export_backend": "onnx (if installed) or numpy-npz",
        }


# ── Inference Engine ────────────────────────────────────────────────────────

class InferenceEngine:
    """Inference engine for DOCR-MARL.

    Loads models exported by :class:`ONNXExporter` (either real ONNX or the
    numpy-npz fallback) and runs forward passes with latency measurement.

    Supports:
    - onnxruntime (fast, if installed)
    - Numpy MLP fallback (always available, sub-1 ms per call)
    - Batch inference
    - Warmup + statistical benchmarking
    """

    def __init__(
        self,
        model_path: str,
        config: InferenceConfig | None = None,
    ) -> None:
        self.model_path = model_path
        self.config = config or InferenceConfig()
        self.latency_history: list[float] = []
        self.memory_usage_mb: float = 0.0
        self._model: Callable | None = None
        self._backend: str = "unloaded"
        self._initialized: bool = False
        self._quantization_applied: bool = False

    # ── Initialisation ──────────────────────────────────────────────────────

    def initialize(self) -> bool:
        """Load model; try onnxruntime first, fall back to numpy npz."""
        try:
            self._model, self._backend = self._load_model()
            self._initialized = True
            logger.info(
                "InferenceEngine initialized with backend: %s", self._backend
            )
            return True
        except Exception as exc:
            logger.error("InferenceEngine init failed: %s", exc)
            return False

    def _load_model(self) -> tuple[Callable, str]:
        """Return (model_callable, backend_name)."""
        path = self.model_path
        quantize = self.config.use_quantization
        bits = self.config.quantization_bits

        # 1. Try onnxruntime
        try:
            import onnxruntime as ort

            session_path = path
            if quantize:
                # Quantize the ONNX graph's weight initializers in-place
                # (fake quantize -> dequantize, so the graph shape/ops are
                # unchanged) and run inference from the perturbed copy.
                quantized_path = self._quantize_onnx_model(path, bits)
                if quantized_path is not None:
                    session_path = quantized_path

            sess = ort.InferenceSession(session_path)
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name

            def _ort_infer(obs: np.ndarray) -> np.ndarray:
                x = obs.reshape(1, -1).astype(np.float32)
                return sess.run([out_name], {in_name: x})[0][0]

            self._quantization_applied = bool(quantize and session_path != path)
            return _ort_infer, "onnxruntime"
        except Exception as exc:
            logger.debug("onnxruntime load failed: %s", exc)

        # 2. Try numpy npz (fallback export format: real 2-hidden-layer MLP
        #    weights W1/b1/W2/b2/W3/b3, as written by ONNXExporter.export()).
        try:
            data = np.load(path, allow_pickle=True)
            W1 = data["W1"].astype(np.float64)
            b1 = data["b1"].astype(np.float64)
            W2 = data["W2"].astype(np.float64)
            b2 = data["b2"].astype(np.float64)
            W3 = data["W3"].astype(np.float64)
            b3 = data["b3"].astype(np.float64)
            apply_tanh = bool(data["apply_tanh"]) if "apply_tanh" in data else False

            if quantize:
                W1, b1 = fake_quantize(W1, bits), fake_quantize(b1, bits)
                W2, b2 = fake_quantize(W2, bits), fake_quantize(b2, bits)
                W3, b3 = fake_quantize(W3, bits), fake_quantize(b3, bits)

            def _npz_infer(obs: np.ndarray) -> np.ndarray:
                x = obs.ravel().astype(np.float64)
                h1 = np.maximum(0.0, W1 @ x + b1)
                h2 = np.maximum(0.0, W2 @ h1 + b2)
                out = W3 @ h2 + b3
                if apply_tanh:
                    out = np.tanh(out)
                return out.astype(np.float32)

            self._quantization_applied = bool(quantize)
            return _npz_infer, "numpy-npz"
        except Exception as exc:
            logger.debug("npz load failed: %s", exc)

        # 3. Synthetic fallback — always succeeds, useful for latency benchmarking
        in_features = int(np.prod(self.config.input_shape))
        rng = np.random.default_rng(0)
        W = rng.standard_normal((2, in_features)).astype(np.float32) * 0.01
        b = np.zeros(2, dtype=np.float32)

        if quantize:
            W = fake_quantize(W, bits)
            b = fake_quantize(b, bits)

        def _synth_infer(obs: np.ndarray) -> np.ndarray:
            x = obs.ravel().astype(np.float32)
            return W @ x + b

        logger.warning("Using synthetic fallback model (no real weights loaded)")
        self._quantization_applied = bool(quantize)
        return _synth_infer, "synthetic-fallback"

    def _quantize_onnx_model(self, path: str, bits: int) -> str | None:
        """Write a copy of the ONNX model at ``path`` with all float
        initializers passed through :func:`fake_quantize`. Returns the path
        to the quantized copy, or ``None`` if the ``onnx`` package isn't
        available or the model can't be parsed (caller falls back to the
        original, unquantized path in that case).
        """
        try:
            import onnx
            import onnx.numpy_helper as onh
        except ImportError:
            logger.debug("onnx package unavailable; cannot quantize ONNX initializers")
            return None

        try:
            model_proto = onnx.load(path)
            for init in model_proto.graph.initializer:
                arr = onh.to_array(init)
                if not np.issubdtype(arr.dtype, np.floating):
                    continue
                q_arr = fake_quantize(arr, bits)
                new_init = onh.from_array(q_arr, name=init.name)
                init.CopyFrom(new_init)

            quantized_path = str(Path(path).with_suffix(f".int{bits}.onnx"))
            onnx.save(model_proto, quantized_path)
            return quantized_path
        except Exception as exc:
            logger.debug("ONNX quantization failed, using unquantized model: %s", exc)
            return None

    # ── Inference ─────────────────────────────────────────────────────────

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """Run single inference.

        Parameters
        ----------
        observation : np.ndarray
            Shape ``(obs_dim,)`` for single agent or ``(n_agents, obs_dim)``
            for batched agents.

        Returns
        -------
        np.ndarray
            Shape ``(2,)`` with ``[delta_TMS, delta_PS]`` per agent.
        """
        if not self._initialized:
            self.initialize()

        t0 = time.perf_counter()

        if observation.ndim == 1:
            action = self._model(observation)
        else:
            action = np.stack([
                self._model(observation[i])
                for i in range(observation.shape[0])
            ])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.latency_history.append(elapsed_ms)
        return action

    def infer_batch(self, observations: list[np.ndarray]) -> list[np.ndarray]:
        """Batch inference for a list of observation arrays."""
        return [self.infer(obs) for obs in observations]

    # ── Benchmarking ──────────────────────────────────────────────────────

    def benchmark(self, n_runs: int | None = None) -> dict:
        """Benchmark inference latency with warmup.

        Parameters
        ----------
        n_runs : int | None
            Number of benchmark runs. Defaults to ``config.benchmark_runs``.

        Returns
        -------
        dict
            Statistics: mean_ms, std_ms, min_ms, max_ms, p50_ms, p99_ms,
            meets_target, target_ms, n_runs, backend.
        """
        if not self._initialized:
            self.initialize()

        n_runs = n_runs or self.config.benchmark_runs
        dummy = np.random.default_rng(0).standard_normal(self.config.input_shape).astype(np.float32)

        # Warmup (JIT / cache priming)
        for _ in range(self.config.warmup_runs):
            self.infer(dummy)

        self.latency_history = []
        for _ in range(n_runs):
            self.infer(dummy)

        arr = np.array(self.latency_history)
        return {
            "mean_ms": float(np.mean(arr)),
            "std_ms": float(np.std(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            # Worst-case (max), not mean: this is a hard real-time deadline
            # (module docstring: "<100 ms target"). Judging it on the mean
            # let a run with a handful of latency spikes past the target
            # report "meets target: YES" right next to its own printed
            # p99/max showing otherwise. Matches hils_bridge.py's
            # analogous GOOSE-latency check, which already uses max.
            "meets_target": bool(np.max(arr) < self.config.target_latency_ms),
            "target_ms": self.config.target_latency_ms,
            "n_runs": n_runs,
            "backend": self._backend,
        }

    def get_latency_report(self) -> str:
        """Generate a human-readable latency report."""
        if not self.latency_history:
            return "No inference data available."

        stats = self.benchmark()

        lines = [
            "=" * 60,
            "INFERENCE LATENCY REPORT",
            "=" * 60,
        ]
        if stats["backend"] == "synthetic-fallback":
            lines.append("  *** WARNING: SYNTHETIC FALLBACK MODEL — no real weights were")
            lines.append("  *** loaded (ONNX/npz unavailable). Numbers below time a random")
            lines.append("  *** untrained model, not the actual DOCR-MARL policy.")
        if self._quantization_applied:
            lines.append(
                f"  Quantization:       ENABLED ({self.config.quantization_bits}-bit "
                "fake-quantize/dequantize applied to weights)"
            )
        lines += [
            f"  Backend:            {stats['backend']}",
            f"  Target latency:     {stats['target_ms']:.1f} ms",
            f"  Mean latency:       {stats['mean_ms']:.2f} ms",
            f"  Std deviation:      {stats['std_ms']:.2f} ms",
            f"  P99 latency:        {stats['p99_ms']:.2f} ms",
            f"  Min/Max:            {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms",
            (
                "  Meets target:       YES ✓"
                if stats["meets_target"]
                else "  Meets target:       NO ✗"
            ),
            "=" * 60,
        ]

        return "\n".join(lines)


# ── Deployment Packaging ──────────────────────────────────────────────────

class RelayDeploymentPack:
    """Complete deployment package for a single relay agent.

    Includes:
    - ONNX model path
    - Configuration file
    - Metadata
    - Verification checksums
    """

    def __init__(
        self,
        relay_id: int,
        model_path: str,
        config: dict,
    ) -> None:
        self.relay_id = relay_id
        self.model_path = model_path
        self.config = config
        self.packaged = False

    def package(self, output_dir: str) -> str:
        """Create deployment package.

        Returns
        -------
        str
            Path to the generated package JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pack = {
            "relay_id": self.relay_id,
            "model_path": self.model_path,
            "config": self.config,
            "checksum": self._compute_checksum(),
            "version": "1.0.0",
        }

        pack_path = output_dir / f"relay_{self.relay_id}_pack.json"
        with open(pack_path, "w", encoding="utf-8") as f:
            json.dump(pack, f, indent=2)

        self.packaged = True
        return str(pack_path)

    def _compute_checksum(self) -> str:
        """Compute a placeholder package checksum."""
        # In production, compute SHA-256 over model weights + config
        data = f"{self.relay_id}:{self.model_path}:{json.dumps(self.config, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


def export_marl_deployment(
    actor_network: Callable,
    n_relays: int,
    output_dir: str = "deployment/packages",
    config: InferenceConfig | None = None,
) -> list[str]:
    """Export complete MARL deployment for all relays.

    Parameters
    ----------
    actor_network : Callable
        Actor network callable.
    n_relays : int
        Number of relay agents.
    output_dir : str
        Output directory for packages.
    config : InferenceConfig | None
        Inference configuration.

    Returns
    -------
    list[str]
        Paths to relay deployment packages.
    """
    config = config or InferenceConfig()
    exporter = ONNXExporter(actor_network, input_shape=config.input_shape)

    # Export shared model
    model_path = f"{output_dir}/shared_actor.onnx"
    exporter.export(model_path)

    # Create per-relay packages
    packages: list[str] = []
    for relay_id in range(n_relays):
        relay_config = {
            "relay_id": relay_id,
            "n_relays": n_relays,
            "inference_config": config.__dict__,
        }
        pack = RelayDeploymentPack(relay_id, model_path, relay_config)
        pack_path = pack.package(output_dir)
        packages.append(pack_path)

    return packages