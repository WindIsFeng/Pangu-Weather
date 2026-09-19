"""Bounded ONNX session cache. Requested CUDA must actually initialize."""
import gc
from collections import OrderedDict

from .era5 import validate_state


class Runtime:
    def __init__(self, config, logger):
        import onnxruntime as ort
        self.ort, self.config, self.logger = ort, config, logger
        self.sessions = OrderedDict()
        self.actual_providers = {}
        if config.device == "cuda":
            ort.preload_dlls(directory="")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("CUDAExecutionProvider unavailable; install the GPU runtime or explicitly select cpu")

    def session(self, step):
        if step in self.sessions:
            self.sessions.move_to_end(step)
            return self.sessions[step]
        while len(self.sessions) >= self.config.max_sessions:
            _, old = self.sessions.popitem(last=False)
            del old
            gc.collect()
        options = self.ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        options.intra_op_num_threads = self.config.threads
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if self.config.device == "cuda":
            providers = [("CUDAExecutionProvider", {"device_id": self.config.device_id,
                                                    "arena_extend_strategy": "kSameAsRequested"})]
        path = self.config.model_dir / f"pangu_weather_{step}.onnx"
        self.logger.info("Loading %sh model on %s", step, self.config.device)
        session = self.ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if self.config.device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError("CUDA initialization failed; refusing silent CPU fallback")
        session.disable_fallback()
        if {v.name for v in session.get_inputs()} != {"input", "input_surface"}:
            raise ValueError(f"unexpected ONNX input interface: {path}")
        if {v.name for v in session.get_outputs()} != {"output", "output_surface"}:
            raise ValueError(f"unexpected ONNX output interface: {path}")
        self.sessions[step] = session
        self.actual_providers[str(step)] = session.get_providers()
        return session

    def run(self, step, state):
        validate_state(state)
        upper, surface = state
        result = self.session(step).run(["output", "output_surface"],
                                        {"input": upper, "input_surface": surface})
        validate_state(result)
        return tuple(result)

    def close(self):
        self.sessions.clear()
        gc.collect()
