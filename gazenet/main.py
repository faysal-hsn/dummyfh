# --- GazeNet runner ----------------------------------------------------------
import os
import logging
import numpy as np
import cv2

logger = logging.getLogger("GazeNet")

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import tensorrt as trt
    import pycuda.driver as cuda  
    import pycuda.autoinit        
except Exception:
    trt = None
    cuda = None


def _prep_224_bgr(img):
    if img is None or img.size == 0:
        return None
    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    arr = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, [0,1]
    return np.transpose(arr, (2, 0, 1))[None, ...]     # (1,3,224,224)


def _prep_facegrid(fg_flat):
    if fg_flat is None:
        return np.zeros((1, 625), dtype=np.float32), np.zeros((1, 1, 625, 1), dtype=np.float32)
    fg = np.asarray(fg_flat, dtype=np.float32).reshape(-1)
    if fg.size != 625:
        fg = np.zeros((625,), dtype=np.float32)
    return fg.reshape(1, 625), fg.reshape(1, 1, 625, 1)


class _TRTRunner:
    def __init__(self, engine_path):
        assert trt is not None and cuda is not None, "TensorRT/PyCUDA not available"
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.bindings = []
        self.host_inputs, self.dev_inputs = {}, {}
        self.host_outputs, self.dev_outputs = {}, {}

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = list(self.engine.get_binding_shape(i))
            if shape and shape[0] == -1:
                shape[0] = 1
            n = int(np.prod(shape)) if shape else 1
            h = cuda.pagelocked_empty(n, dtype)
            d = cuda.mem_alloc(h.nbytes)
            self.bindings.append(int(d))
            if self.engine.binding_is_input(i):
                self.host_inputs[name] = (h, shape, i)
                self.dev_inputs[name] = d
            else:
                self.host_outputs[name] = (h, shape, i)
                self.dev_outputs[name] = d

    def infer(self, inputs: dict):
        for name, data in inputs.items():
            h, shape, idx = self.host_inputs[name]
            np.copyto(h.reshape(shape), data.astype(h.dtype))
            cuda.memcpy_htod_async(self.dev_inputs[name], h, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        outputs = {}
        for name, (h, shape, idx) in self.host_outputs.items():
            cuda.memcpy_dtoh_async(h, self.dev_outputs[name], self.stream)
            outputs[name] = h.reshape(shape)
        self.stream.synchronize()
        return outputs


class GazeNet:
    """
    Unified runner for TAO GazeNet exported to TensorRT (.engine/.plan) or ONNX (.onnx).
    Place the artifact in ./gazenet/model/.
    """
    def __init__(self, model_dir="./gazenet/model"):
        self.model_dir = model_dir
        self.backend = None
        self.sess = None
        self.trt_runner = None

        engine, onnx, tlt = None, None, os.path.join(model_dir, "model.tlt")
        for fn in os.listdir(model_dir):
            low = fn.lower()
            if low.endswith((".engine", ".plan")):
                engine = os.path.join(model_dir, fn)
                break
        if engine is None:
            for fn in os.listdir(model_dir):
                if fn.lower().endswith(".onnx"):
                    onnx = os.path.join(model_dir, fn)
                    break

        if engine and trt is not None:
            logger.info("GazeNet: using TensorRT: %s", engine)
            self.backend = "trt"
            self.trt_runner = _TRTRunner(engine)
            self.input_names = list(self.trt_runner.host_inputs.keys())
            self.output_names = list(self.trt_runner.host_outputs.keys())
        elif onnx and ort is not None:
            logger.info("GazeNet: using ONNXRuntime: %s", onnx)
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in (ort.get_all_providers() if ort else []) else ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(onnx, providers=providers)
            self.backend = "onnx"
            self.input_names = [i.name for i in self.sess.get_inputs()]
            self.output_names = [o.name for o in self.sess.get_outputs()]
        else:
            if os.path.exists(tlt):
                raise RuntimeError(
                    "Found model.tlt but no runnable artifact. Convert the TAO model to .engine or .onnx and place it in ./gazenet/model."
                )
            raise RuntimeError("No model artifact found in ./gazenet/model (.engine/.onnx/.tlt).")

    def _shape_map_inputs(self, face, left, right, fg2d, fg_nchw):
        mapped = {}
        # helper to get static shape per input name
        def get_shape(name):
            if self.backend == "onnx":
                node = next(i for i in self.sess.get_inputs() if i.name == name)
                return tuple(int(s) if isinstance(s, int) else 1 for s in node.shape)
            else:
                return tuple(self.trt_runner.host_inputs[name][1])

        used_224 = 0
        three = [face, left, right]
        for name in self.input_names:
            shp = get_shape(name)
            if len(shp) == 4 and shp[1:] == (3, 224, 224) and used_224 < 3:
                mapped[name] = three[used_224]
                used_224 += 1
            elif shp == (1, 625) or (len(shp) == 2 and shp[-1] == 625):
                mapped[name] = fg2d
            elif shp == (1, 1, 625, 1):
                mapped[name] = fg_nchw
            else:
                # fallback
                mapped[name] = fg2d
        return mapped

    def predict_on_person(self, person):
        """Attach yaw/pitch + 2D direction to person.face."""
        f = getattr(person, "face", None)
        if f is None:
            return person

        face = _prep_224_bgr(getattr(f, "crop", None))
        left = _prep_224_bgr(getattr(f, "left_eye", None))
        right = _prep_224_bgr(getattr(f, "right_eye", None))
        if face is None or left is None or right is None:
            return person

        fg2d, fg_nchw = _prep_facegrid(getattr(f, "facegrid", None))
        inputs = self._shape_map_inputs(face, left, right, fg2d, fg_nchw)

        if self.backend == "onnx":
            out = self.sess.run(self.output_names, inputs)[0]
        else:
            out = list(self.trt_runner.infer(inputs).values())[0]

        out = np.asarray(out)
        if out.ndim == 2 and out.shape[1] == 2:
            yaw, pitch = float(out[0, 0]), float(out[0, 1])
        elif out.ndim == 2 and out.shape[1] == 3:
            gx, gy, gz = out[0].astype(np.float32)
            yaw = float(np.arctan2(gx, gz))
            pitch = float(np.arctan2(-gy, np.sqrt(gx * gx + gz * gz)))
        else:
            yaw, pitch = float(out.flatten()[0]), float(out.flatten()[1])

        person.face.gaze_yaw = yaw
        person.face.gaze_pitch = pitch

        dx = np.sin(yaw) * np.cos(pitch)
        dy = -np.sin(pitch)
        n = (dx * dx + dy * dy) ** 0.5 + 1e-6
        person.face.gaze_vec2d = (dx / n, dy / n)
        return person

    @staticmethod
    def draw_gaze(frame_bgr, person, length=150, thickness=2, color=(0, 255, 0)):
        f = getattr(person, "face", None)
        if f is None or not hasattr(f, "gaze_vec2d"):
            return
        x1, y1, x2, y2 = f.bbox
        cx, cy = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)
        dx, dy = f.gaze_vec2d
        end = (int(cx + length * dx), int(cy + length * dy))
        cv2.arrowedLine(frame_bgr, (cx, cy), end, color, thickness, tipLength=0.25)
