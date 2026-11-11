"""Thin synchronous wrapper around the official LeapTalk streaming engine.

Model loading and per-turn preparation reuse `leaptalk_stream.LeapTalkEngine` verbatim, and the
per-chunk path mirrors `LeapTalkEngine._generate_chunk_sync` / `inference.py` so that timings
reflect the real pipeline. Only instrumentation (CUDA-event stage breakdown) is added.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402

from leaptalk_stream import LeapTalkEngine, LeapTalkStreamSettings, _TurnState  # noqa: E402

MODELS = ROOT / "models"
DEFAULT_IMAGE = ROOT / "assets" / "photo.jpg"


def make_settings(lite: bool, compile_mode: str, steps: int = 1, cuda: str = "0",
                  fp8: bool = False, fp8_scope: str = "default", fp8_dual: bool = False) -> LeapTalkStreamSettings:
    runtime = ROOT / "bench" / "runtime"
    return LeapTalkStreamSettings(
        ckpt_dir=str(MODELS / "SoulX-FlashHead-1_3B"),
        wav2vec_dir=str(MODELS / "wav2vec2-base-960h"),
        lora_dir=str(MODELS / "leaptalk"),
        audio_proj=str(MODELS / "leaptalk"),
        compile=compile_mode,
        steps=int(steps),
        lite=bool(lite),
        fp8=bool(fp8),
        fp8_scope=fp8_scope,
        fp8_dual=bool(fp8_dual),
        cuda_visible_devices=cuda,
        output_sample_rate=16000,
        root_dir=ROOT,
        web_dir=runtime,
        audio_dir=runtime / "audio",
        video_dir=runtime / "video",
    )


def crop_to_face(src: str | Path, ratio: float, out_path: str | Path, size: int = 512) -> dict:
    """Square crop around the largest face, then resize to size x size.

    Same framing as upstream flash_head.utils.facecrop.get_scaled_bbox: side = face width x ratio,
    55 % of it above the face centre. Upstream detects with mediapipe `solutions`, which the pinned
    mediapipe 0.10.35 no longer has, so this uses OpenCV's bundled Haar face detector instead.
    The box is shifted (not clipped) to stay inside the image, so the crop stays square.
    Returns {"found": bool, "box": [x1, y1, x2, y2] | None}; without a face, nothing is written.
    """
    import cv2
    from PIL import Image

    img = Image.open(src).convert("RGB")
    w, h = img.size
    scale = min(1.0, 800 / max(w, h))  # detect on a smaller copy; faces in portraits are large
    small = np.asarray(img.resize((max(1, round(w * scale)), max(1, round(h * scale)))))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    m = max(24, min(gray.shape) // 10)
    faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(m, m))
    if len(faces) == 0:
        return {"found": False, "box": None}
    fx, fy, fw, fh = (v / scale for v in max(faces, key=lambda f: f[2] * f[3]))
    side = min(fw * float(ratio), w, h)
    cx, cy = fx + fw / 2, fy + fh / 2
    x1 = min(max(0.0, cx - side / 2), w - side)
    y1 = min(max(0.0, cy - side * 0.55), h - side)
    box = [int(round(x1)), int(round(y1)), int(round(x1 + side)), int(round(y1 + side))]
    img.crop(box).resize((size, size), Image.LANCZOS).save(out_path)
    return {"found": True, "box": box}


class LeapTalkRunner:
    def __init__(self, lite: bool = True, compile_mode: str = "off", steps: int = 1, cuda: str = "0",
                 fp8: bool = False, fp8_scope: str = "default", fp8_dual: bool = False) -> None:
        self.settings = make_settings(lite, compile_mode, steps, cuda, fp8, fp8_scope, fp8_dual)
        self.engine = LeapTalkEngine(self.settings)
        t0 = time.perf_counter()
        self.engine._load_sync()
        self.load_seconds = time.perf_counter() - t0
        self.fp8_summary = getattr(self.engine, "fp8_summary", None)  # None when FP8 is off
        self.torch = self.engine._helpers["torch"]
        self.h = self.engine._helpers
        self.pipe = self.engine._pipeline
        self.sp = self.engine._sp
        self.dtype = self.engine._amp_dtype
        self.sr = self.settings.sample_rate
        self.slice_frames = int(self.sp.slice_len)  # new frames per chunk (28, any fps)

    # Precision switching: self.pipe is engine._pipeline, and every per-chunk call reads self.pipe.model at
    # call time, so swapping engine._pipeline.model is enough.
    @property
    def precisions(self) -> list[str]:
        keys = list(self.engine.models)
        return sorted(keys, key=lambda k: (k != "bf16", k))

    @property
    def precision(self) -> str:
        return self.engine.precision

    def set_precision(self, name: str) -> None:
        self.engine.set_precision(name)

    # fps can change per turn (prepare); a chunk always has 28 frames, so its audio length follows fps
    @property
    def fps(self) -> int:
        return int(self.settings.fps)

    @property
    def slice_samples(self) -> int:
        return int(round(self.slice_frames * self.sr / self.fps))  # 17920 @ 25 fps, 16 kHz

    @property
    def slice_seconds(self) -> float:
        return self.slice_frames / self.fps  # 1.12 s @ 25 fps

    # ------------------------------------------------------------------ helpers
    def _sync(self) -> None:
        self.torch.cuda.synchronize()

    def _event(self):
        e = self.torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def load_audio(self, path: str) -> np.ndarray:
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        return audio.astype(np.float32)

    def split_slices(self, audio: np.ndarray) -> list[np.ndarray]:
        rem = len(audio) % self.slice_samples
        if rem:
            audio = np.concatenate([audio, np.zeros(self.slice_samples - rem, dtype=np.float32)])
        return list(audio.reshape(-1, self.slice_samples))

    def prepare(self, image_path: str | Path = DEFAULT_IMAGE, size: int | None = None,
                fps: int | None = None) -> tuple[_TurnState, float]:
        """Encode the reference image; `size` (e.g. 256 or 512) sets the square output resolution.

        `fps` sets the generation frame rate (the model was trained at 25). Only the audio-feature
        rate changes; the DiT input shapes stay the same, so compiled graphs are reused.
        """
        if size is not None:
            size = int(size)
            if size % 16:
                raise ValueError(f"size must be a multiple of 16, got {size}")
            self.settings.height = self.settings.width = size
        if fps is not None:
            self.settings.fps = int(fps)
            self.sp.tgt_fps = int(fps)
        self._sync()
        t0 = time.perf_counter()
        state = self.engine._prepare_turn_sync(str(image_path))
        self._sync()
        return state, time.perf_counter() - t0

    def _history_and_frames(self, state, x_final, decoded, drop_overlap: bool):
        s = self.settings
        if s.history_update_mode == "roundtrip":
            state.latent_motion_frames = self.h["encode_motion"](
                self.pipe,
                decoded_video_cthw=decoded,
                motion_frames_num=self.sp.motion_frames_num,
                device=s.device,
                dtype=self.dtype,
            ).unsqueeze(0)
        else:
            state.latent_motion_frames = self.h["motion_tail"](
                x_final, motion_frames_latent_num=self.sp.motion_frames_latent_num
            ).unsqueeze(0)
        state.clamp_latent_len = int(state.latent_motion_frames.shape[2])
        if drop_overlap:
            decoded = decoded[:, self.sp.motion_frames_num:]
        return ((decoded + 1.0) / 2.0).permute(1, 2, 3, 0).clamp(0.0, 1.0).mul(255.0).contiguous()

    def _denoise_decode(self, state, audio_ctx, ev):
        s = self.settings
        x_final = self.h["bridge_sample"](
            self.pipe,
            scheduler=self.engine._scheduler,
            ref_latent=state.x0,
            audio_context=audio_ctx,
            guidance_scale=s.guidance_scale,
            latent_motion_frames=state.latent_motion_frames,
            clamp_latent_len=state.clamp_latent_len,
            device=s.device,
            dtype=self.dtype,
        )
        ev.append(self._event())
        decoded = self.h["decode"](self.pipe, x_final)
        ev.append(self._event())
        decoded = self.h["color"](self.pipe, decoded)
        return x_final, decoded

    # ------------------------------------------------------------------ stream path
    def stream_chunk(self, state: _TurnState, slice16k: np.ndarray) -> tuple[np.ndarray, dict]:
        """One streaming chunk exactly like LeapTalkEngine._generate_chunk_sync.

        Timed from "audio slice available" to "uint8 frames on host" (includes the 8 s wav2vec
        ring-buffer re-encode, DiT denoise, VAE decode, colour correction, history re-encode, D2H).
        """
        torch = self.torch
        s = self.settings
        self._sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            state.audio_deque.extend(slice16k.tolist())
            audio_cache = np.array(state.audio_deque, dtype=np.float32)
            ev = [self._event()]
            emb = self.pipe.preprocess_audio(audio_cache, sr=s.sample_rate, fps=s.fps)
            emb = emb.to(device=s.device, dtype=self.dtype)
            end_idx = self.sp.cached_audio_duration * s.fps
            ctx = self.h["audio_context"](emb, start_idx=end_idx - self.sp.frame_num, end_idx=end_idx,
                                          device=s.device, dtype=self.dtype)
            ev.append(self._event())
            x_final, decoded = self._denoise_decode(state, ctx, ev)
            ev.append(self._event())
            video = self._history_and_frames(state, x_final, decoded, drop_overlap=True)
            ev.append(self._event())
            frames = video.to(torch.uint8).cpu().numpy()
        t1 = time.perf_counter()
        return frames, self._stage_dict(ev, t1 - t0)

    def _stage_dict(self, ev, total):
        names = ["audio_enc", "denoise", "decode", "color+hist"]
        d = {n: ev[i].elapsed_time(ev[i + 1]) / 1000.0 for i, n in enumerate(names)}
        d["total"] = total
        return d

    # ------------------------------------------------------------------ non-stream path
    def generate_full(self, state: _TurnState, audio: np.ndarray) -> tuple[np.ndarray, dict]:
        """Offline path = inference.py --audio_encode_mode once (wav2vec over whole clip once)."""
        torch = self.torch
        s = self.settings
        fn, sl = self.sp.frame_num, self.slice_frames
        # pad to whole chunks, counted in frames: at 15/30 fps a frame is a fractional number of samples
        need = math.ceil(len(audio) * s.fps / s.sample_rate)
        n_target = 1 + max(0, math.ceil((need - fn) / sl))
        target = math.ceil((fn + (n_target - 1) * sl) * s.sample_rate / s.fps) + 1  # +1: wav2vec floors len*fps/sr
        audio = np.concatenate([audio, np.zeros(max(0, target - len(audio)), dtype=np.float32)])

        self._sync()
        t0 = time.perf_counter()
        chunk_times = []
        out = []
        with torch.no_grad():
            emb = self.pipe.preprocess_audio(audio, sr=s.sample_rate, fps=s.fps).to(device=s.device, dtype=self.dtype)
            total_frames = int(emb.shape[0])
            ctx_all = self.h["audio_context"](emb, start_idx=0, end_idx=total_frames, device=s.device, dtype=self.dtype)
            self._sync()
            t_audio = time.perf_counter() - t0
            n_chunks = 1 + max(0, (total_frames - fn) // sl)
            for ci in range(n_chunks):
                tc = time.perf_counter()
                start = ci * sl
                ctx = ctx_all[:, start:min(start + fn, total_frames)].contiguous()
                ev = []
                x_final, decoded = self._denoise_decode(state, ctx, ev)
                video = self._history_and_frames(state, x_final, decoded, drop_overlap=ci != 0)
                out.append(video.to(torch.uint8).cpu().numpy())
                chunk_times.append(time.perf_counter() - tc)
        frames = np.concatenate(out, axis=0)
        total = time.perf_counter() - t0
        return frames, {"total": total, "audio_enc": t_audio, "chunks": chunk_times}

    # ------------------------------------------------------------------ encoding
    def encode_fmp4_chunk(self, frames: np.ndarray, slice16k: np.ndarray, out_path: Path, fast: bool = False) -> float:
        """Encode one chunk as fragmented MP4 (h264+aac); returns seconds.

        fast=False: exactly the web demo (LeapTalkEngine._write_av_chunk, x264 default preset).
        fast=True : same two-step encode+mux, but x264 ultrafast/zerolatency.
        """
        t0 = time.perf_counter()
        pcm = (np.clip(slice16k, -1, 1) * 32767).astype("<i2").tobytes()
        out_path = Path(out_path)
        if not fast:
            self.engine._write_av_chunk(frames, pcm, out_path)
            return time.perf_counter() - t0
        from leaptalk_stream import _write_wav
        tmp_v = out_path.with_name(out_path.stem + "_v.mp4")
        tmp_a = out_path.with_name(out_path.stem + ".wav")
        with imageio.get_writer(str(tmp_v), format="mp4", mode="I", fps=self.fps, codec="h264",
                                ffmpeg_params=["-bf", "0", "-preset", "ultrafast", "-tune", "zerolatency"]) as w:
            for fr in frames:
                w.append_data(fr)
        _write_wav(tmp_a, pcm, sample_rate=self.sr)
        dur = frames.shape[0] / self.fps
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(tmp_v), "-i", str(tmp_a), "-c:v", "copy",
             "-c:a", "aac", "-b:a", "96k", "-af", "apad", "-t", f"{dur:.6f}", "-movflags",
             "+empty_moov+default_base_moof+frag_keyframe+omit_tfhd_offset", str(out_path), "-y"],
            check=True,
        )
        tmp_v.unlink(missing_ok=True)
        tmp_a.unlink(missing_ok=True)
        return time.perf_counter() - t0

    def save_mp4(self, frames: np.ndarray, audio16k: np.ndarray, out_path: Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as td:
            vid = Path(td) / "v.mp4"
            wav = Path(td) / "a.wav"
            with imageio.get_writer(str(vid), fps=self.fps, codec="libx264", quality=8,
                                    macro_block_size=1, ffmpeg_log_level="error") as w:
                for fr in frames:
                    w.append_data(fr)
            import soundfile as sf
            sf.write(str(wav), audio16k, self.sr)
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(vid), "-i", str(wav),
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
                 str(out_path)],
                check=True,
            )
