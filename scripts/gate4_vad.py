#!/usr/bin/env python3
"""Gate 4: VAD auto-stop diagnostic (real SpeechRecognizer, real mic, real Silero).

Phase 1 (control, no human): 6s silence - expect NO transcript, near-zero probs.
Phase 2 (human): speak the reference sentence ONCE at normal volume, then STAY
SILENT. The real SpeechRecognizer's VAD should auto-stop ~1.5s (vad.silence_
timeout) after your last word and emit a transcript.

Run with --control-only to execute just Phase 1 (no speech needed).
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, r"D:\work - personal\Hexa Ai\hexa")
from hexa.config import load_config, get_config

load_config()
cfg = get_config()

from hexa.stt import SpeechRecognizer, STTConfig, VADConfig

REFERENCE = "Hexa is a local voice assistant that runs completely offline"
THRESHOLD = cfg.stt.vad_threshold


def levenshtein(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.lower().replace(",", "").replace(".", "").split()
    hyp = hypothesis.lower().replace(",", "").replace(".", "").split()
    return 100.0 * levenshtein(ref, hyp) / len(ref) if ref else 0.0


class Gate4:
    def __init__(self) -> None:
        self.events: list[tuple[float, float]] = []  # (t, prob) per chunk
        self.transcript: str | None = None
        self.t_transcript: float | None = None
        self.t_process_start: float | None = None  # when auto-stop fired _process_speech
        self.rec = SpeechRecognizer(
            stt_config=STTConfig(
                model=cfg.stt.model, device=cfg.stt.device,
                compute_type=cfg.stt.compute_type, language=cfg.stt.language,
                beam_size=cfg.stt.beam_size, vad_threshold=cfg.stt.vad_threshold,
            ),
            vad_config=VADConfig(
                enabled=cfg.vad.enabled, silence_timeout=cfg.vad.silence_timeout,
                min_speech_duration=cfg.vad.min_speech_duration,
            ),
            on_transcript=self._on_transcript,
        )
        # Instrument the real VAD (records probs; does not alter behavior)
        orig = self.rec._vad.get_speech_prob

        def prob_wrapper(chunk, sample_rate=16000):
            p = orig(chunk, sample_rate)
            self.events.append((time.perf_counter(), p))
            return p

        self.rec._vad.get_speech_prob = prob_wrapper

        # Instrument the auto-stop moment (when _process_speech is entered)
        orig_process = self.rec._process_speech

        def process_wrapper():
            self.t_process_start = time.perf_counter()
            orig_process()

        self.rec._process_speech = process_wrapper

    def _on_transcript(self, text: str) -> None:
        self.transcript = text
        self.t_transcript = time.perf_counter()
        print(f"  >>> TRANSCRIPT FIRED: '{text}'")

    def prob_summary(self) -> str:
        probs = [p for _, p in self.events]
        if not probs:
            return "no chunks processed"
        ts = [t for t, _ in self.events]
        span = ts[-1] - ts[0]
        gaps = np.diff(ts)
        return (f"n={len(probs)} span={span:.2f}s rate={len(probs)/span:.1f}/s "
                f"gap_med={np.median(gaps)*1000:.0f}ms gap_max={gaps.max()*1000:.0f}ms "
                f"max_prob={max(probs):.3f} mean={np.mean(probs):.3f} "
                f"frac>thr={np.mean([p >= THRESHOLD for p in probs]):.2f}")

    def speech_window(self) -> tuple[float, float] | None:
        """(speech_start, speech_end) from instrumented probs."""
        speech_ts = [t for t, p in self.events if p >= THRESHOLD]
        if not speech_ts:
            return None
        return speech_ts[0], speech_ts[-1]


def main() -> None:
    control_only = "--control-only" in sys.argv
    speech_only = "--speech-only" in sys.argv
    print("=" * 70)
    print("GATE 4: VAD AUTO-STOP DIAGNOSTIC")
    print("=" * 70)
    g = Gate4()
    # The shared stream's lifecycle belongs to the pipeline owner; voice.py now
    # starts it (Gate 4 root cause fix). The diagnostic mirrors that here.
    from hexa.audio import get_audio_stream
    stream = get_audio_stream()
    if not stream.is_running():
        stream.start()
    g.rec.start()
    print(f"Recognizer started (threshold={THRESHOLD}, "
          f"silence_timeout={cfg.vad.silence_timeout}s, "
          f"min_speech={cfg.vad.min_speech_duration}s)")

    # ---- Phase 1: control (silence) ----
    if not speech_only:
        print("\nPHASE 1: CONTROL - 6s of silence, please stay quiet...")
        import faulthandler
        faulthandler.enable()  # crash diagnosis only; silent unless a hard crash
        for i in range(6):
            time.sleep(1.0)
            qsz = g.rec._audio_queue.qsize() if g.rec._audio_queue else -1
            alive = g.rec._thread.is_alive() if g.rec._thread else False
            print(f"  t={i+1}s: vad_events={len(g.events)} rec_thread_alive={alive} "
                  f"queue_size={qsz} stream_running={stream.is_running()}")
        tr = g.transcript
        print(f"  VAD probs: {g.prob_summary()}")
        print(f"  transcript: {tr!r}")
        if tr:
            print("  FAIL: false trigger in silence")
        else:
            print("  OK: no transcript in silence")

    if control_only:
        g.rec.stop()
        from hexa.audio import shutdown_audio_stream
        shutdown_audio_stream()
        print("\nCONTROL PHASE COMPLETE")
        return

    # ---- Phase 2: human speech + auto-stop ----
    print("\nPHASE 2: SPEAK NOW (once, normal volume), then STAY SILENT:")
    print(f"  '{REFERENCE}'")
    t_start = time.perf_counter()
    g.events.clear()
    g.transcript = None
    g.t_process_start = None

    deadline = t_start + 30.0
    while time.perf_counter() < deadline and g.transcript is None:
        time.sleep(0.2)

    g.rec.stop()
    from hexa.audio import shutdown_audio_stream
    shutdown_audio_stream()

    if g.transcript is None:
        print(f"\nFAIL: no transcript within {time.perf_counter()-t_start:.0f}s")
        print(f"  VAD probs: {g.prob_summary()}")
        sys.exit(1)

    window = g.speech_window()
    print(f"\n  VAD probs (speech phase): {g.prob_summary()}")
    if window:
        speech_ts = [t for t, p in g.events if p >= THRESHOLD and window[0] <= t <= window[1]]
        gaps = np.diff(speech_ts) if len(speech_ts) > 1 else np.array([0.0])
        print(f"  speech span: {window[0]-t_start:.2f}s -> {window[1]-t_start:.2f}s "
              f"({window[1]-window[0]:.2f}s of speech chunks), "
              f"max gap WITHIN speech: {gaps.max()*1000:.0f}ms")
    if g.t_process_start is not None and window:
        stop_delay = g.t_process_start - window[1]
        print(f"  VAD AUTO-STOP latency (last speech chunk -> finalize): {stop_delay:.2f}s "
              f"[target ~{cfg.vad.silence_timeout}s]")
        print(f"  transcription after auto-stop: {g.t_transcript - g.t_process_start:.2f}s "
              f"(CPU Whisper cost; pipeline latency is Gate 7's business)")
    print(f"  transcript: '{g.transcript}'")
    w = wer(REFERENCE, g.transcript)
    print(f"  WER vs reference: {w:.1f}%")

    ok_stop = window and g.t_process_start is not None and 1.0 <= (g.t_process_start - window[1]) <= 2.0
    ok_text = w <= 40.0
    print(f"\n  VAD auto-stop {'OK' if ok_stop else 'FAIL'} (target ~{cfg.vad.silence_timeout}s)")
    print(f"  transcript {'OK' if ok_text else 'FAIL'} (WER <= 40%)")
    print("\nGATE 4:", "PASS" if (ok_stop and ok_text) else "CHECK OUTPUT ABOVE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import logging
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("gate4").exception("Gate 4 failed")
        sys.exit(1)
