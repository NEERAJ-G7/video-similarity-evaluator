"""
Video-to-Text Similarity Evaluation System
==========================================
Pipeline:
  1. Extract audio      → Whisper transcript
  2. Extract frames     → Key frames (interval + scene change)
  3. Analyze frames     → OCR text + Claude Vision description
  4. Auto-gen reference → Optional: AI generates reference from transcript topic
  5. Merge content      → transcript + frame text = full content
  6. Evaluate           → 3 similarity methods vs reference samples
  7. Report             → Terminal summary + professional HTML dashboard

Usage:
    python main.py --video video.mp4 --refs samples/
    python main.py --video video.mp4 --refs samples/ --auto-ref
    python main.py --video video.mp4 --refs samples/ --no-vision
"""

import argparse
import os
import sys
import time
from pathlib import Path

from utils.transcriber     import transcribe_video
from utils.frame_extractor import extract_key_frames
from utils.frame_analyzer  import analyze_all_frames, build_frame_text_summary
from utils.evaluator       import evaluate_all
from utils.report          import generate_report, print_summary


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_references(ref_args: list) -> dict:
    references = {}
    for ref in ref_args:
        path = Path(ref)
        if path.is_dir():
            txt_files = sorted(path.glob("*.txt"))
            if not txt_files:
                print(f"  [!] No .txt files found in: {path}")
            for f in txt_files:
                references[f.stem] = f.read_text(encoding="utf-8").strip()
                print(f"  ✔ Loaded reference: {f.name}")
        elif path.is_file() and path.suffix == ".txt":
            references[path.stem] = path.read_text(encoding="utf-8").strip()
            print(f"  ✔ Loaded reference: {path.name}")
        else:
            print(f"  [!] Skipping: {ref}")
    return references


def auto_generate_reference(transcript: str) -> str:
    """
    Use Claude API to auto-generate a reference text from the transcript.
    Extracts the topic and generates an ideal reference answer.
    Falls back to transcript itself if API not available.
    """
    try:
        import anthropic
    except ImportError:
        print("  [!] anthropic not installed — using transcript as reference.")
        return transcript

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  [!] ANTHROPIC_API_KEY not set — using transcript as reference.")
        return transcript

    try:
        client = anthropic.Anthropic(api_key=api_key)
        print("  → Calling Claude to generate reference text...")

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast + cheap for this task
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": (
                    f"Here is a video transcript:\n\n\"{transcript}\"\n\n"
                    "Based on the topic and content of this transcript, write an ideal "
                    "reference answer that covers the same topic comprehensively. "
                    "The reference should:\n"
                    "1. Cover the same subject matter\n"
                    "2. Use clear, complete sentences\n"
                    "3. Be 3-5 sentences long\n"
                    "4. Include key concepts and terms from the topic\n"
                    "Write ONLY the reference text, no preamble or explanation."
                )
            }]
        )

        ref_text = message.content[0].text.strip()
        print(f"  ✔ Auto-generated reference ({len(ref_text)} chars)")
        return ref_text

    except Exception as e:
        print(f"  [!] Auto-generation failed: {str(e)[:80]}")
        print("      Using transcript as reference fallback.")
        return transcript


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Video-to-Text Similarity Evaluation System"
    )
    parser.add_argument("--video",      required=True,  help="Path to video file")
    parser.add_argument("--refs",       nargs="+", required=True,
                        help="Reference .txt file(s) or folder")
    parser.add_argument("--model",      default="base",
                        choices=["tiny","base","small","medium","large"])
    parser.add_argument("--output",     default="outputs/report.html")
    parser.add_argument("--language",   default=None)
    parser.add_argument("--interval",   default=2,  type=int,
                        help="Frame interval in seconds (default: 2)")
    parser.add_argument("--no-vision",  action="store_true",
                        help="Skip Claude Vision, use OCR only")
    parser.add_argument("--max-frames", default=30, type=int,
                        help="Max frames to AI-analyze (default: 30)")
    parser.add_argument("--auto-ref",   action="store_true",
                        help="Auto-generate an extra reference using Claude AI")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"\n❌ Video not found: {video_path}")
        sys.exit(1)

    use_vision = not args.no_vision

    print("\n╔" + "═"*58 + "╗")
    print("║   🎬  VIDEO-TO-TEXT SIMILARITY EVALUATION SYSTEM       ║")
    print("╚" + "═"*58 + "╝")
    print(f"\n  Video    : {video_path.name}")
    print(f"  Whisper  : {args.model} model")
    print(f"  Vision   : {'Claude Vision + OCR' if use_vision else 'OCR only'}")
    print(f"  Frames   : every {args.interval}s + scene changes")
    print(f"  Auto-ref : {'Yes ✨' if args.auto_ref else 'No'}")

    # ── STEP 1: References ─────────────────────────────────────────────────
    section("STEP 1 — Loading Reference Samples")
    references = load_references(args.refs)
    if not references:
        print("\n❌ No valid reference files found. Exiting.")
        sys.exit(1)
    print(f"\n  → {len(references)} reference(s) loaded.")

    # ── STEP 2: Transcribe ─────────────────────────────────────────────────
    section("STEP 2 — Audio Transcription (Whisper)")
    t0 = time.time()
    transcript = transcribe_video(
        video_path=str(video_path),
        model_size=args.model,
        language=args.language
    )
    print(f"\n  ✔ Transcription done ({time.time()-t0:.1f}s)")
    print(f"\n  TRANSCRIPT PREVIEW:")
    print(f"  {'─'*54}")
    preview = transcript[:400] + ("..." if len(transcript) > 400 else "")
    print(f"  {preview}")

    # ── STEP 3: Auto-generate reference ───────────────────────────────────
    auto_generated_refs = set()
    if args.auto_ref:
        section("STEP 3 — Auto-Generating Reference (Claude AI)")
        auto_ref_text = auto_generate_reference(transcript)
        auto_ref_name = "auto_generated_reference"
        references[auto_ref_name] = auto_ref_text
        auto_generated_refs.add(auto_ref_name)
        print(f"\n  Preview: {auto_ref_text[:200]}...")

        # Save it to samples folder for reuse
        auto_ref_path = Path("samples/auto_generated_reference.txt")
        auto_ref_path.parent.mkdir(exist_ok=True)
        auto_ref_path.write_text(auto_ref_text, encoding="utf-8")
        print(f"  ✔ Saved to: {auto_ref_path}")
    else:
        section("STEP 3 — Auto-Reference")
        print("  ℹ️  Skipped. Use --auto-ref to enable AI reference generation.")

    # ── STEP 4: Extract frames ─────────────────────────────────────────────
    section("STEP 4 — Frame Extraction (Interval + Scene Change)")
    frames_dir = Path("outputs/frames") / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = extract_key_frames(
        video_path=str(video_path),
        output_dir=str(frames_dir),
        interval_seconds=args.interval,
        scene_threshold=0.35
    )

    analyzed_frames = []
    frame_summary   = ""

    if not frames:
        print("  [!] No frames extracted — check ffmpeg installation.")
    else:
        # ── STEP 5: Analyze frames ─────────────────────────────────────────
        section("STEP 5 — Frame Analysis (OCR + Claude Vision)")

        if not use_vision:
            print("  ℹ️  Vision disabled — OCR only.\n")
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            print("  ⚠️  ANTHROPIC_API_KEY not set — skipping AI descriptions.")
            print("     export ANTHROPIC_API_KEY=your_key_here\n")

        analyzed_frames = analyze_all_frames(
            frames=frames,
            use_claude_vision=use_vision,
            max_frames=args.max_frames
        )
        frame_summary = build_frame_text_summary(analyzed_frames)

        print(f"\n  ✔ Frame analysis complete.")
        if frame_summary.strip():
            print(f"\n  FRAME CONTENT PREVIEW:")
            print(f"  {'─'*54}")
            for line in frame_summary.splitlines()[:5]:
                print(f"  {line}")

    # ── STEP 6: Merge ──────────────────────────────────────────────────────
    section("STEP 6 — Merging Audio + Visual Content")
    if frame_summary.strip():
        merged_content = (
            f"{transcript}\n\n"
            f"=== VISUAL CONTENT FROM FRAMES ===\n"
            f"{frame_summary}"
        )
        print(f"  ✔ Merged: transcript ({len(transcript)} chars)"
              f" + frames ({len(frame_summary)} chars)")
    else:
        merged_content = transcript
        print("  ℹ️  No frame content — using transcript only.")

    # ── STEP 7: Evaluate ───────────────────────────────────────────────────
    section("STEP 7 — Similarity Evaluation")
    results = evaluate_all(merged_content, references)
    print_summary(merged_content, results)

    # ── STEP 8: Report ─────────────────────────────────────────────────────
    section("STEP 8 — Generating Professional HTML Report")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_report(
        transcript=merged_content,
        references=references,
        results=results,
        video_name=video_path.name,
        output_path=str(output_path),
        analyzed_frames=analyzed_frames,
        frame_summary=frame_summary,
        auto_generated_refs=auto_generated_refs
    )

    print(f"\n  ✅ Report saved → {output_path.resolve()}")
    print("     Open it in your browser!\n")

    # Print best reference
    if results:
        best = max(results.items(), key=lambda x: x[1]["overall"])
        print(f"  🏆 Best matching reference: '{best[0]}' ({best[1]['overall']:.2%})\n")


if __name__ == "__main__":
    main()