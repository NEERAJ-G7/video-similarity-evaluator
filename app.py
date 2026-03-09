"""
app.py - VideoEval Flask Web Application
"""
import os, re, uuid, threading, base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from utils.transcriber     import transcribe_video
from utils.frame_extractor import extract_key_frames
from utils.frame_analyzer  import analyze_all_frames, build_frame_text_summary, semantic_score_frames
from utils.evaluator       import evaluate_all

app = Flask(__name__)
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
jobs = {}

# ── Leaderboard ───────────────────────────────────────────────────────────────
LEADERBOARD_FILE = Path("leaderboard.json")

def load_leaderboard():
    import json
    try:
        if LEADERBOARD_FILE.exists():
            return json.loads(LEADERBOARD_FILE.read_text())
    except: pass
    return []

def save_to_leaderboard(entry: dict):
    import json
    board = load_leaderboard()
    board.append(entry)
    board.sort(key=lambda x: x.get("overall", 0), reverse=True)
    LEADERBOARD_FILE.write_text(json.dumps(board, indent=2))

def _extract_topics(text, top_n=14):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2),
                              token_pattern=r"[a-zA-Z]{3,}", max_features=100)
        vec.fit([text])
        scores = dict(zip(vec.get_feature_names_out(), vec.transform([text]).toarray()[0]))
        return [k for k,v in sorted(scores.items(), key=lambda x:-x[1])[:top_n]]
    except:
        words = re.findall(r"[a-zA-Z]{4,}", text.lower())
        stop  = {"this","that","with","from","they","have","been","will","your",
                 "what","when","which","there","their","about","into","also","more"}
        freq  = {}
        for w in words:
            if w not in stop: freq[w] = freq.get(w,0)+1
        return [k for k,v in sorted(freq.items(), key=lambda x:-x[1])[:top_n]]

def _sw(transcript, reference, scores):
    try:
        stop = {"this","that","with","from","they","have","been","will","your",
                "what","when","which","there","their","about","into","also","more","very"}
        tw = set(re.findall(r"[a-zA-Z]{4,}", transcript.lower())) - stop
        rw = set(re.findall(r"[a-zA-Z]{4,}", reference.lower())) - stop
        matched = sorted(tw & rw)[:10]
        missing = sorted(rw - tw)[:10]
        extra   = sorted(tw - rw)[:6]
        strengths, weaknesses = [], []
        if scores.get("semantic",0)>=0.6: strengths.append("✅ Strong semantic alignment — core meaning matches")
        if scores.get("rouge1",0)>=0.4:   strengths.append("✅ Good vocabulary overlap with reference")
        if scores.get("tfidf",0)>=0.4:    strengths.append("✅ Key topics and keywords well covered")
        if scores.get("bleu",0)>=0.3:     strengths.append("✅ Phrases closely mirror the reference")
        if matched: strengths.append(f"✅ Matched terms: {', '.join(matched[:7])}")
        if scores.get("semantic",1)<0.5:  weaknesses.append("⚠️ Core meaning diverges from reference")
        if scores.get("rouge2",1)<0.3:    weaknesses.append("⚠️ Few matching phrases or bigrams found")
        if scores.get("tfidf",1)<0.3:     weaknesses.append("⚠️ Key topics from reference not well covered")
        if missing: weaknesses.append(f"⚠️ Missing terms: {', '.join(missing[:7])}")
        return {
            "strengths":  strengths  or ["No clear strengths identified"],
            "weaknesses": weaknesses or ["No significant weaknesses found"],
            "matched": matched, "missing": missing, "extra": extra
        }
    except:
        return {"strengths":[],"weaknesses":[],"matched":[],"missing":[],"extra":[]}

def process_video(job_id, video_path, reference_text, model):
    try:
        jobs[job_id].update({"status":"transcribing","step":"Step 1/4 — Transcribing audio with Whisper..."})
        transcript = transcribe_video(video_path=video_path, model_size=model)
        jobs[job_id].update({"status":"extracting","step":"Step 2/4 — Extracting key frames...","transcript":transcript})
        frames_dir = Path("uploads") / f"{job_id}_frames"
        frames = extract_key_frames(video_path=video_path, output_dir=str(frames_dir), interval_seconds=2, scene_threshold=0.35)
        jobs[job_id].update({"status":"analyzing","step":"Step 3/4 — Analyzing frames with Semantic Visual Context Layer..."})
        analyzed = analyze_all_frames(frames=frames, use_claude_vision=False, max_frames=20)
        frame_summary = build_frame_text_summary(analyzed)
        jobs[job_id].update({"status":"evaluating","step":"Step 4/4 — Computing similarity scores..."})
        merged = transcript
        if frame_summary.strip():
            merged = f"{transcript}\n\n=== VISUAL CONTENT FROM FRAMES ===\n{frame_summary}"
        results = evaluate_all(merged, {"reference": reference_text})
        scores  = results.get("reference", {})

        # Semantic scoring of each frame's visual action vs reference
        analyzed = semantic_score_frames(analyzed, reference_text)

        sw       = _sw(transcript, reference_text, scores)
        topics_t = _extract_topics(transcript)
        topics_r = _extract_topics(reference_text)
        thumbs   = []
        for f in analyzed[:16]:
            try:
                with open(f["path"],"rb") as img:
                    b64 = base64.b64encode(img.read()).decode("utf-8")
                thumbs.append({
                    "b64":           b64,
                    "time":          f.get("time_label","??:??"),
                    "source":        f.get("source","interval"),
                    "ocr":           f.get("ocr_text","")[:80],
                    "description":   f.get("description","")[:260],
                    "actions":       f.get("actions", []),
                    "change_level":  f.get("change_level", "none"),
                    "change_label":  f.get("change_label", ""),
                    "semantic_tags": f.get("semantic_tags", [])[:5],
                    "scene_type":    f.get("scene_type", ""),
                    "semantic_score": f.get("semantic_score", -1),
                    "semantic_label": f.get("semantic_label", ""),
                    "semantic_color": f.get("semantic_color", "#64748b"),
                    "semantic_pct":   f.get("semantic_pct", 0),
                })
            except: pass
        jobs[job_id].update({"status":"done","step":"Complete!","transcript":transcript,
                             "reference":reference_text,"scores":scores,"sw":sw,
                             "topics_t":topics_t,"topics_r":topics_r,
                             "frame_thumbs":thumbs,"frame_count":len(analyzed)})

        # Save to leaderboard
        from datetime import datetime
        video_name = Path(video_path).name
        save_to_leaderboard({
            "job_id":    job_id,
            "candidate": video_name,
            "overall":   round(scores.get("overall", 0), 4),
            "semantic":  round(scores.get("semantic", 0), 4),
            "rouge_l":   round(scores.get("rougeL", 0), 4),
            "tfidf":     round(scores.get("tfidf", 0), 4),
            "bleu":      round(scores.get("bleu", 0), 4),
            "frames":    len(analyzed),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
    except Exception as e:
        import traceback
        jobs[job_id].update({"status":"error","error":str(e),"trace":traceback.format_exc()})

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/submit", methods=["POST"])
def submit():
    if "video" not in request.files: return jsonify({"error":"No video"}),400
    vf  = request.files["video"]
    ref = request.form.get("reference","").strip()
    mdl = request.form.get("model","base")
    if not ref: return jsonify({"error":"Reference required"}),400
    jid = str(uuid.uuid4())[:8]
    ext = Path(vf.filename).suffix or ".mp4"
    vp  = UPLOAD_FOLDER / f"{jid}{ext}"
    vf.save(str(vp))
    jobs[jid] = {"status":"queued","step":"Queued...","reference":ref}
    threading.Thread(target=process_video, args=(jid,str(vp),ref,mdl), daemon=True).start()
    return jsonify({"job_id":jid})

@app.route("/status/<jid>")
def status(jid):
    j = jobs.get(jid)
    return (jsonify(j) if j else jsonify({"error":"Not found"}),404)[not bool(j)]

@app.route("/leaderboard")
def leaderboard():
    return jsonify(load_leaderboard())

@app.route("/leaderboard/clear", methods=["POST"])
def clear_leaderboard():
    import json
    LEADERBOARD_FILE.write_text(json.dumps([]))
    return jsonify({"ok": True})

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>VideoEval — AI Video Evaluation</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#05080f;--card:#0f1624;--border:#1a2744;--accent:#3b6bff;--accent2:#00d4aa;--text:#e8edf5;--muted:#5a7299;--dim:#1e2d4a;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Outfit',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(var(--border) 1px,transparent 1px),linear-gradient(90deg,var(--border) 1px,transparent 1px);background-size:40px 40px;opacity:0.18;pointer-events:none;z-index:0;}
.wrap{position:relative;z-index:1;max-width:920px;margin:0 auto;padding:44px 20px 80px;}
.header{text-align:center;margin-bottom:48px;animation:fD .7s ease both;}
.logo-tag{display:inline-flex;align-items:center;gap:8px;background:#0e1f44;border:1px solid #1e3a7a;border-radius:40px;padding:5px 16px;font-family:'JetBrains Mono',monospace;font-size:.7em;color:#7aabff;margin-bottom:20px;letter-spacing:1px;}
.logo-tag span{color:var(--accent2);}
h1{font-family:'Syne',sans-serif;font-size:clamp(1.9em,5vw,3em);font-weight:800;line-height:1.05;letter-spacing:-1px;margin-bottom:14px;}
h1 em{font-style:normal;background:linear-gradient(135deg,#3b6bff,#00d4aa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.subtitle{color:var(--muted);font-size:.95em;font-weight:300;max-width:500px;margin:0 auto;line-height:1.7;}
.card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:26px;margin-bottom:14px;position:relative;overflow:hidden;animation:fU .5s ease both;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,#3b6bff2a,transparent);}
.slabel{font-family:'JetBrains Mono',monospace;font-size:.67em;color:var(--accent);text-transform:uppercase;letter-spacing:3px;margin-bottom:16px;display:flex;align-items:center;gap:10px;}
.slabel::after{content:'';flex:1;height:1px;background:var(--border);}
.upload-zone{border:2px dashed var(--dim);border-radius:12px;padding:32px 20px;text-align:center;cursor:pointer;transition:all .3s;position:relative;background:#080d1a;}
.upload-zone:hover,.upload-zone.drag{border-color:var(--accent);background:#0a1228;}
.upload-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;}
.upload-zone h3{font-size:.92em;font-weight:600;margin-bottom:4px;}
.upload-zone p{font-size:.78em;color:var(--muted);}
.file-chip{display:none;align-items:center;gap:10px;background:#0e1f44;border:1px solid var(--accent);border-radius:9px;padding:9px 13px;margin-top:10px;font-size:.82em;}
.file-chip .fn{color:var(--accent2);font-family:'JetBrains Mono',monospace;font-size:.88em;}
textarea{width:100%;background:#080d1a;border:1px solid var(--dim);border-radius:11px;color:var(--text);font-family:'Outfit',sans-serif;font-size:.88em;line-height:1.7;padding:13px;resize:vertical;min-height:120px;outline:none;transition:border-color .2s;}
textarea:focus{border-color:var(--accent);}
textarea::placeholder{color:var(--muted);}
label{display:block;font-size:.8em;color:var(--muted);margin-bottom:6px;font-weight:500;}
.sw-wrap{position:relative;}.sw-wrap::after{content:'▾';position:absolute;right:13px;top:50%;transform:translateY(-50%);color:var(--muted);pointer-events:none;}
select{width:100%;background:#080d1a;border:1px solid var(--dim);border-radius:10px;color:var(--text);font-family:'Outfit',sans-serif;font-size:.86em;padding:10px 13px;outline:none;appearance:none;cursor:pointer;}
.form-row{margin-bottom:16px;}
.btn{width:100%;padding:14px;background:linear-gradient(135deg,var(--accent),#2255dd);border:none;border-radius:11px;color:#fff;font-family:'Syne',sans-serif;font-size:.97em;font-weight:700;cursor:pointer;letter-spacing:.5px;transition:all .2s;}
.btn:hover{transform:translateY(-1px);box-shadow:0 8px 28px #3b6bff2a;}
.btn-ghost{background:transparent;border:1px solid var(--border);border-radius:9px;color:var(--muted);padding:9px 20px;font-family:'Outfit',sans-serif;font-size:.83em;cursor:pointer;transition:all .2s;margin-top:12px;}
.btn-ghost:hover{border-color:var(--accent);color:var(--text);}
#progressSection{display:none;animation:fU .5s ease both;}
.prog-card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:38px;text-align:center;}
.spinner{width:48px;height:48px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;margin:0 auto 18px;}
.step-txt{font-family:'JetBrains Mono',monospace;font-size:.8em;color:var(--accent2);margin-bottom:14px;}
.pbar-wrap{background:var(--dim);border-radius:8px;height:5px;max-width:380px;margin:0 auto;overflow:hidden;}
.pbar{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:8px;transition:width .5s ease;width:0%;}
#resultsSection{display:none;animation:fU .6s ease both;}
.score-hero{text-align:center;padding:32px 16px;}
.score-ring{width:150px;height:150px;margin:0 auto 14px;position:relative;}
.score-ring svg{transform:rotate(-90deg);}
.srt{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.spct{font-family:'Syne',sans-serif;font-size:2.2em;font-weight:800;line-height:1;}
.slbl{font-size:.63em;color:var(--muted);margin-top:3px;letter-spacing:1px;}
.sinterp{font-size:.88em;color:var(--muted);}
.mgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:9px;margin-bottom:14px;}
.mc{background:#080d1a;border:1px solid var(--border);border-radius:11px;padding:13px;}
.mn{font-size:.68em;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:5px;}
.mv{font-family:'Syne',sans-serif;font-size:1.45em;font-weight:700;}
.md{font-size:.68em;color:var(--muted);margin-top:2px;}
.mbar{margin-top:7px;background:var(--dim);border-radius:3px;height:3px;}
.mbarf{height:3px;border-radius:3px;transition:width 1.2s ease;}
.cgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
@media(max-width:580px){.cgrid{grid-template-columns:1fr;}}
.cbox{background:#080d1a;border-radius:11px;padding:13px;border:1px solid var(--border);}
.cbox h4{font-size:.7em;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;font-family:'JetBrains Mono',monospace;}
.ctxt{font-size:.83em;line-height:1.75;color:#94a3b8;max-height:190px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;}
.swgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
@media(max-width:560px){.swgrid{grid-template-columns:1fr;}}
.swbox{background:#080d1a;border-radius:11px;padding:14px;border:1px solid var(--border);}
.swbox h4{font-size:.7em;text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;font-family:'JetBrains Mono',monospace;}
.swi{font-size:.82em;color:#94a3b8;line-height:1.6;margin-bottom:7px;padding-bottom:7px;border-bottom:1px solid var(--border);}
.swi:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0;}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px;}
.tag{padding:3px 9px;border-radius:20px;font-size:.7em;font-family:'JetBrains Mono',monospace;font-weight:500;}
.tg{background:#10b98118;color:#10b981;}.tr{background:#ef444418;color:#ef4444;}.tb{background:#3b6bff18;color:#7aabff;}
.tcloud{display:flex;flex-wrap:wrap;gap:7px;}
.tpill{padding:4px 13px;border-radius:20px;font-size:.78em;font-weight:500;border:1px solid;}
.tpill.transcript{background:#3b6bff0d;border-color:#3b6bff33;color:#7aabff;}
.tpill.reference{background:#00d4aa0d;border-color:#00d4aa33;color:#00d4aa;}
.tpill.shared{background:#f59e0b0d;border-color:#f59e0b33;color:#fbbf24;}
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:9px;}
.fcard{background:#080d1a;border-radius:9px;overflow:hidden;border:1px solid var(--border);}
.fcard img{width:100%;height:115px;object-fit:cover;display:block;}
.finfo{padding:9px 11px;}
.ftime{font-family:'JetBrains Mono',monospace;font-size:.73em;color:var(--accent2);font-weight:500;}
.fsrc{font-size:.66em;padding:2px 6px;border-radius:10px;font-weight:600;margin-left:5px;}
.fdesc{font-size:.75em;color:#cbd5e1;margin-top:7px;line-height:1.55;border-top:1px solid var(--border);padding-top:7px;}
.fsem{display:flex;align-items:center;gap:6px;margin-top:7px;padding:6px 9px;border-radius:8px;border:1px solid;}
.fsem-pct{font-family:'JetBrains Mono',monospace;font-size:.78em;font-weight:700;}
.fsem-lbl{font-size:.72em;flex:1;}
.fsem-bar{height:3px;border-radius:3px;margin-top:4px;transition:width 1s ease;}
.focr{font-size:.68em;color:var(--muted);margin-top:5px;line-height:1.4;font-family:'JetBrains Mono',monospace;}
/* Activity Timeline */
.atl-item{display:flex;gap:14px;margin-bottom:14px;animation:fU .4s ease both;}
.atl-left{display:flex;flex-direction:column;align-items:center;min-width:52px;}
.atl-time{font-family:'JetBrains Mono',monospace;font-size:.72em;color:var(--accent2);font-weight:700;white-space:nowrap;}
.atl-dot{width:10px;height:10px;border-radius:50%;margin:5px 0;flex-shrink:0;}
.atl-line{flex:1;width:2px;background:var(--border);margin:0 auto;}
.atl-right{flex:1;background:#080d1a;border:1px solid var(--border);border-radius:10px;padding:12px 14px;margin-bottom:0;}
.atl-desc{font-size:.83em;color:#cbd5e1;line-height:1.6;margin-bottom:8px;}
.atl-actions{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:6px;}
.atl-action{padding:2px 9px;border-radius:20px;font-size:.7em;font-weight:600;font-family:'JetBrains Mono',monospace;background:#3b6bff18;color:#7aabff;border:1px solid #3b6bff22;}
.atl-action.scene_change{background:#f59e0b18;color:#fbbf24;border-color:#f59e0b22;}
.atl-action.transition{background:#8b5cf618;color:#a78bfa;border-color:#8b5cf622;}
.atl-meta{display:flex;align-items:center;gap:10px;font-size:.7em;color:var(--muted);}
.motion-badge{padding:1px 7px;border-radius:10px;font-size:.9em;font-weight:600;}
.motion-high{background:#ef444418;color:#ef4444;}
.motion-medium{background:#f59e0b18;color:#fbbf24;}
.motion-low{background:#10b98118;color:#10b981;}
.err-box{background:#1a0a0a;border:1px solid #5a1a1a;border-radius:11px;padding:18px;color:#ff6b6b;font-family:'JetBrains Mono',monospace;font-size:.8em;}
/* Nav tabs */
.nav-tabs{display:flex;gap:6px;margin-bottom:28px;background:#0a101e;border:1px solid var(--border);border-radius:14px;padding:5px;animation:fD .5s ease both;}
.nav-tab{flex:1;padding:10px;border:none;background:transparent;color:var(--muted);font-family:'Outfit',sans-serif;font-size:.88em;font-weight:500;cursor:pointer;border-radius:10px;transition:all .2s;text-align:center;}
.nav-tab.active{background:var(--card);color:var(--text);box-shadow:0 2px 8px #00000044;}
.nav-tab:hover:not(.active){color:var(--text);}
/* Leaderboard */
.lb-table{width:100%;border-collapse:collapse;}
.lb-table th{padding:10px 14px;text-align:left;font-size:.68em;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;border-bottom:1px solid var(--border);font-family:'JetBrains Mono',monospace;font-weight:500;}
.lb-table td{padding:12px 14px;font-size:.85em;border-bottom:1px solid #0f1a30;vertical-align:middle;}
.lb-table tr:last-child td{border-bottom:none;}
.lb-table tr:hover td{background:#ffffff04;}
.lb-rank{font-family:'Syne',sans-serif;font-size:1.1em;font-weight:800;width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;}
.lb-score{font-family:'Syne',sans-serif;font-size:1.15em;font-weight:700;}
.lb-bar-wrap{background:var(--dim);border-radius:4px;height:5px;width:80px;display:inline-block;vertical-align:middle;margin-left:8px;}
.lb-bar{height:5px;border-radius:4px;transition:width 1s ease;}
.lb-empty{text-align:center;padding:48px 20px;color:var(--muted);font-size:.9em;}
.lb-empty span{display:block;font-size:2em;margin-bottom:12px;}
@keyframes fD{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fU{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">

<div class="header">
  <div class="logo-tag">⬡ VideoEval &nbsp;·&nbsp; <span>AI Powered</span></div>
  <h1>Video Response<br><em>Evaluation System</em></h1>
  <p class="subtitle">Upload a candidate video and reference answer. Our AI transcribes, analyzes frames, and scores similarity across multiple dimensions.</p>
</div>

<div id="formSection">
  <div class="card">
    <div class="slabel">01 &nbsp; Upload Video</div>
    <div class="upload-zone" id="uploadZone">
      <input type="file" id="videoFile" accept="video/*">
      <span style="font-size:2.2em;display:block;margin-bottom:9px">🎬</span>
      <h3>Drop your video here or click to browse</h3>
      <p>MP4, MOV, AVI, MKV supported</p>
    </div>
    <div class="file-chip" id="fileChip">
      <span>📹</span><span class="fn" id="fileName"></span>
      <span style="color:var(--muted);font-size:.78em" id="fileSize"></span>
    </div>
  </div>

  <div class="card" style="animation-delay:.08s">
    <div class="slabel">02 &nbsp; Reference Answer</div>
    <div class="form-row">
      <label>Paste the expected correct answer</label>
      <textarea id="referenceText" placeholder="Enter the reference answer that the video response should be evaluated against..."></textarea>
    </div>
  </div>

  <div class="card" style="animation-delay:.14s">
    <div class="slabel">03 &nbsp; Settings</div>
    <div class="form-row">
      <label>Whisper Model</label>
      <div class="sw-wrap">
        <select id="modelSelect">
          <option value="tiny">Tiny — Fastest</option>
          <option value="base" selected>Base — Recommended ✓</option>
          <option value="small">Small — More Accurate</option>
          <option value="medium">Medium — High Accuracy</option>
        </select>
      </div>
    </div>
  </div>
  <button class="btn" onclick="submitForm()">⚡ Analyze Video</button>
</div>

<div id="progressSection">
  <div class="prog-card">
    <div class="spinner"></div>
    <div class="step-txt" id="stepText">Initializing...</div>
    <div class="pbar-wrap"><div class="pbar" id="progressBar"></div></div>
    <p style="color:var(--muted);font-size:.78em;margin-top:13px">Processing may take 1–3 minutes</p>
  </div>
</div>

<div id="resultsSection">
  <div class="card">
    <div class="score-hero">
      <div class="score-ring">
        <svg width="150" height="150" viewBox="0 0 150 150">
          <circle cx="75" cy="75" r="63" fill="none" stroke="#1a2744" stroke-width="10"/>
          <circle cx="75" cy="75" r="63" fill="none" stroke-width="10" stroke-linecap="round"
            id="scoreCircle" stroke-dasharray="396" stroke-dashoffset="396"
            style="transition:stroke-dashoffset 1.5s ease;stroke:#3b6bff"/>
        </svg>
        <div class="srt">
          <div class="spct" id="overallPct">—</div>
          <div class="slbl">OVERALL</div>
        </div>
      </div>
      <div class="sinterp" id="scoreInterp"></div>
    </div>
  </div>

  <div class="mgrid" id="metricGrid"></div>

  <!-- 1. Side-by-side comparison -->
  <div class="card">
    <div class="slabel">📄 Transcript vs Reference</div>
    <div class="cgrid">
      <div class="cbox">
        <h4 style="color:#7aabff">🎤 Video Transcript</h4>
        <div class="ctxt" id="transcriptBox"></div>
      </div>
      <div class="cbox">
        <h4 style="color:#00d4aa">📋 Reference Answer</h4>
        <div class="ctxt" id="referenceBox"></div>
      </div>
    </div>
  </div>

  <!-- 2. Strengths & Weaknesses -->
  <div class="card">
    <div class="slabel">⚖️ Strengths & Weaknesses</div>
    <div class="swgrid">
      <div class="swbox">
        <h4 style="color:#10b981">✅ Strengths</h4>
        <div id="strengthsList"></div>
      </div>
      <div class="swbox">
        <h4 style="color:#ef4444">⚠️ Weaknesses</h4>
        <div id="weaknessesList"></div>
      </div>
    </div>
    <div style="margin-top:13px">
      <div style="font-size:.73em;color:var(--muted);margin-bottom:7px;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:1px">Term Analysis</div>
      <div class="tags" id="tagList"></div>
    </div>
  </div>

  <!-- 3. Key Topics -->
  <div class="card">
    <div class="slabel">🏷️ Key Topics Extracted</div>
    <div style="margin-bottom:9px;font-size:.76em;color:var(--muted)">
      <span style="background:#3b6bff18;color:#7aabff;padding:2px 8px;border-radius:10px;margin-right:5px">■ Transcript only</span>
      <span style="background:#00d4aa18;color:#00d4aa;padding:2px 8px;border-radius:10px;margin-right:5px">■ Reference only</span>
      <span style="background:#f59e0b18;color:#fbbf24;padding:2px 8px;border-radius:10px">■ Shared</span>
    </div>
    <div class="tcloud" id="topicCloud"></div>
  </div>

  <!-- 4. Activity Timeline -->
  <div class="card">
    <div class="slabel">🎬 Semantic Visual Context — Activity Timeline</div>
    <p style="font-size:.78em;color:var(--muted);margin-bottom:14px">Every action detected across the video — scene by scene</p>
    <div id="activityTimeline"></div>
  </div>

  <!-- 5. Frame Gallery -->
  <div class="card">
    <div class="slabel">🖼️ Extracted Frames with Visual Context</div>
    <p style="font-size:.78em;color:var(--muted);margin-bottom:13px" id="frameCount"></p>
    <div class="fgrid" id="framesGrid"></div>
  </div>

  <div style="text-align:center">
    <button class="btn-ghost" onclick="resetForm()">↩ Evaluate Another Video</button>
  </div>
</div>
</div>

<script>
const zone=document.getElementById('uploadZone'),input=document.getElementById('videoFile');
zone.addEventListener('dragover',e=>{e.preventDefault();zone.classList.add('drag');});
zone.addEventListener('dragleave',()=>zone.classList.remove('drag'));
zone.addEventListener('drop',e=>{e.preventDefault();zone.classList.remove('drag');if(e.dataTransfer.files[0]){input.files=e.dataTransfer.files;showFile(e.dataTransfer.files[0]);}});
input.addEventListener('change',()=>{if(input.files[0])showFile(input.files[0]);});
function showFile(f){document.getElementById('fileName').textContent=f.name;document.getElementById('fileSize').textContent=(f.size/1024/1024).toFixed(1)+' MB';document.getElementById('fileChip').style.display='flex';}

async function submitForm(){
  const video=input.files[0],ref=document.getElementById('referenceText').value.trim(),model=document.getElementById('modelSelect').value;
  if(!video){alert('Please upload a video.');return;}
  if(!ref){alert('Please enter reference text.');return;}
  document.getElementById('formSection').style.display='none';
  document.getElementById('progressSection').style.display='block';
  const fd=new FormData();fd.append('video',video);fd.append('reference',ref);fd.append('model',model);
  const {job_id}=await(await fetch('/submit',{method:'POST',body:fd})).json();
  pollStatus(job_id);
}

const stp={queued:5,transcribing:25,extracting:50,analyzing:70,evaluating:90,done:100};
function pollStatus(jid){
  const iv=setInterval(async()=>{
    const d=await(await fetch('/status/'+jid)).json();
    document.getElementById('stepText').textContent=d.step||'...';
    document.getElementById('progressBar').style.width=(stp[d.status]||0)+'%';
    if(d.status==='done'){clearInterval(iv);showResults(d);}
    if(d.status==='error'){clearInterval(iv);document.getElementById('progressSection').innerHTML='<div class="err-box">❌ '+d.error+'</div><div style="text-align:center"><button class="btn-ghost" onclick="resetForm()">↩ Try Again</button></div>';}
  },2000);
}

function showResults(d){
  document.getElementById('progressSection').style.display='none';
  document.getElementById('resultsSection').style.display='block';
  const sc=d.scores||{},ov=sc.overall||0,pct=Math.round(ov*100);
  const clr=pct>=80?'#10b981':pct>=50?'#f59e0b':'#ef4444';
  setTimeout(()=>{const c=document.getElementById('scoreCircle');c.style.strokeDashoffset=396-(pct/100)*396;c.style.stroke=clr;},100);
  document.getElementById('overallPct').textContent=pct+'%';
  document.getElementById('overallPct').style.color=clr;
  const ints=[[.85,'🏆 Excellent — content closely matches reference'],[.70,'✅ Good — strong topical alignment'],[.55,'🟡 Moderate — related but differences exist'],[.40,'🟠 Weak — loosely related content'],[0,'🔴 Poor — content does not match reference']];
  document.getElementById('scoreInterp').textContent=ints.find(([t])=>ov>=t)[1];

  const metrics=[{name:'Semantic',val:sc.semantic,desc:'Meaning match'},{name:'BLEU',val:sc.bleu,desc:'N-gram precision'},{name:'ROUGE-1',val:sc.rouge1,desc:'Unigram recall'},{name:'ROUGE-L',val:sc.rougeL,desc:'Sequence match'},{name:'TF-IDF',val:sc.tfidf,desc:'Keyword overlap'}];
  document.getElementById('metricGrid').innerHTML=metrics.map(m=>{const p=Math.round((m.val||0)*100),c=p>=80?'#10b981':p>=50?'#f59e0b':'#ef4444';return`<div class="mc"><div class="mn">${m.name}</div><div class="mv" style="color:${c}">${p}%</div><div class="md">${m.desc}</div><div class="mbar"><div class="mbarf" style="width:${p}%;background:${c}"></div></div></div>`;}).join('');

  document.getElementById('transcriptBox').textContent=d.transcript||'—';
  document.getElementById('referenceBox').textContent=d.reference||'—';

  const sw=d.sw||{};
  document.getElementById('strengthsList').innerHTML=(sw.strengths||[]).map(s=>`<div class="swi">${s}</div>`).join('');
  document.getElementById('weaknessesList').innerHTML=(sw.weaknesses||[]).map(s=>`<div class="swi">${s}</div>`).join('');
  document.getElementById('tagList').innerHTML=[
    ...(sw.matched||[]).slice(0,8).map(w=>`<span class="tag tg">${w}</span>`),
    ...(sw.missing||[]).slice(0,8).map(w=>`<span class="tag tr">${w}</span>`),
    ...(sw.extra||[]).slice(0,4).map(w=>`<span class="tag tb">${w}</span>`),
  ].join('')||'<span style="color:var(--muted);font-size:.8em">No term data</span>';

  const tt=new Set(d.topics_t||[]),tr=new Set(d.topics_r||[]),all=[...new Set([...(d.topics_t||[]),...(d.topics_r||[])])];
  document.getElementById('topicCloud').innerHTML=all.map(w=>{const inT=tt.has(w),inR=tr.has(w),cls=(inT&&inR)?'shared':inT?'transcript':'reference';return`<span class="tpill ${cls}">${w}</span>`;}).join('');

  // Activity Timeline
  const atl = d.activity_timeline || [];
  document.getElementById('activityTimeline').innerHTML = atl.length ? atl.map((item, i) => {
    const isLast = i === atl.length - 1;
    const dotClr = item.sem_score >= 0.7 ? '#10b981' : item.sem_score >= 0.45 ? '#f59e0b' : item.sem_score >= 0 ? '#ef4444' : '#3b6bff';
    const motionBadge = item.motion !== 'low' ? `<span class="motion-badge motion-${item.motion}">⚡ ${item.motion} motion</span>` : '';
    const semBadge = item.sem_score >= 0 ? `<span style="font-size:.7em;color:${item.sem_color}">🧠 ${Math.round(item.sem_score*100)}% match</span>` : '';
    const srcBadge = item.source === 'scene_change' ? '<span style="font-size:.7em;color:#f59e0b">🔀 scene change</span>' : '';
    const actionBadges = (item.actions||[]).map(a => {
      const cls = a.includes('transition') || a.includes('scene') ? 'atl-action transition' : 'atl-action';
      return `<span class="${cls}">${a}</span>`;
    }).join('');
    return `<div class="atl-item" style="animation-delay:${i*0.04}s">
      <div class="atl-left">
        <span class="atl-time">${item.time}</span>
        <div class="atl-dot" style="background:${dotClr}"></div>
        ${!isLast ? '<div class="atl-line"></div>' : ''}
      </div>
      <div class="atl-right">
        ${item.description ? `<div class="atl-desc">${item.description}</div>` : ''}
        ${actionBadges ? `<div class="atl-actions">${actionBadges}</div>` : ''}
        <div class="atl-meta">${motionBadge}${semBadge}${srcBadge}</div>
        ${item.ocr ? `<div style="font-size:.7em;color:var(--muted);margin-top:5px;font-family:'JetBrains Mono',monospace">📝 ${item.ocr}</div>` : ''}
      </div>
    </div>`;
  }).join('') : '<p style="color:var(--muted);font-size:.83em">No activity data available.</p>';

  // Frame Gallery
  const fr=d.frame_thumbs||[];
  document.getElementById('frameCount').textContent=`${d.frame_count||0} frames extracted — showing top ${fr.length}`;

  const actionIcons={
    introducing:'🎤',explaining:'💬',defining:'📖',listing:'📋',
    comparing:'⚖️',demonstrating:'🔍',concluding:'✅',questioning:'❓',
    presenting_data:'📊',transitioning:'➡️',emphasizing:'⚡',narrating:'📝',
    displaying_slide:'🖥️',scene_transition:'🔀',new_content_appeared:'🆕',
    showing_video_content:'🎬',showing_visual_content:'🖼️'
  };
  const changePalette={none:'#334155',minor:'#64748b',moderate:'#f59e0b',major:'#ef4444',transition:'#3b6bff'};

  document.getElementById('framesGrid').innerHTML=fr.length?fr.map(f=>{
    const sc    = f.source==='scene_change'?'#3b6bff':'#10b981';
    const sl    = f.source==='scene_change'?'🔀 Scene':'⏱ Interval';
    const sClr  = f.semantic_color||'#64748b';
    const sPct  = f.semantic_pct||0;
    const cClr  = changePalette[f.change_level]||'#334155';

    const actionBadges=(f.actions||[]).map(a=>
      `<span style="display:inline-flex;align-items:center;gap:3px;background:#1a2744;border:1px solid #2a3a5c;border-radius:6px;padding:2px 8px;font-size:.66em;color:#94a3b8;margin:2px 2px 0 0">
        ${actionIcons[a]||'▸'} ${a.replace(/_/g,' ')}
      </span>`).join('');

    const tagBadges=(f.semantic_tags||[]).map(t=>
      `<span style="background:#3b6bff0d;border:1px solid #3b6bff22;color:#7aabff;padding:1px 6px;border-radius:8px;font-size:.63em">${t}</span>`
    ).join('');

    const changeLine = f.change_label ?
      `<div style="display:flex;align-items:center;gap:5px;margin-top:6px;font-size:.68em;color:${cClr}">
        <div style="width:6px;height:6px;border-radius:50%;background:${cClr};flex-shrink:0"></div>${f.change_label}
      </div>` : '';

    const semBar = sPct>0 ?
      `<div style="margin-top:8px;padding:7px 9px;background:${sClr}0d;border:1px solid ${sClr}22;border-radius:8px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
          <span style="font-size:.67em;color:${sClr};font-family:'JetBrains Mono',monospace">🧠 Semantic Match</span>
          <span style="font-size:.72em;font-weight:700;color:${sClr}">${sPct}%</span>
        </div>
        <div style="background:var(--dim);border-radius:3px;height:3px">
          <div style="width:${sPct}%;height:3px;border-radius:3px;background:${sClr}"></div>
        </div>
        <div style="font-size:.64em;color:${sClr};margin-top:3px;opacity:.85">${f.semantic_label||''}</div>
      </div>` : '';

    return`<div class="fcard">
      <img src="data:image/jpeg;base64,${f.b64}" alt="frame ${f.time}">
      <div class="finfo">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">
          <span class="ftime">${f.time}</span>
          <span class="fsrc" style="background:${sc}18;color:${sc}">${sl}</span>
        </div>
        ${f.description?`<div class="fdesc">${f.description}</div>`:''}
        ${changeLine}
        ${actionBadges?`<div style="margin-top:6px;display:flex;flex-wrap:wrap">${actionBadges}</div>`:''}
        ${tagBadges?`<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:3px">${tagBadges}</div>`:''}
        ${semBar}
        ${f.ocr?`<div class="focr" style="margin-top:6px">📝 ${f.ocr}</div>`:''}
      </div>
    </div>`;
  }).join(''):'<p style="color:var(--muted);font-size:.82em">No frames extracted.</p>';
}

function resetForm(){
  document.getElementById('formSection').style.display='block';
  document.getElementById('progressSection').style.display='none';
  document.getElementById('resultsSection').style.display='none';
  input.value='';document.getElementById('referenceText').value='';document.getElementById('fileChip').style.display='none';
}
</script>
</body>
</html>"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)