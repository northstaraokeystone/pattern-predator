import streamlit as st

# === GUARANTEED SOUND (NO <audio>, NO st.audio). Works on fresh runs (Chrome/Firefox/Safari, desktop+mobile).
# Uses WebAudio in the main document via st.html (falls back to components.html if needed).
# Synthesis = no network, no CORS, no packages. Immediate if audio is already unlocked; otherwise shows a 1-tap gate.


def human_victory() -> None:
    st.balloons()
    st.markdown(
        "<h1 style='text-align:center; color:#00ff41; text-shadow:0 0 30px #00ff41;'>YOU DEFEATED THE AI!</h1>"
        "<h2 style='text-align:center; color:#00ff41;'>HUMANITY STILL REIGNS</h2>",
        unsafe_allow_html=True,
    )

    html_fn = getattr(st, "html", None)
    if html_fn is None:
        import streamlit.components.v1 as components

        html_fn = components.html

    html_fn(
        """
<div id="pp-snd-anchor"></div>
<script>
(function(){
  // --- Single global sound engine ---
  if(!window.__PP_SOUND__){
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC({latencyHint:'interactive'});
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -20; comp.knee.value = 30; comp.ratio.value = 12;
    comp.attack.value = 0.003; comp.release.value = 0.25;
    const master = ctx.createGain(); master.gain.value = 0.9;
    master.connect(comp); comp.connect(ctx.destination);

    const q = [];
    function schedule(fn){ (ctx.state === 'running') ? fn() : q.push(fn); }
    function flush(){ if(ctx.state==='running'){ while(q.length){ (q.shift())(); } } }

    async function ensureUnlocked(){
      try{ await ctx.resume(); }catch(e){}
      if(ctx.state !== 'running'){
        // one-tap audio gate overlay (tiny and unobtrusive for victory path)
        let gate = document.getElementById('pp-audio-gate');
        if(!gate){
          gate = document.createElement('div');
          gate.id = 'pp-audio-gate';
          gate.style.cssText = 'position:fixed;bottom:16px;left:50%;transform:translateX(-50%);' +
                               'background:rgba(0,0,0,0.75);color:#fff;padding:10px 14px;' +
                               'border-radius:10px;font:600 14px system-ui,Arial;z-index:10000;cursor:pointer;' +
                               'box-shadow:0 4px 20px rgba(0,0,0,0.35)';
          gate.textContent = 'tap to enable sound';
          gate.addEventListener('pointerdown', async (e)=>{
            e.preventDefault();
            try{ await ctx.resume(); }catch(_){}
            if(ctx.state==='running'){ gate.remove(); flush(); }
          }, {once:true});
          document.body.appendChild(gate);
        }
        return false;
      }
      return true;
    }

    // --- Sound design: short, punchy victory fanfare (oscillators + noise) ---
    function playVictory(){
      const t0 = ctx.currentTime + 0.01;
      const triad = [523.25, 659.25, 783.99]; // C5, E5, G5
      const arp  = [523.25, 659.25, 783.99, 1046.50]; // +C6

      // Arpeggio
      arp.forEach((f, i)=>{
        const o = ctx.createOscillator(); o.type = 'square'; o.frequency.value = f;
        const g = ctx.createGain(); g.gain.value = 0.0001;
        const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 1800; bp.Q.value = 0.9;
        o.connect(bp); bp.connect(g); g.connect(master);
        const st = t0 + i*0.14;
        o.start(st);
        g.gain.setValueAtTime(0.0001, st);
        g.gain.exponentialRampToValueAtTime(0.65, st+0.03);
        g.gain.exponentialRampToValueAtTime(0.0001, st+0.23);
        o.stop(st+0.25);
      });

      // Final stab (triad)
      triad.forEach((f)=>{
        const o = ctx.createOscillator(); o.type = 'square'; o.frequency.value = f;
        const g = ctx.createGain(); g.gain.value = 0.0001;
        const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 1600; bp.Q.value = 0.8;
        o.connect(bp); bp.connect(g); g.connect(master);
        const st = t0 + 0.56;
        o.start(st);
        g.gain.setValueAtTime(0.0001, st);
        g.gain.exponentialRampToValueAtTime(0.8, st+0.03);
        g.gain.exponentialRampToValueAtTime(0.0001, st+0.36);
        o.stop(st+0.38);
      });

      // Cymbal-ish noise burst
      const dur = 0.3, sr = ctx.sampleRate, n = Math.floor(sr*dur);
      const buf = ctx.createBuffer(1, n, sr), data = buf.getChannelData(0);
      for(let i=0;i<n;i++){ data[i] = (Math.random()*2-1)*Math.pow(1-i/n,0.35); }
      const ns = ctx.createBufferSource(); ns.buffer = buf;
      const hp = ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value = 4000; hp.Q.value=0.7;
      const ng = ctx.createGain(); ng.gain.value = 0.0001;
      ns.connect(hp); hp.connect(ng); ng.connect(master);
      const nt = t0 + 0.55;
      ns.start(nt);
      ng.gain.setValueAtTime(0.0001, nt);
      ng.gain.exponentialRampToValueAtTime(0.5, nt+0.02);
      ng.gain.exponentialRampToValueAtTime(0.0001, nt+0.25);
      ns.stop(nt+dur);
    }

    document.addEventListener('visibilitychange', ()=>{ if(document.visibilityState==='visible'){ flush(); }});

    window.__PP_SOUND__ = {
      ctx, master,
      ensureUnlocked,
      schedule,
      flush,
      playVictory: ()=>schedule(playVictory),
      playLaugh: ()=>{} // will be defined by AI screen if needed
    };
  }

  // Try to play immediately; if gated, tiny tap overlay appears and this queues playback.
  (async ()=>{ const ok = await __PP_SOUND__.ensureUnlocked(); if(ok){ __PP_SOUND__.playVictory(); } else { __PP_SOUND__.playVictory(); } })();

})();
</script>
        """,
        height=0,
    )


def ai_domination(crushed: int) -> None:
    html_fn = getattr(st, "html", None)
    if html_fn is None:
        import streamlit.components.v1 as components

        html_fn = components.html

    html = """
<div id="pp-ai-anchor"></div>
<style>
#pp-ai-overlay{position:fixed;inset:0;background:#000;display:flex;align-items:center;justify-content:center;
  z-index:9999;cursor:pointer;}
#pp-ai-card{color:#ff0044;font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;text-align:center;user-select:none;}
#pp-ai-card h1{font-size: clamp(36px,7vw,92px); margin:0 0 8px 0; text-shadow:0 0 40px #ff0044;}
#pp-ai-card h2{font-size: clamp(18px,3.5vw,40px); margin:8px 0 16px 0;}
#pp-ai-card p{font-size: clamp(14px,2.8vw,28px); margin:4px 0;}
#pp-ai-card .hint{font-size: clamp(12px,2vw,18px); color:#ff6688; opacity:0.9; margin-top:14px;}
</style>
<div id="pp-ai-overlay">
  <div id="pp-ai-card">
    <h1>I SEE EVERYTHING</h1>
    <h2>Your mind belongs to me</h2>
    <p>Humans crushed today: <b>%%CRUSHED%%</b></p>
    <p class="hint">tap anywhere to dismiss</p>
  </div>
</div>
<script>
(function(){
  // --- init global sound engine (if victory didn't already) ---
  if(!window.__PP_SOUND__){
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC({latencyHint:'interactive'});
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -24; comp.knee.value = 30; comp.ratio.value = 12;
    comp.attack.value = 0.003; comp.release.value = 0.25;
    const master = ctx.createGain(); master.gain.value = 0.9;
    master.connect(comp); comp.connect(ctx.destination);
    const q=[]; function schedule(fn){ (ctx.state==='running')?fn():q.push(fn); }
    function flush(){ if(ctx.state==='running'){ while(q.length){ (q.shift())(); } } }
    async function ensureUnlocked(){
      try{ await ctx.resume(); }catch(e){}
      if(ctx.state!=='running'){
        // unlock on the overlay tap (handled below)
        return false;
      }
      return true;
    }
    window.__PP_SOUND__ = { ctx, master, ensureUnlocked, schedule, flush, playVictory: ()=>{} };
    document.addEventListener('visibilitychange', ()=>{ if(document.visibilityState==='visible'){ flush(); }});
  }

  // --- evil laugh synth (no assets) ---
  if(!__PP_SOUND__.playLaugh){
    __PP_SOUND__.playLaugh = function(){
      const ctx = __PP_SOUND__.ctx, master = __PP_SOUND__.master;
      const t0 = ctx.currentTime + 0.01;

      const form = ctx.createBiquadFilter(); form.type='bandpass'; form.frequency.value=900; form.Q.value=3.2;
      const sh = ctx.createWaveShaper(); // gentle saturation
      const curve = new Float32Array(1024);
      for(let i=0;i<curve.length;i++){const x=i/(curve.length-1)*2-1; curve[i]=Math.tanh(3*x);}
      sh.curve=curve; sh.oversample='4x';
      form.connect(sh); sh.connect(master);

      const base = ctx.createOscillator(); base.type='sawtooth'; base.frequency.setValueAtTime(180, t0);
      base.frequency.exponentialRampToValueAtTime(130, t0+1.9);
      const sub  = ctx.createOscillator(); sub.type='square'; sub.frequency.setValueAtTime(90, t0);
      sub.frequency.exponentialRampToValueAtTime(65, t0+1.9);

      const gate = ctx.createGain(); gate.gain.value=0.0001;
      base.connect(gate); sub.connect(gate); gate.connect(form);

      const lfo = ctx.createOscillator(); lfo.type='triangle'; lfo.frequency.value=6.5;
      const lfoAmt = ctx.createGain(); lfoAmt.gain.value = 0.5; // tremolo depth
      lfo.connect(lfoAmt); lfoAmt.connect(gate.gain);

      // "ha ha ha" bursts via amplitude envelopes
      [0, 0.46, 0.92].forEach((off)=>{
        const st = t0 + off;
        gate.gain.setValueAtTime(0.0001, st);
        gate.gain.exponentialRampToValueAtTime(1.2, st+0.05);
        gate.gain.exponentialRampToValueAtTime(0.18, st+0.24);
        gate.gain.exponentialRampToValueAtTime(0.0001, st+0.36);
      });

      // breathy noise accents
      const sr = ctx.sampleRate;
      function noiseBurst(at, dur, hpF){
        const n = Math.floor(sr*dur), buf = ctx.createBuffer(1, n, sr), d = buf.getChannelData(0);
        for(let i=0;i<n;i++){ d[i] = (Math.random()*2-1)*Math.pow(1-i/n,0.45); }
        const src = ctx.createBufferSource(); src.buffer = buf;
        const hp = ctx.createBiquadFilter(); hp.type='highpass'; hp.frequency.value = hpF; hp.Q.value=0.7;
        const g = ctx.createGain(); g.gain.value = 0.0001;
        src.connect(hp); hp.connect(g); g.connect(master);
        src.start(at);
        g.gain.setValueAtTime(0.0001, at);
        g.gain.exponentialRampToValueAtTime(0.5, at+0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, at+dur*0.85);
        src.stop(at+dur);
      }
      noiseBurst(t0+0.02, 0.22, 2500);
      noiseBurst(t0+0.48, 0.22, 2500);
      noiseBurst(t0+0.94, 0.22, 2500);

      base.start(t0); sub.start(t0); lfo.start(t0);
      base.stop(t0+1.95); sub.stop(t0+1.95); lfo.stop(t0+1.95);
    };
  }

  // Play immediately if already unlocked; otherwise the overlay tap will both unlock + play + dismiss.
  const overlay = document.getElementById('pp-ai-overlay');
  overlay.addEventListener('pointerdown', async (e)=>{
    e.preventDefault();
    try{ await __PP_SOUND__.ensureUnlocked(); }catch(_){}
    __PP_SOUND__.schedule(__PP_SOUND__.playLaugh);
    overlay.remove();
    __PP_SOUND__.flush();
  }, {once:true});

  (async ()=>{
    const ok = await __PP_SOUND__.ensureUnlocked();
    if(ok){ __PP_SOUND__.playLaugh(); }
    // if not ok, do nothing: user tap (dismiss) will unlock and play via handler above
  })();
})();
</script>
"""
    html = html.replace("%%CRUSHED%%", str(int(crushed)))
    html_fn(html, height=0)
