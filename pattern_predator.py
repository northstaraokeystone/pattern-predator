import random
import time
import sys
import pickle
import os
from typing import List

import numpy as np
import streamlit as st

# ====================== CONFIG ======================
class Config:
    num_choices = 3
    sequence_length = 5
    feature_dim = 18
    alpha = 0.02
    epsilon_easy = 0.3
    epsilon_hard = 0.05
    rounds_to_win = 3
    bootstrap_games = 100
    guess_delay = 0.5


# ====================== ML CORE ======================
class FeatureExtractor:
    def __init__(self) -> None:
        self.dim = Config.feature_dim

    def encode(self, history: List[int]) -> np.ndarray:
        phi = np.zeros(self.dim)
        if history:
            phi[history[-1]] = 1
            if len(history) >= 2:
                phi[ = phi.at[6 + history[-2]].set(1)
            streak = 1
            for i in range(len(history) - 2, -1, -1):
                if history[i] == history[-1]:
                    streak += 1
                else:
                    break
            phi[12] = streak / 5.0
            phi[13] = len(set(history)) / len(history)
            for i in range(3):
                phi[14 + i] = history.count(i) / len(history)
        return phi


class LinearModel:
    def __init__(self, dim: int) -> None:
        self.w = np.zeros((3, dim))
        self.b = np.zeros(3)

    def predict_probs(self, phi: np.ndarray) -> np.ndarray:
        logits = self.w @ phi + self.b
        exp = np.exp(logits)
        return exp / exp.sum()

    def update(self, phi: np.ndarray, target: int, delta: float) -> None:
        probs = self.predict_probs(phi)
        grad = np.outer(probs, phi)
        grad[target] -= phi
        self.w -= Config.alpha * delta * grad
        self.b -= Config.alpha * delta * probs
        self.b[target] += Config.alpha * delta


class Agent:
    def __init__(self, model: LinearModel, extractor: FeatureExtractor) -> None:
        self.model = model
        self.extractor = extractor

    def guess_next(self, history: List[int], epsilon: float) -> int:
        phi = self.extractor.encode(history)
        return random.randint(0, 2) if random.random() < epsilon else int(np.argmax(self.model.predict_probs(phi)))


class Trainer:
    def __init__(self) -> None:
        self.extractor = FeatureExtractor()
        self.model = LinearModel(Config.feature_dim)
        self.global_stats = {"plays": 0, "ai_wins": 0}
        self.ai_level = "Easy"
        self.load_model()
        self.bootstrap_if_needed()

    def bootstrap_if_needed(self) -> None:
        if not os.path.exists("model.pkl"):
            for _ in range(Config.bootstrap_games):
                history = [random.randint(0, 2) for _ in range(random.randint(1, Config.sequence_length - 1))]
                target = random.randint(0, 2)
                phi = self.extractor.encode(history)
                guess = self.guess(history, 0.5)
                delta = 1 if guess == target else -1
                self.model.update(phi, target, delta)
            self.save_model()

    def guess(self, history: List[int], epsilon: float) -> int:
        return Agent(self.model, self.extractor).guess_next(history, epsilon)

    def train_from_game(self, user_seq: List[int], ai_guesses: List[int]) -> tuple[int, int]:
        history: List[int] = []
        for u, a in zip(user_seq, ai_guesses):
            phi = self.extractor.encode(history)
            delta = 1 if a == u else -1
            self.model.update(phi, u, delta)
            history.append(u)

        self.save_model()

        ai_score = sum(1 for u, a in zip(user_seq, ai_guesses) if u == a)
        user_score = Config.sequence_length - ai_score

        from itertools import groupby
        ai_streak = max((len(list(g)) for k, g in groupby((u == a for u, a in zip(user_seq, ai_guesses))) if k), default=0)
        user_streak = max((len(list(g)) for k, g in groupby((u != a for u, a in zip(user_seq, ai_guesses))) if k), default=0)
        ai_score += 1 if ai_streak >= 3 else 0
        user_score += 1 if user_streak >= 3 else 0

        self.global_stats["plays"] += 1
        if ai_score > user_score:
            self.global_stats["ai_wins"] += 1
        self.ai_level = "Hard" if self.global_stats["ai_wins"] / self.global_stats["plays"] > 0.6 else "Easy"

        return user_score, ai_score

    def save_model(self) -> None:
        with open("model.pkl", "wb") as f:
            pickle.dump((self.model.w, self.model.b, self.global_stats, self.ai_level), f)

    def load_model(self) -> None:
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                self.model.w, self.model.b, self.global_stats, self.ai_level = pickle.load(f)


# ====================== FINAL SOUND ENGINE — WORKS EVERYWHERE ======================
def human_victory() -> None:
    st.balloons()
    st.markdown(
        "<h1 style='text-align:center; color:#00ff41; text-shadow:0 0 30px #00ff41;'>YOU DEFEATED THE AI!</h1>"
        "<h2 style='text-align:center; color:#00ff41;'>HUMANITY STILL REIGNS</h2>",
        unsafe_allow_html=True,
    )
    html_fn = getattr(st, "html", None) or __import__("streamlit.components.v1").html
    html_fn(
        """
        <div id="pp-snd-anchor"></div>
        <script>
        (function(){
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
                let gate = document.getElementById('pp-audio-gate');
                if(!gate){
                  gate = document.createElement('div');
                  gate.id = 'pp-audio-gate';
                  gate.style.cssText = 'position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.75);color:#fff;padding:10px 14px;border-radius:10px;font:600 14px system-ui,Arial;z-index:10000;cursor:pointer;box-shadow:0 4px 20px rgba(0,0,0,0.35)';
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
            function playVictory(){
              const t0 = ctx.currentTime + 0.01;
              const arp = [523.25, 659.25, 783.99, 1046.50];
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
              const triad = [523.25, 659.25, 783.99];
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
            }
            window.__PP_SOUND__ = { ctx, master, ensureUnlocked, schedule, flush, playVictory: ()=>schedule(playVictory) };
            document.addEventListener('visibilitychange', ()=>{ if(document.visibilityState==='visible'){ flush(); }});
          }
          (async ()=>{ const ok = await window.__PP_SOUND__.ensureUnlocked(); if(ok){ window.__PP_SOUND__.playVictory(); } else { window.__PP_SOUND__.playVictory(); } })();
        })();
        </script>
        """,
        height=0,
    )


def ai_domination(crushed: int) -> None:
    html_fn = getattr(st, "html", None) or __import__("streamlit.components.v1").html
    html = f"""
    <div id="pp-ai-overlay" style="position:fixed;inset:0;background:#000;display:flex;align-items:center;justify-content:center;z-index:9999;cursor:pointer;">
      <div style="color:#ff0044;font-family:system-ui;text-align:center;">
        <h1 style="font-size:clamp(36px,7vw,92px);margin:0;text-shadow:0 0 40px #ff0044;">I SEE EVERYTHING</h1>
        <h2 style="font-size:clamp(18px,3.5vw,40px);margin:16px 0;">Your mind belongs to me</h2>
        <p style="font-size:clamp(14px,2.8vw,28px);">Humans crushed today: <b>{crushed}</b></p>
        <p style="font-size:clamp(12px,2vw,18px);color:#ff6688;opacity:0.9;margin-top:20px;">tap to dismiss</p>
      </div>
    </div>
    <div id="pp-ai-anchor"></div>
    <script>
    (function(){
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
        async function ensureUnlocked(){ try{ await ctx.resume(); }catch(e){} return ctx.state==='running'; }
        window.__PP_SOUND__ = { ctx, master, ensureUnlocked, schedule, flush };
      }
      if(!window.__PP_SOUND__.playLaugh){
        window.__PP_SOUND__.playLaugh = function(){
          const ctx = window.__PP_SOUND__.ctx, master = window.__PP_SOUND__.master;
          const t0 = ctx.currentTime + 0.01;
          const base = ctx.createOscillator(); base.type='sawtooth'; base.frequency.setValueAtTime(180, t0);
          base.frequency.exponentialRampToValueAtTime(130, t0+1.9);
          const sub = ctx.createOscillator(); sub.type='square'; sub.frequency.setValueAtTime(90, t0);
          sub.frequency.exponentialRampToValueAtTime(65, t0+1.9);
          const gate = ctx.createGain(); gate.gain.value=0.0001;
          base.connect(gate); sub.connect(gate); gate.connect(master);
          const lfo = ctx.createOscillator(); lfo.type='triangle'; lfo.frequency.value=6.5;
          const lfoAmt = ctx.createGain(); lfoAmt.gain.value = 0.5;
          lfo.connect(lfoAmt); lfoAmt.connect(gate.gain);
          [0, 0.46, 0.92].forEach((off)=>{
            const st = t0 + off;
            gate.gain.setValueAtTime(0.0001, st);
            gate.gain.exponentialRampToValueAtTime(1.2, st+0.05);
            gate.gain.exponentialRampToValueAtTime(0.18, st+0.24);
            gate.gain.exponentialRampToValueAtTime(0.0001, st+0.36);
          });
          base.start(t0); sub.start(t0); lfo.start(t0);
          base.stop(t0+1.95); sub.stop(t0+1.95); lfo.stop(t0+1.95);
        };
      }
      const overlay = document.getElementById('pp-ai-overlay');
      overlay.addEventListener('pointerdown', async (e)=>{
        e.preventDefault();
        await window.__PP_SOUND__.ensureUnlocked();
        window.__PP_SOUND__.schedule(window.__PP_SOUND__.playLaugh);
        overlay.remove();
        window.__PP_SOUND__.flush();
      }, {once:true});
      (async ()=>{
        if(await window.__PP_SOUND__.ensureUnlocked()){
          window.__PP_SOUND__.playLaugh();
        }
      })();
    })();
    </script>
    """
    html_fn(html, height=0)
    st.image("https://i.imgur.com/8QJ9Y9j.gif", use_column_width=True)


# ====================== MAIN APP ======================
def main() -> None:
    st.set_page_config(page_title="Pattern Predator", page_icon="brain")
    st.title("Pattern Predator")
    st.markdown("Outsmart the AI! Pick a 5-shape sequence — it'll try to predict you. Best of 3 rounds.")

    trainer = Trainer()

    defaults = {
        "round": 1,
        "scores": {"user": 0, "ai": 0},
        "user_sequence": [],
        "ai_guesses": [],
        "predicting": False,
        "current_guess": 0,
        "history": [],
        "reveal_complete": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with st.sidebar:
        st.header("AI Stats")
        plays = trainer.global_stats["plays"]
        win_rate = (trainer.global_stats["ai_wins"] / plays * 100) if plays else 0
        st.write(f"**Level:** {trainer.ai_level}")
        st.write(f"**Global AI Win %:** {win_rate:.1f}%")
        st.write(f"**Plays Today:** {plays}")
        if st.button("Reset Game"):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()
        try:
            url = st.secrets["app_url"]
        except:
            url = "http://localhost:8501"
        st.markdown(f"Share: [LinkedIn Post](https://www.linkedin.com/sharing/share-offsite/?url={url})")

    shapes = ["circle", "square", "triangle"]

    if len(st.session_state.user_sequence) < Config.sequence_length and not st.session_state.predicting:
        cols = st.columns(3)
        for i, shape in enumerate(shapes):
            if cols[i].button(shape, key=f"btn_{i}"):
                st.session_state.user_sequence.append(i)
                st.rerun()

    seq_display = "".join(shapes[i] for i in st.session_state.user_sequence) + "_" * (
        Config.sequence_length - len(st.session_state.user_sequence)
    )
    st.write(f"**Your Sequence:** {seq_display}")

    if len(st.session_state.user_sequence) == Config.sequence_length and not st.session_state.predicting:
        if st.button("Submit & Let AI Predict"):
            st.session_state.predicting = True
            st.session_state.current_guess = 0
            st.session_state.ai_guesses = []
            st.session_state.history = []
            st.rerun()

    if st.session_state.predicting and st.session_state.current_guess < Config.sequence_length:
        with st.spinner(f"AI reading your mind... ({st.session_state.current_guess + 1}/{Config.sequence_length})"):
            time.sleep(Config.guess_delay)
            epsilon = Config.epsilon_easy if trainer.ai_level == "Easy" else Config.epsilon_hard
            guess = trainer.guess(st.session_state.history, epsilon)
            st.session_state.ai_guesses.append(guess)
            st.session_state.history.append(st.session_state.user_sequence[st.session_state.current_guess])
            st.session_state.current_guess += 1
            st.rerun()

    if st.session_state.current_guess == Config.sequence_length and not st.session_state.reveal_complete:
        st.session_state.reveal_complete = True
        st.rerun()

    if st.session_state.reveal_complete:
        st.write(f"**AI Guesses:** {''.join(shapes[g] for g in st.session_state.ai_guesses)}")
        user_score, ai_score = trainer.train_from_game(st.session_state.user_sequence, st.session_state.ai_guesses)
        st.write(f"**Scores** → You: **{user_score}** | AI: **{ai_score}**")

        if user_score > ai_score:
            st.session_state.scores["user"] += 1
            human_victory()
        else:
            st.session_state.scores["ai"] += 1
            ai_domination(trainer.global_stats["ai_wins"])

        st.write(f"**Overall:** You {st.session_state.scores['user']} – AI {st.session_state.scores['ai']}")
        st.success("AI learned from your play!")

        if max(st.session_state.scores.values()) < Config.rounds_to_win:
            st.session_state.round += 1
            time.sleep(2)
            for key in ["user_sequence", "ai_guesses", "predicting", "current_guess", "history", "reveal_complete"]:
                st.session_state[key] = [] if "sequence" in key or "guesses" in key or "history" in key else False
            st.rerun()
        else:
            winner = "You" if st.session_state.scores["user"] > st.session_state.scores["ai"] else "AI"
            st.header(f"**Game Over: {winner} Wins!**")
            st.text_area("Share this:", f"I {'beat' if winner=='You' else 'got mind-read by'} the AI! {st.session_state.scores['user']}-{st.session_state.scores['ai']} #PatternPredator")
            if st.button("New Game"):
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.rerun()


# ====================== AUTOMATED TRAINING MODE ======================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("Starting automated self-play training...")
        trainer = Trainer()
        for _ in range(15000):
            seq = [random.randint(0, 2) for _ in range(Config.sequence_length)]
            guesses = []
            hist = []
            eps = Config.epsilon_easy if trainer.ai_level == "Easy" else Config.epsilon_hard
            for i in range(Config.sequence_length):
                g = trainer.guess(hist, eps)
                guesses.append(g)
                hist.append(seq[i])
            trainer.train_from_game(seq, guesses)
        print(f"Training complete. AI win rate: {trainer.global_stats['ai_wins']/trainer.global_stats['plays']:.1%}")
    else:
        main()