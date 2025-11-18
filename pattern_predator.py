import os
import pickle
import random
import sys
import time

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
    guess_delay = 0.5  # seconds per AI guess reveal


# ====================== ML CORE ======================
class FeatureExtractor:
    def __init__(self):
        self.dim = Config.feature_dim

    def encode(self, history):
        phi = np.zeros(self.dim)
        if len(history) >= 1:
            phi[history[-1]] = 1
        if len(history) >= 2:
            phi[6 + history[-2]] = 1
        streak = 1
        for i in range(len(history) - 2, -1, -1):
            if history[i] == history[-1]:
                streak += 1
            else:
                break
        phi[12] = streak / 5.0
        if history:
            phi[13] = len(set(history)) / len(history)
            for i in range(3):
                phi[14 + i] = history.count(i) / len(history)
        return phi


class LinearModel:
    def __init__(self, dim):
        self.w = np.zeros((3, dim))
        self.b = np.zeros(3)

    def predict_probs(self, phi):
        logits = np.dot(self.w, phi) + self.b
        return np.exp(logits) / np.sum(np.exp(logits))

    def update(self, phi, target, delta):
        probs = self.predict_probs(phi)
        grad = np.outer(probs, phi)
        grad[target] -= phi
        self.w -= Config.alpha * delta * grad
        self.b -= Config.alpha * delta * probs
        self.b[target] += Config.alpha * delta


class Agent:
    def __init__(self, model, extractor):
        self.model = model
        self.extractor = extractor

    def guess_next(self, history, epsilon):
        phi = self.extractor.encode(history)
        probs = self.model.predict_probs(phi)
        return (
            random.randint(0, 2) if random.random() < epsilon else int(np.argmax(probs))
        )


class Trainer:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.model = LinearModel(Config.feature_dim)
        self.global_stats = {"plays": 0, "ai_wins": 0}
        self.ai_level = "Easy"
        self.load_model()
        self.bootstrap_if_needed()

    def bootstrap_if_needed(self):
        if not os.path.exists("model.pkl"):
            for _ in range(Config.bootstrap_games):
                history = [
                    random.randint(0, 2)
                    for _ in range(random.randint(1, Config.sequence_length - 1))
                ]
                target = random.randint(0, 2)
                phi = self.extractor.encode(history)
                guess = self.guess(history, 0.5)
                delta = 1 if guess == target else -1
                self.model.update(phi, target, delta)
            self.save_model()

    def guess(self, history, epsilon):
        return self.agent.guess_next(history, epsilon)

    @property
    def agent(self):
        return Agent(self.model, self.extractor)

    def train_from_game(self, user_seq, ai_guesses):
        history = []
        for u, a in zip(user_seq, ai_guesses):
            phi = self.extractor.encode(history)
            delta = 1 if a == u else -1
            self.model.update(phi, u, delta)
            history.append(u)
        self.save_model()
        # Score calculation
        ai_score = sum(1 for u, a in zip(user_seq, ai_guesses) if u == a)
        user_score = Config.sequence_length - ai_score
        ai_streak = max(
            (
                len(list(g))
                for k, g in __import__("itertools").groupby(
                    (u == a for u, a in zip(user_seq, ai_guesses))
                )
                if k
            ),
            default=0,
        )
        user_streak = max(
            (
                len(list(g))
                for k, g in __import__("itertools").groupby(
                    (u != a for u, a in zip(user_seq, ai_guesses))
                )
                if k
            ),
            default=0,
        )
        ai_score += 1 if ai_streak >= 3 else 0
        user_score += 1 if user_streak >= 3 else 0
        self.global_stats["plays"] += 1
        if ai_score > user_score:
            self.global_stats["ai_wins"] += 1
        self.ai_level = (
            "Hard"
            if self.global_stats["ai_wins"] / self.global_stats["plays"] > 0.6
            else "Easy"
        )
        return user_score, ai_score

    def save_model(self):
        with open("model.pkl", "wb") as f:
            pickle.dump(
                (self.model.w, self.model.b, self.global_stats, self.ai_level), f
            )

    def load_model(self):
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                self.model.w, self.model.b, self.global_stats, self.ai_level = (
                    pickle.load(f)
                )


# ====================== CELEBRATIONS ======================
def human_victory():
    st.balloons()
    st.markdown(
        "<h1 style='text-align:center; color:#00ff41; text-shadow:0 0 30px #00ff41;'>YOU DEFEATED THE AI!</h1><h2 style='text-align:center; color:#00ff41;'>HUMANITY PREVAILS</h2>",
        unsafe_allow_html=True,
    )
    st.audio(
        "https://cdn.pixabay.com/download/audio/2022/03/15/audio_2e6e4a19c5.mp3?filename=success-fanfare-trumpets-6185.mp3",
        format="audio/mp3",
        autoplay=True,
    )


def ai_domination(crushed_count):
    st.markdown(
        f"""
    <style>.big {{font-size:90px !important; font-weight:bold;}}</style>
    <div style="position:fixed;top:0;left:0;width:100%;height:100%;background:#000;opacity:0.95;z-index:9998;"></div>
    <div style="position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:9999;text-align:center;color:#ff0044;">
        <h1 class="big" style="text-shadow:0 0 40px #ff0044;">I SEE EVERYTHING</h1>
        <h2>Your mind belongs to me</h2>
        <p style="font-size:28px;">Humans crushed today: <b>{crushed_count}</b></p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.image("https://i.imgur.com/8QJ9Y9j.gif", use_column_width=True)
    st.audio(
        "https://cdn.pixabay.com/download/audio/2023/10/26/audio_2c1c2a1f3e.mp3?filename=evil-laugh-6297.mp3",
        format="audio/mp3",
        autoplay=True,
    )


# ====================== MAIN APP ======================
def main():
    st.title("Pattern Predator 🧠")
    st.markdown(
        "Outsmart the AI! Pick a 5-shape sequence—it'll try to predict you. Best of 3 rounds."
    )
    trainer = Trainer()

    # Init session state
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

    # Sidebar
    with st.sidebar:
        st.header("AI Stats")
        plays = trainer.global_stats["plays"]
        win_rate = (trainer.global_stats["ai_wins"] / plays * 100) if plays else 0
        st.write(f"Level: **{trainer.ai_level}**")
        st.write(f"Global AI Win %: {win_rate:.1f}%")
        st.write(f"Plays Today: {plays}")
        if st.button("Reset Game"):
            [st.session_state.update({k: v}) for k, v in defaults.items()]
            st.rerun()
        try:
            url = st.secrets["app_url"]
        except:
            url = "http://localhost:8501"
        st.markdown(
            f"Share: [LinkedIn Post](https://www.linkedin.com/sharing/share-offsite/?url={url})"
        )

    shapes = ["⭕️", "■", "🔺"]

    # Input phase
    if (
        len(st.session_state.user_sequence) < Config.sequence_length
        and not st.session_state.predicting
    ):
        cols = st.columns(3)
        for i, shape in enumerate(shapes):
            if cols[i].button(shape, key=f"btn_{i}"):
                st.session_state.user_sequence.append(i)
                st.rerun()

    seq_display = "".join(shapes[i] for i in st.session_state.user_sequence) + "_" * (
        Config.sequence_length - len(st.session_state.user_sequence)
    )
    st.write(f"Your Sequence: **{seq_display}**")

    # Submit button
    if (
        len(st.session_state.user_sequence) == Config.sequence_length
        and not st.session_state.predicting
    ):
        if st.button("Submit & Let AI Predict"):
            st.session_state.predicting = True
            st.session_state.current_guess = 0
            st.session_state.ai_guesses = []
            st.session_state.history = []
            st.rerun()

    # Prediction animation
    if (
        st.session_state.predicting
        and st.session_state.current_guess < Config.sequence_length
    ):
        with st.spinner(
            f"AI reading your mind... ({st.session_state.current_guess + 1}/{Config.sequence_length})"
        ):
            time.sleep(Config.guess_delay)
            epsilon = (
                Config.epsilon_easy
                if trainer.ai_level == "Easy"
                else Config.epsilon_hard
            )
            guess = trainer.guess(st.session_state.history, epsilon)
            st.session_state.ai_guesses.append(guess)
            st.session_state.history.append(
                st.session_state.user_sequence[st.session_state.current_guess]
            )
            st.session_state.current_guess += 1
            st.rerun()

    # Reveal results
    if (
        st.session_state.current_guess == Config.sequence_length
        and not st.session_state.reveal_complete
    ):
        st.session_state.reveal_complete = True
        st.rerun()

    if st.session_state.reveal_complete:
        st.write(
            f"AI Guesses: {''.join(shapes[g] for g in st.session_state.ai_guesses)}"
        )
        user_score, ai_score = trainer.train_from_game(
            st.session_state.user_sequence, st.session_state.ai_guesses
        )
        st.write(f"Scores → You: **{user_score}** | AI: **{ai_score}**")

        if user_score > ai_score:
            st.session_state.scores["user"] += 1
            human_victory()
        else:
            st.session_state.scores["ai"] += 1
            ai_domination(trainer.global_stats["ai_wins"])

        st.write(
            f"Overall: You {st.session_state.scores['user']} – AI {st.session_state.scores['ai']}"
        )
        st.success("AI learned from your play!")

        if max(st.session_state.scores.values()) < Config.rounds_to_win:
            st.session_state.round += 1
            time.sleep(2)
            st.session_state.user_sequence = []
            st.session_state.ai_guesses = []
            st.session_state.predicting = False
            st.session_state.current_guess = 0
            st.session_state.history = []
            st.session_state.reveal_complete = False
            st.rerun()
        else:
            winner = (
                "You"
                if st.session_state.scores["user"] > st.session_state.scores["ai"]
                else "AI"
                if st.session_state.scores["ai"] > st.session_state.scores["user"]
                else "Tie"
            )
            st.header(f"Game Over: **{winner} Wins!**")
            share_text = f"I {'beat' if winner == 'You' else 'got owned by'} the AI in Pattern Predator! {st.session_state.scores['user']}-{st.session_state.scores['ai']} Try it: [link] #BeatTheAI"
            st.text_area("Share this:", share_text)
            if st.button("New Game"):
                for k in defaults:
                    st.session_state[k] = defaults[k]
                st.rerun()


# ====================== STANDALONE TRAINING MODE FOR GITHUB ACTIONS ======================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("Starting automated training...")
        trainer = Trainer()
        for _ in range(15000):
            seq = [random.randint(0, 2) for _ in range(Config.sequence_length)]
            guesses = []
            hist = []
            eps = (
                Config.epsilon_easy
                if trainer.ai_level == "Easy"
                else Config.epsilon_hard
            )
            for i in range(Config.sequence_length):
                g = trainer.guess(hist, eps)
                guesses.append(g)
                hist.append(seq[i])
            trainer.train_from_game(seq, guesses)
        print(
            f"Training complete. AI win rate: {trainer.global_stats['ai_wins'] / trainer.global_stats['plays']:.1%}"
        )
    else:
        main()
