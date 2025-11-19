import os
import pickle
import random
import sys
import time

import numpy as np


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

    def encode(self, history: list[int]) -> np.ndarray:
        phi = np.zeros(self.dim)
        if not history:
            return phi
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
        exp = np.exp(logits - logits.max())
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

    def guess_next(self, history: list[int], epsilon: float) -> int:
        phi = self.extractor.encode(history)
        return (
            random.randint(0, 2)
            if random.random() < epsilon
            else int(np.argmax(self.model.predict_probs(phi)))
        )


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

    def guess(self, history: list[int], epsilon: float) -> int:
        return Agent(self.model, self.extractor).guess_next(history, epsilon)

    def train_from_game(
        self, user_seq: list[int], ai_guesses: list[int]
    ) -> tuple[int, int]:
        history: list[int] = []
        for u, a in zip(user_seq, ai_guesses):
            phi = self.extractor.encode(history)
            delta = 1 if a == u else -1
            self.model.update(phi, u, delta)
            history.append(u)

        self.save_model()

        ai_score = sum(1 for u, a in zip(user_seq, ai_guesses) if u == a)
        user_score = Config.sequence_length - ai_score

        from itertools import groupby

        ai_streak = max(
            (
                len(list(g))
                for k, g in groupby((u == a for u, a in zip(user_seq, ai_guesses)))
                if k
            ),
            default=0,
        )
        user_streak = max(
            (
                len(list(g))
                for k, g in groupby((u != a for u, a in zip(user_seq, ai_guesses)))
                if k
            ),
            default=0,
        )
        ai_score += 1 if ai_streak >= 3 else 0
        user_score += 1 if user_streak >= 3 else 0

        self.global_stats["plays"] += 1
        if ai_score > user_score:
            self.global_stats["ai_wins"] += 1
        win_rate = (
            self.global_stats["ai_wins"] / self.global_stats["plays"]
            if self.global_stats["plays"]
            else 0
        )
        self.ai_level = "Hard" if win_rate > 0.6 else "Easy"

        return user_score, ai_score

    def save_model(self) -> None:
        with open("model.pkl", "wb") as f:
            pickle.dump(
                (self.model.w, self.model.b, self.global_stats, self.ai_level), f
            )

    def load_model(self) -> None:
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                self.model.w, self.model.b, self.global_stats, self.ai_level = (
                    pickle.load(f)
                )


# ====================== FINAL SOUND & CELEBRATIONS ======================
def human_victory() -> None:
    import streamlit as st  # Local import — absent in --train mode

    st.balloons()
    st.markdown(
        "<h1 style='text-align:center; color:#00ff41; text-shadow:0 0 30px #00ff41;'>YOU DEFEATED THE AI!</h1>"
        "<h2 style='text-align:center; color:#00ff41;'>HUMANITY STILL REIGNS</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <script>
        setTimeout(() => {
            const audio = new Audio('https://cdn.pixabay.com/download/audio/2022/03/15/audio_2e6e4a19c5.mp3');
            audio.volume = 0.8;
            audio.play().catch(() => {});
        }, 100);
        </script>
        """,
        unsafe_allow_html=True,
    )


def ai_domination(crushed: int) -> None:
    import streamlit as st  # Local import — absent in --train mode

    st.markdown(
        f"""
        <div id="ai-takeover" style="position:fixed;top:0;left:0;width:100%;height:100%;background:#000;opacity:0.95;z-index:9999;
            display:flex;flex-direction:column;justify-content:center;align-items:center;cursor:pointer;color:#ff0044;
            font-family:system-ui;text-align:center;" onclick="this.style.display='none';">
            <h1 style="font-size:90px;text-shadow:0 0 40px #ff0044;margin:0;">I SEE EVERYTHING</h1>
            <h2 style="font-size:40px;margin:20px 0;">Your mind belongs to me</h2>
            <p style="font-size:28px;">Humans crushed today: <b>{crushed}</b></p>
            <p style="font-size:18px;color:#ff6688;opacity:0.9;margin-top:20px;">tap anywhere to dismiss</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image("https://i.imgur.com/8QJ9Y9j.gif", use_column_width=True)
    st.markdown(
        """
        <script>
        setTimeout(() => {
            const audio = new Audio('https://cdn.pixabay.com/download/audio/2023/10/26/audio_2c1c2a1f3e.mp3');
            audio.volume = 0.9;
            audio.play().catch(() => {});
        }, 300);
        </script>
        """,
        unsafe_allow_html=True,
    )


# ====================== MAIN APP ======================
def main() -> None:
    import streamlit as st  # Imported only in UI mode

    st.set_page_config(page_title="Pattern Predator", page_icon="🧠")
    st.title("Pattern Predator 🧠")
    st.markdown(
        "Outsmart the AI! Pick a 5-shape sequence — it'll try to predict you. Best of 3 rounds."
    )

    trainer = Trainer()

    if "round" not in st.session_state:
        st.session_state.round = 1
        st.session_state.scores = {"user": 0, "ai": 0}
        st.session_state.user_sequence = []
        st.session_state.ai_guesses = []
        st.session_state.predicting = False
        st.session_state.current_guess = 0
        st.session_state.history = []
        st.session_state.reveal_complete = False

    with st.sidebar:
        st.header("AI Stats")
        plays = trainer.global_stats["plays"]
        win_rate = (trainer.global_stats["ai_wins"] / plays * 100) if plays else 0
        st.write(f"**Level:** {trainer.ai_level}")
        st.write(f"**Global AI Win %:** {win_rate:.1f}%")
        st.write(f"**Plays Today:** {plays}")
        if st.button("Reset Game"):
            st.session_state.round = 1
            st.session_state.scores = {"user": 0, "ai": 0}
            st.session_state.user_sequence = []
            st.session_state.ai_guesses = []
            st.session_state.predicting = False
            st.session_state.current_guess = 0
            st.session_state.history = []
            st.session_state.reveal_complete = False
            st.rerun()

        url = getattr(st, "secrets", {}).get("app_url", "http://localhost:8501")
        st.markdown(
            f"Share: [LinkedIn Post](https://www.linkedin.com/sharing/share-offsite/?url={url})"
        )

    shapes = ["⭕️", "■", "🔺"]

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
    st.write(f"**Your Sequence:** {seq_display}")

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

    if (
        st.session_state.current_guess == Config.sequence_length
        and not st.session_state.reveal_complete
    ):
        st.session_state.reveal_complete = True
        st.rerun()

    if st.session_state.reveal_complete:
        st.write(
            f"**AI Guesses:** {''.join(shapes[g] for g in st.session_state.ai_guesses)}"
        )
        user_score, ai_score = trainer.train_from_game(
            st.session_state.user_sequence, st.session_state.ai_guesses
        )
        st.write(f"**Scores** → You: **{user_score}** | AI: **{ai_score}**")

        if user_score > ai_score:
            st.session_state.scores["user"] += 1
            human_victory()
        else:
            st.session_state.scores["ai"] += 1
            ai_domination(trainer.global_stats["ai_wins"])

        st.write(
            f"**Overall:** You {st.session_state.scores['user']} – AI {st.session_state.scores['ai']}"
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
            )
            st.header(f"**Game Over: {winner} Wins!**")
            share_text = (
                f"I {'beat' if winner == 'You' else 'got mind-read by'} the AI in "
                f"Pattern Predator! {st.session_state.scores['user']}-"
                f"{st.session_state.scores['ai']} Dare you? [link] #BeatTheAI"
            )
            st.text_area("Share this:", share_text)
            if st.button("New Game"):
                st.session_state.round = 1
                st.session_state.scores = {"user": 0, "ai": 0}
                st.session_state.user_sequence = []
                st.session_state.ai_guesses = []
                st.session_state.predicting = False
                st.session_state.current_guess = 0
                st.session_state.history = []
                st.session_state.reveal_complete = False
                st.rerun()


# ====================== AUTOMATED TRAINING MODE ======================
if __name__ == "__main__":
    if "--train" in sys.argv[1:]:
        print("Starting automated self-play training...")
        trainer = Trainer()
        for _ in range(20000):
            seq = [random.randint(0, 2) for _ in range(Config.sequence_length)]
            guesses: list[int] = []
            hist: list[int] = []
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
        win_rate = (
            trainer.global_stats["ai_wins"] / trainer.global_stats["plays"]
            if trainer.global_stats["plays"]
            else 0
        )
        print(
            f"Training complete! AI win rate: {win_rate:.2%} | "
            f"Total plays: {trainer.global_stats['plays']:,}"
        )
    else:
        try:
            main()
        except ModuleNotFoundError as exc:
            if "streamlit" in str(exc):
                print(
                    "Streamlit is required for the UI. Install with "
                    "'pip install streamlit' or run with '--train' for headless mode."
                )
                sys.exit(1)
            raise
