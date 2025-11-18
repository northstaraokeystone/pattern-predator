import os
import pickle
import random

import numpy as np
import streamlit as st


# Config (simple hypers)
class Config:
    num_choices = 3  # Circle=0, Square=1, Triangle=2
    sequence_length = 5
    feature_dim = 18
    alpha = 0.02
    epsilon_easy = 0.3
    epsilon_hard = 0.05
    rounds_to_win = 3  # Best of 3
    bootstrap_games = 100


# Feature Extractor: Simple history
class FeatureExtractor:
    def __init__(self):
        self.dim = Config.feature_dim

    def encode(self, history):  # history = list of past choices
        phi = np.zeros(self.dim)
        # One-hot last 2 (6 dims each)
        if len(history) >= 1:
            phi[history[-1]] = 1
        if len(history) >= 2:
            phi[6 + history[-2]] = 1
        # Streak (12): length of current repeat
        streak = 1
        for i in range(len(history) - 2, -1, -1):
            if history[i] == history[-1]:
                streak += 1
            else:
                break
        phi[12] = streak / 5.0  # norm
        # Entropy approx (13): unique count / len
        if history:
            phi[13] = len(set(history)) / len(history)
        # Style bits (14-17): freq per choice
        if history:
            for i in range(3):
                phi[14 + i] = history.count(i) / len(history)
        return phi


# Model: Linear prob predictor (softmax over 3)
class LinearModel:
    def __init__(self, dim):
        self.w = np.zeros((3, dim))  # weights for each choice
        self.b = np.zeros(3)

    def predict_probs(self, phi):
        logits = np.dot(self.w, phi) + self.b
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

    def update(self, phi, target, delta):
        probs = self.predict_probs(phi)
        grad = np.outer(probs, phi)  # semi-grad approx
        grad[target] -= phi  # for correct
        self.w -= Config.alpha * delta * grad
        self.b -= Config.alpha * delta * probs
        self.b[target] += Config.alpha * delta  # boost correct


# Agent: Picks guess
class Agent:
    def __init__(self, model, extractor):
        self.model = model
        self.extractor = extractor

    def guess_next(self, history, epsilon):
        phi = self.extractor.encode(history)
        probs = self.model.predict_probs(phi)
        if random.random() < epsilon:
            return random.randint(0, 2)
        return np.argmax(probs)


# Trainer: Handles learning
class Trainer:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.model = LinearModel(Config.feature_dim)
        self.global_stats = {"plays": 0, "ai_wins": 0}
        self.ai_level = "Easy"  # Set default before load/bootstrap
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
                guess = self.guess(history, 0.5)  # noisy
                delta = 1 if guess == target else -1
                self.model.update(phi, target, delta)
            self.save_model()

    def guess(self, history, epsilon):
        return self.agent.guess_next(history, epsilon)

    @property
    def agent(self):
        return Agent(self.model, self.extractor)

    def train_from_game(self, user_sequence, ai_guesses):
        history = []
        for step, (user_choice, ai_guess) in enumerate(zip(user_sequence, ai_guesses)):
            phi = self.extractor.encode(history)
            target = user_choice
            delta = 1 if ai_guess == target else -1
            self.model.update(phi, target, delta)
            history.append(user_choice)
        self.save_model()
        # Compute scores here (same as env.compute_scores logic)
        ai_score = sum(1 for u, a in zip(user_sequence, ai_guesses) if u == a)
        user_score = Config.sequence_length - ai_score
        ai_streak, user_streak = 0, 0
        max_ai, max_user = 0, 0
        for u, a in zip(user_sequence, ai_guesses):
            if u == a:
                ai_streak += 1
                user_streak = 0
                max_ai = max(max_ai, ai_streak)
            else:
                user_streak += 1
                ai_streak = 0
                max_user = max(max_user, user_streak)
        ai_score += 1 if max_ai >= 3 else 0
        user_score += 1 if max_user >= 3 else 0
        # Update stats
        self.global_stats["plays"] += 1
        if ai_score > user_score:
            self.global_stats["ai_wins"] += 1
        win_rate = (
            self.global_stats["ai_wins"] / self.global_stats["plays"]
            if self.global_stats["plays"]
            else 0
        )
        self.ai_level = "Hard" if win_rate > 0.6 else "Easy"

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


# Streamlit UI
def main():
    st.write(
        "Debug: Code Version 3 - With Secrets Fix"
    )  # To confirm you're running the latest
    st.title("Pattern Predator 🧠")
    st.markdown(
        "Outsmart the AI! Pick a 5-shape sequence—it'll try to predict you. Best of 3 rounds."
    )
    trainer = Trainer()

    if "round" not in st.session_state:
        st.session_state.round = 1
        st.session_state.scores = {"user": 0, "ai": 0}
        st.session_state.user_sequence = []
        st.session_state.ai_guesses = []

    # Sidebar: Stats & Share
    with st.sidebar:
        st.header("AI Stats")
        win_rate = (
            trainer.global_stats["ai_wins"] / trainer.global_stats["plays"] * 100
            if trainer.global_stats["plays"]
            else 0
        )
        st.write(f"Level: {trainer.ai_level}")
        st.write(f"Global AI Win %: {win_rate:.1f}%")
        st.write(f"Plays Today: {trainer.global_stats['plays']}")
        if st.button("Reset Game"):
            st.session_state.round = 1
            st.session_state.scores = {"user": 0, "ai": 0}
            st.session_state.user_sequence = []
            st.session_state.ai_guesses = []
        # Handle secrets gracefully
        try:
            url = st.secrets["app_url"]
        except:
            url = "http://localhost:8501"  # Default for local testing
        st.markdown(
            "Share your win: [LinkedIn Post](https://www.linkedin.com/sharing/share-offsite/?url={url})".format(
                url=url
            )
        )

    # Main: Input sequence - single row of buttons, append on click
    shapes = ["⭕️", "■", "🔺"]
    if len(st.session_state.user_sequence) < Config.sequence_length:
        col1, col2, col3 = st.columns(3)
        if col1.button(shapes[0], key="c0"):
            st.session_state.user_sequence.append(0)
        if col2.button(shapes[1], key="c1"):
            st.session_state.user_sequence.append(1)
        if col3.button(shapes[2], key="c2"):
            st.session_state.user_sequence.append(2)

    # Display current sequence
    seq_str = "".join(shapes[c] for c in st.session_state.user_sequence) + "_" * (
        Config.sequence_length - len(st.session_state.user_sequence)
    )
    st.write(f"Your Sequence: {seq_str}")

    if len(st.session_state.user_sequence) == Config.sequence_length and st.button(
        "Submit & Let AI Predict"
    ):
        with st.spinner("AI Predicting..."):
            history = []
            epsilon = (
                Config.epsilon_easy
                if trainer.ai_level == "Easy"
                else Config.epsilon_hard
            )
            for _ in range(Config.sequence_length):
                guess = trainer.guess(history, epsilon)
                st.session_state.ai_guesses.append(guess)
                history.append(
                    st.session_state.user_sequence[len(st.session_state.ai_guesses) - 1]
                )  # append actual after guess

        # Reveal
        ai_str = "".join(shapes[g] for g in st.session_state.ai_guesses)
        st.write(f"AI Guesses: {ai_str}")
        ai_score = sum(
            1
            for u, a in zip(st.session_state.user_sequence, st.session_state.ai_guesses)
            if u == a
        )
        user_score = Config.sequence_length - ai_score
        # Streak bonus (duplicated from compute_scores for simplicity here)
        ai_streak, user_streak = 0, 0
        max_ai, max_user = 0, 0
        for u, a in zip(st.session_state.user_sequence, st.session_state.ai_guesses):
            if u == a:
                ai_streak += 1
                user_streak = 0
                max_ai = max(max_ai, ai_streak)
            else:
                user_streak += 1
                ai_streak = 0
                max_user = max(max_user, user_streak)
        ai_score += 1 if max_ai >= 3 else 0
        user_score += 1 if max_user >= 3 else 0
        st.write(f"Scores: You {user_score} | AI {ai_score}")

        # Update scores
        if user_score > ai_score:
            st.session_state.scores["user"] += 1
            st.balloons()
        elif ai_score > user_score:
            st.session_state.scores["ai"] += 1
        st.write(
            f"Overall: You {st.session_state.scores['user']} | AI {st.session_state.scores['ai']}"
        )

        # Train
        trainer.train_from_game(
            st.session_state.user_sequence, st.session_state.ai_guesses
        )
        st.success("AI learned from your play!")

        # Next round or end
        if max(st.session_state.scores.values()) < Config.rounds_to_win:
            st.session_state.round += 1
            st.session_state.user_sequence = []
            st.session_state.ai_guesses = []
        else:
            winner = (
                "You"
                if st.session_state.scores["user"] > st.session_state.scores["ai"]
                else "AI"
                if st.session_state.scores["ai"] > st.session_state.scores["user"]
                else "Tie"
            )
            st.header(f"Game Over: {winner} Wins!")
            share_text = f"I {'beat' if winner == 'You' else 'lost to'} the AI in Pattern Predator! Score: {st.session_state.scores['user']}-{st.session_state.scores['ai']}. Try it: [link] #BeatTheAI"
            st.text_area("Share This:", share_text)
            if st.button("New Game"):
                st.session_state.round = 1
                st.session_state.scores = {"user": 0, "ai": 0}
                st.session_state.user_sequence = []
                st.session_state.ai_guesses = []


if __name__ == "__main__":
    main()
