# Pattern Predator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pattern-predator.streamlit.app/) <!-- Update this link to your actual deployed URL -->

**Pattern Predator** is a fun, addictive web-based game where you challenge an AI to predict your sequence of shapes. Built with Python, NumPy, and Streamlit, the AI learns in real-time from player inputs using a simple machine learning model (linear predictor with TD updates). It's quick to play (under 1 minute per game), viral-friendly with shareable scores, and demonstrates basic ML concepts in an engaging way.

Play it live: [Pattern Predator App](https://pattern-predator.streamlit.app/) <!-- Replace with your Streamlit URL -->

## Features
- **Simple Gameplay**: Pick a 5-shape sequence (Circle ⭕️, Square ■, Triangle 🔺). The AI guesses your picks—score points by stumping it!
- **AI Learning**: The model bootstraps with random simulations and adapts from every game played, leveling up from "Easy" to "Hard" based on win rates.
- **Addictive Elements**: Best-of-3 rounds, streak bonuses, balloons on wins, and global stats for FOMO.
- **Viral Sharing**: Auto-generated share text for LinkedIn/Twitter, e.g., "I beat the AI 2-1! Can you? #BeatTheAI".
- **Tech Stack**: NumPy for ML, Streamlit for UI—deployable in minutes.

## Installation (For Local Development)
1. Clone the repo:
   ```
   git clone https://github.com/your-username/pattern-predator.git
   cd pattern-predator
   ```
2. Set up a virtual environment (Python 3.9+):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install streamlit numpy
   ```
4. Run locally:
   ```
   streamlit run pattern_predator.py
   ```
   Open http://localhost:8501 in your browser.

## Usage
- **Play**: Tap shapes to build your sequence, submit, and watch the AI predict. Compete in rounds!
- **Train the AI**: Every game updates the model (saved as `model.pkl` locally or on the server).
- **Deploy**: Push to GitHub and deploy free on [Streamlit Cloud](https://streamlit.io/cloud). It auto-bootsraps on first run.

## How It Works
- **Game Loop**: User inputs sequence → AI guesses via ε-greedy policy → Scores with streaks → TD updates for learning.
- **ML Details**: Linear model predicts next shape probs (softmax). Features: History one-hots, streaks, entropy. Trained on self-play + user data.
- **Persistence**: Model and stats saved/loaded via Pickle for continual evolution.

## Contributing
Pull requests welcome! Fork the repo, make changes, and submit a PR. Ideas: Add more shapes, deeper ML (e.g., LSTM), or multiplayer.

## License
MIT License. See [LICENSE](LICENSE) for details.

---

Built with help from Grok. Share your high scores! 🚀
