# Tour Round Predictor

Pick a PGA Tour player, a course, the weather and the wind. The page predicts the score and simulates
the tournament 10,000 times to estimate the chance of winning the round, making the cut, and winning
the tournament.

Live site: https://jjbenxu.github.io/golf-predictor/

## How it works

- **Score** = course difficulty + weather effect + player skill, fitted on PGA Tour rounds since 2021.
- **Odds** come from a Monte Carlo simulation that runs in the browser against the field from the
  most recent edition of the event.

## Refreshing the data

```bash
python3 pipeline/fetch_data.py    # download scores and weather into data/
python3 pipeline/build_model.py   # refit and write docs/model.js
python3 pipeline/backtest.py      # test against the 2026 season
```

Scores come from public leaderboards and weather from Open-Meteo. A hobby project, not betting advice.
