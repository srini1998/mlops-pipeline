# Matchday model release lab

Primary UI now uses 1520 real CC0 OpenFootball Premier League matches across four seasons. Four-weight logistic regression trains in the browser with pre-match Elo and last-five-match form features. Features for an entire date are built before applying any scores from that date. The current model trains on 2021/22 (380 matches); challenger trains on 2021/22–2023/24 (1140). Validation is Aug–Dec 2024 (188); later test is Jan–May 2025 (192). User activates/rolls back a model only inside their tab.

Measured default gate rejects: Brier 0.22750495 -> 0.22736454 is less than the required 0.005 improvement. The optional zero-margin exploratory gate accepts this tiny gain with non-decreasing accuracy. Later test accuracy is 60.94% vs 63.54%; this is not a powered result or future prediction claim. Repeated UI exploration is not a locked benchmark. No betting odds, injuries or lineups are included; probabilities are uncalibrated.

Six engine checks cover source count, train split, future-label independence, same-day feature isolation, overlap rejection and a deliberately strict gate. Real Chrome train/reject/accept/activate/rollback/reveal/mobile flows pass. Original electricity lab and its five checks remain at /electricity; underlying Python/MLflow history is preserved. Both UIs are English-only and need no downloaded LLM. Optional cloud explanation on electricity requires an explicitly confirmed free Groq account/key; no credentials or paid fallback configured.

Refetch data: python3 scripts/fetch-football.py. Re-evaluate: node tests/football.mjs. Build: npm run build.
