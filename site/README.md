# Current — a model release rehearsal
Run `npm ci && npm run dev`. All training and inference run in the visitor's browser. Real Elec2/OpenML151 history, source SHA and attribution in src/data/provenance.json. No live market data or trading claim.

The compact browser logistic model predicts each row's UP/DOWN label from the preceding row's measurements and direction only. A challenger trains on five earlier weeks, validates on the following week, and must beat the incumbent by the selected margin AND majority-class accuracy +2 percentage points. Explicit promotion changes the active weights; rollback restores the previous weights. Moving the week resets to the initial champion so later-trained models cannot be evaluated in earlier history. The selected future week never enters model training or the gate, but users can explore outcomes repeatedly: this UI is not a blinded statistical study.

Optional Gemma 3 1B q4 runs on device to explain the measured decision. It has no release authority. The recorded Python gradient-boosting/MLflow pipeline is distinct from this browser implementation; no claim of equal scores. ThreeUI Community StreamConvergenceBackground (MIT), GSAP entrance respects reduced motion, EN/JA interface.

`node tests/engine.mjs` checks chronology, gate rejection and future-label independence. Browser evidence is in delivery/evidence/current-browser.json in the workspace. No cloud MLflow, paid APIs or infrastructure are provisioned.
