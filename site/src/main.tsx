import { createRoot } from "react-dom/client";
import { useMemo, useState, lazy, Suspense } from "react";
import matches from "./data/football.json";
import source from "./data/football-provenance.json";
import {
  makeExamples,
  trainFootball,
  metrics,
  gateFootball,
  probability,
  type FootballModel,
} from "./football-engine";
import { Shell, Trace } from "./ui";
import "./styles.css";
const Electricity = lazy(() =>
  import("./ElectricityLab").then((m) => ({ default: m.ElectricityLab })),
);
const pct = (n: number) => (n * 100).toFixed(1) + "%";
const examples = makeExamples(matches),
  validation = examples.filter(
    (m) => m.date >= "2024-08-01" && m.date < "2025-01-01",
  ),
  test = examples.filter((m) => m.date >= "2025-01-01");
function App() {
  const initial = useMemo(
    () => trainFootball(examples, "2022-08-01", "First-season model"),
    [],
  );
  const [active, setActive] = useState(initial),
    [previous, setPrevious] = useState<FootballModel | null>(null),
    [candidate, setCandidate] = useState<FootballModel | null>(null),
    [idx, setIdx] = useState(0),
    [reveal, setReveal] = useState(false),
    [required, setRequired] = useState(0.005),
    [entries, setEntries] = useState<unknown[]>([]);
  const gate = candidate
      ? gateFootball(active, candidate, validation, required)
      : null,
    result = metrics(active, test),
    match = test[idx],
    prediction = probability(active, match.x);
  const log = (e: unknown) => setEntries((a) => [...a, e]);
  return (
    <Shell name="current / matchday" number="03" ja={false} toggle={() => {}}>
      <section className="hero match-hero">
        <div>
          <p className="eyebrow">FOOTBALL MEETS MACHINE LEARNING</p>
          <h1>
            New season.
            <br />
            Better model?
          </h1>
          <p className="intro">
            Teach a small model using real Premier League results. See whether
            three seasons of experience beat one, then decide which model gets
            the next match.
          </p>
          <a className="cta" href="#lab">
            Train your challenger ↗
          </a>
          <p className="note">
            Actual browser training. No account, cloud bill or AI download.
          </p>
        </div>
        <div className="pitch" aria-hidden="true">
          <div className="halfway" />
          <div className="pitch-circle" />
          <span className="pitch-node node-a">TRAIN</span>
          <span className="pitch-node node-b">TEST</span>
          <span className="pitch-node node-c">RELEASE</span>
          <div className="pitch-caption">1,520 MATCHES / 4 SEASONS</div>
        </div>
      </section>
      <main id="lab">
        <div className="section-head">
          <div>
            <p className="eyebrow">YOU PICK THE STARTING MODEL</p>
            <h2>Earn the next release.</h2>
          </div>
          <span>Historical replay · 2021–2025</span>
        </div>
        <p className="intro">
          Predict one simple outcome: does the home team win? A draw and an away
          win both count as “no.” We use only team strength and form available
          before kickoff.
        </p>
        <div className="lab">
          <section className="controls">
            <h3>1. Train on earlier seasons</h3>
            <p>
              Your starting model learned from {initial.count} matches in
              2021/22. The challenger learns from the first three seasons.
              Neither trains on 2024/25 outcomes.
            </p>
            <button
              className="cta"
              onClick={() => {
                const m = trainFootball(
                  examples,
                  "2024-08-01",
                  "Three-season model",
                );
                setCandidate(m);
                log({
                  action: "train",
                  count: m.count,
                  through: m.through,
                  weights: m.weights,
                });
              }}
            >
              Train challenger
            </button>
            <label htmlFor="required" style={{ marginTop: 25 }}>
              Required improvement in probability error
            </label>
            <select
              id="required"
              value={required}
              onChange={(e) => {
                setRequired(Number(e.target.value));
                log({
                  action: "change-gate",
                  required: Number(e.target.value),
                });
              }}
            >
              <option value={0}>Any measured improvement (exploratory)</option>
              <option value={0.005}>0.005 lower Brier error</option>
              <option value={0.02}>0.020 lower Brier error</option>
              <option value={0.5}>0.500 lower (deliberately strict)</option>
            </select>
            <p className="note">
              Brier error measures how close a probability was to the actual
              outcome. Lower is better. The challenger must also match or beat
              the current model’s win/no-win accuracy.
            </p>
            <h3>2. Review before switching</h3>
            <button
              disabled={!gate?.pass}
              onClick={() => {
                if (!candidate || !gate?.pass) return;
                setPrevious(active);
                setActive(candidate);
                setCandidate(null);
                log({ action: "activate", name: candidate.name, gate });
              }}
            >
              Use this model
            </button>
            <button
              disabled={!previous}
              onClick={() => {
                if (!previous) return;
                setActive(previous);
                setPrevious(null);
                setCandidate(null);
                log({ action: "rollback" });
              }}
            >
              Roll back
            </button>
            <p className="note">
              This changes the active model in this browser tab. No production
              service is modified.
            </p>
          </section>
          <section className="decision" aria-live="polite">
            <p className="decision-label">
              VALIDATION / AUG–DEC 2024 · {validation.length} MATCHES
            </p>
            <h3>
              {gate
                ? gate.pass
                  ? "The challenger earned a start."
                  : "Keep your current model."
                : "A longer training history has to prove itself."}
            </h3>
            <p>
              {gate
                ? "These matches were held out of training. The separate later test below measures what happened next."
                : "Train the challenger to compare both models on exactly the same matches."}
            </p>
            {gate && (
              <div className="comparison">
                <div>
                  <strong>{gate.current.brier.toFixed(5)}</strong>
                  <span>Current Brier error</span>
                </div>
                <div>
                  <strong>{gate.candidate.brier.toFixed(5)}</strong>
                  <span>Challenger Brier error</span>
                </div>
                <div>
                  <strong>{pct(gate.candidate.accuracy)}</strong>
                  <span>Challenger accuracy</span>
                </div>
              </div>
            )}
            <details>
              <summary>What goes into the model?</summary>
              <p>
                Four logistic regression weights: a constant, pre-match Elo
                difference, last-five-match points difference and
                last-five-match goal difference. Full-batch gradient descent
                runs here. Team state is updated only after all features for
                that date are built.
              </p>
            </details>
          </section>
        </div>
        <section className="football-test">
          <div className="section-head">
            <div>
              <p className="eyebrow">3 / THE LATER MATCHES</p>
              <h2>How did it do next?</h2>
            </div>
            <span>JAN–MAY 2025 · {test.length} matches</span>
          </div>
          <div className="comparison">
            <div>
              <strong>{pct(result.accuracy)}</strong>
              <span>Active model accuracy</span>
            </div>
            <div>
              <strong>{result.brier.toFixed(3)}</strong>
              <span>Active model Brier error</span>
            </div>
            <div>
              <strong>
                {pct(test.filter((r) => r.y === 0).length / test.length)}
              </strong>
              <span>Always predict “no home win” accuracy</span>
            </div>
          </div>
          <p className="note">
            The later set is separate from release validation. These are
            historical measurements, not live forecasts. Browsing results
            repeatedly makes this an exploratory demo, not a locked competition
            benchmark.
          </p>
          <label htmlFor="match">Replay a historical match</label>
          <select
            id="match"
            value={idx}
            onChange={(e) => {
              setIdx(Number(e.target.value));
              setReveal(false);
            }}
          >
            {test.map((m, i) => (
              <option key={m.date + m.home} value={i}>
                {m.date} · {m.home} vs {m.away}
              </option>
            ))}
          </select>
          <div className="match-card">
            <p>
              {match.date} / {active.name}
            </p>
            <h3>
              {match.home}
              <span>vs</span>
              {match.away}
            </h3>
            <strong>{pct(prediction)}</strong>
            <p>Estimated home-win probability from pre-match features</p>
            <button onClick={() => setReveal((r) => !r)}>
              {reveal ? "Hide result" : "Reveal actual result"}
            </button>
            {reveal && (
              <p role="status">
                Final score: {match.home} {match.hg}–{match.ag} {match.away}.
                Home win: {match.y ? "yes" : "no"}.
              </p>
            )}
          </div>
        </section>
        <section id="how" className="how">
          <p className="eyebrow">THE DATA BEHIND THE MATCHDAY</p>
          <h2>Real matches. Small model. Clear limits.</h2>
          <p>
            This is a release-management tool illustrated through football:
            train, compare, activate and roll back. It helps show why a newer
            model should not replace a working model automatically.
          </p>
          <p>
            All {source.rows} completed results come from{" "}
            <a href="https://github.com/openfootball/football.json">
              OpenFootball’s CC0 data ↗
            </a>
            . The snapshot spans the 2021/22–2024/25 Premier League seasons. No
            injuries, lineups or odds are included. New/promoted teams start
            with neutral ratings. Match probabilities are not calibrated betting
            advice.
          </p>
          <p>
            <a href="/FOOTBALL-LICENSE.md">Dataset license ↗</a> ·{" "}
            <a href="/electricity">
              Explore the original electricity release lab ↗
            </a>
          </p>
          <details>
            <summary>Snapshot provenance and checksums</summary>
            <pre>{JSON.stringify(source, null, 2)}</pre>
          </details>
        </section>
      </main>
      <Trace entries={entries} ja={false} />
    </Shell>
  );
}
createRoot(document.getElementById("root")!).render(
  location.pathname === "/electricity" ? (
    <Suspense fallback={<p>Opening electricity lab…</p>}>
      <Electricity />
    </Suspense>
  ) : (
    <App />
  ),
);
