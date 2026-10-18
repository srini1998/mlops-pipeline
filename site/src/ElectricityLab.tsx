import { createRoot } from "react-dom/client";
import { useState, useMemo } from "react";
import { StreamConvergenceBackground } from "@designcodeio/threeui/components/StreamConvergenceBackground";
import { Lightning, ArrowUpRight } from "@phosphor-icons/react";
import rows from "./data/electricity.json";
import source from "./data/provenance.json";
import recorded from "./data/recorded.json";
import { train, score, evaluate, type Model } from "./engine";
import { useModel } from "./ai/useModel";
import { Shell, Trace, ModelControl, useLanguage } from "./ui";
import "./styles.css";
const pct = (v: number) => (v * 100).toFixed(1) + "%";
export function ElectricityLab() {
  const { ja, t, toggle } = useLanguage(),
    ai = useModel();
  const initial = useMemo(() => train(rows, 1, 7 * 336), []);
  const [champ, setChamp] = useState<Model>(initial),
    [previous, setPrevious] = useState<Model | null>(null),
    [week, setWeek] = useState(8),
    [gain, setGain] = useState(0.01),
    [result, setResult] = useState<ReturnType<typeof evaluate> | null>(null),
    [entries, setEntries] = useState<unknown[]>([]),
    [explanation, setExplanation] = useState(""),
    [busy, setBusy] = useState(false);
  const live = score(champ, rows, week * 336, (week + 1) * 336),
    staticScore = score(initial, rows, week * 336, (week + 1) * 336);
  const log = (e: unknown) => setEntries((a) => [...a, e]);
  function run() {
    const r = evaluate(rows, week, champ, gain);
    setResult(r);
    setExplanation("");
    log({
      action: "train/evaluate",
      trainedThrough: r.candidate.trainedThrough,
      validationStart: (week - 1) * 336,
      week,
      gain,
      challenger: r.challenger,
      champion: r.champion,
      floor: r.floor,
      pass: r.pass,
    });
  }
  function promote() {
    if (!result?.pass || result.week !== week || result.gain !== gain) return;
    setPrevious(champ);
    setChamp(result.candidate);
    setResult(null);
    setExplanation("");
    log({ action: "promote", week, model: result.candidate });
  }
  async function explain() {
    if (!result) return;
    setBusy(true);
    try {
      const prompt = `Explain this model release decision in 3 short sentences. Do not change the decision or invent values. Gate: ${result.pass ? "PASS" : "KEEP CURRENT"}. Challenger holdout ${pct(result.challenger.accuracy)}; current ${pct(result.champion.accuracy)}; required gain ${pct(gain)}; floor ${pct(result.floor)}. Historical electricity data, not a forecast. ${ja ? "Reply in Japanese." : ""}`;
      const r = await ai.generate(
        [{ role: "user", content: prompt }],
        undefined,
        160,
      );
      setExplanation(r.text);
      log({ action: "explain", prompt, ...r });
    } catch (e) {
      setExplanation(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Shell name="current" number="03" ja={ja} toggle={toggle}>
      <section className="hero">
        <div className="flow" aria-hidden="true">
          <StreamConvergenceBackground
            hue={0.55}
            saturation={0.4}
            speed={0.35}
          />
        </div>
        <div>
          <p className="eyebrow reveal">
            {t(
              "A BETTER MODEL. EARNED, NOT ASSUMED.",
              "良いモデルかどうか、測定して決める。",
            )}
          </p>
          <h1 className="reveal">
            {t("Keep your", "モデルを")}
            <br />
            {t("model current.", "新鮮に。")}
          </h1>
          <p className="intro reveal">
            {t(
              "Electricity markets change. Your model should earn its next release. Train a challenger, test it on unseen history, then decide what goes live in this replay.",
              "電力市場は変わります。新しいモデルを学習し、未知の履歴で比較してから、このリプレイで使うモデルを選びましょう。",
            )}
          </p>
          <a className="cta reveal" href="#lab">
            {t("Try a release", "リリースを試す")}
            <ArrowUpRight size={20} />
          </a>
        </div>
        <div className="hero-stat reveal">
          <Lightning size={34} />
          <strong>45,312</strong>
          <p>{t("real half-hour records", "実際の30分間隔レコード")}</p>
          <span>NSW / ELEC2 / HISTORICAL</span>
        </div>
      </section>
      <main id="lab">
        <div className="section-head">
          <div>
            <p className="eyebrow">
              {t("YOU ARE THE RELEASE MANAGER", "あなたがリリース担当者")}
            </p>
            <h2>{t("Train. Compare. Decide.", "学習。比較。判断。")}</h2>
          </div>
          <span>
            {t("Week", "週")} {week} / 100
          </span>
        </div>
        <div className="lab">
          <section className="controls">
            <label>
              {t("Replay week", "リプレイの週")}
              <input
                disabled={busy}
                aria-label="Replay week"
                type="range"
                min="8"
                max="100"
                value={week}
                onChange={(e) => {
                  setWeek(Number(e.target.value));
                  setChamp(initial);
                  setPrevious(null);
                  setResult(null);
                  setExplanation("");
                }}
              />
            </label>
            <p>
              {t(
                "Moving weeks resets the active model. The challenger learns from the previous five weeks. The next completed week is held out for the gate. The selected week is a separate future check.",
                "挑戦モデルは過去5週間で学習。次の1週間を評価用に保留し、選択した週は別の将来チェックに使います。",
              )}
            </p>
            <label>
              {t("Required improvement", "必要な改善幅")}
              <select
                disabled={busy}
                value={gain}
                onChange={(e) => {
                  setGain(Number(e.target.value));
                  setResult(null);
                  setExplanation("");
                }}
              >
                {[0, 0.01, 0.03, 0.05].map((n) => (
                  <option key={n} value={n}>
                    {pct(n)} {t("percentage points", "ポイント")}
                  </option>
                ))}
              </select>
            </label>
            <button className="cta" onClick={run} disabled={busy}>
              {t("Train & evaluate", "学習して比較")}
            </button>
            <p className="note">
              {t(
                "Browser logistic regression, 30 epochs. No labels from the holdout or future week enter training.",
                "ブラウザのロジスティック回帰、30エポック。評価週や将来週の正解は学習に使いません。",
              )}
            </p>
          </section>
          <section className="decision" aria-live="polite">
            <div className="decision-label">
              {result
                ? result.pass
                  ? t("READY FOR REVIEW", "確認して昇格")
                  : t("KEEP THE CURRENT MODEL", "現在のモデルを維持")
                : t("WAITING FOR A CHALLENGER", "挑戦モデルの準備待ち")}
            </div>
            <h3>
              {result
                ? result.pass
                  ? t("It cleared the gate.", "条件を満たしました。")
                  : t("This one stays on the bench.", "今回は見送ります。")
                : t("Better is a measurement.", "良いかどうかは測定する。")}
            </h3>
            {result ? (
              <>
                <div className="comparison">
                  <div>
                    <span>{t("Challenger holdout", "挑戦モデル")}</span>
                    <strong>{pct(result.challenger.accuracy)}</strong>
                  </div>
                  <div>
                    <span>{t("Current holdout", "現在のモデル")}</span>
                    <strong>{pct(result.champion.accuracy)}</strong>
                  </div>
                  <div>
                    <span>{t("Minimum floor", "最低基準")}</span>
                    <strong>{pct(result.floor)}</strong>
                  </div>
                </div>
                <p>
                  {t(
                    "Pass requires the chosen improvement AND majority-class accuracy + 2 points. A pass is permission to compare, not proof of future profit.",
                    "改善幅と、多数派正解率＋2ポイントの両方が必要です。合格しても将来の収益を保証しません。",
                  )}
                </p>
                <button
                  className="cta"
                  onClick={promote}
                  disabled={busy || !result.pass}
                >
                  {t("Promote challenger", "挑戦モデルを昇格")}
                </button>
              </>
            ) : (
              <p>
                {t(
                  "Choose a week, then train a candidate. A release never happens just because a model sounds confident.",
                  "週を選び、候補を学習します。AI の説明だけでリリースは行いません。",
                )}
              </p>
            )}
            <button
              className="quiet"
              disabled={busy || !previous}
              onClick={() => {
                if (previous) {
                  setChamp(previous);
                  setPrevious(null);
                  setResult(null);
                  setExplanation("");
                  log({ action: "rollback", week });
                }
              }}
            >
              {t("Roll back last promotion", "直前の昇格を戻す")}
            </button>
          </section>
        </div>
        <section className="future">
          <header>
            <h3>{t("How it handles the next week", "次の週での実測")}</h3>
            <p>
              {t(
                "This is held out from both training and release decisions.",
                "学習にもリリース判断にも使わないデータです。",
              )}
            </p>
          </header>
          <div className="comparison">
            <div>
              <span>{t("Active model", "使用中モデル")}</span>
              <strong>{pct(live.accuracy)}</strong>
            </div>
            <div>
              <span>{t("Never retrained", "再学習なし")}</span>
              <strong>{pct(staticScore.accuracy)}</strong>
            </div>
            <div>
              <span>{t("Repeat last direction", "前回と同じ方向")}</span>
              <strong>{pct(live.persistence)}</strong>
            </div>
          </div>
          <svg
            role="img"
            aria-label={t(
              "Actual normalized price in selected week",
              "選択週の実際の正規化価格",
            )}
            viewBox="0 0 900 140"
          >
            <polyline
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              points={rows
                .slice(week * 336, (week + 1) * 336)
                .map((r, i) => `${(i * 900) / 335},${130 - r[2] * 120}`)
                .join(" ")}
            />
          </svg>
          <p className="note">
            {t(
              "Line = actual normalized NSW price, not model predictions. Predictions classify UP/DOWN using only prior-period fields. This compact browser model is different from the Python gradient-boosting pipeline.",
              "線は実際の正規化価格で、予測値ではありません。前の時間帯の情報だけで上昇・下落を分類します。Python 版の勾配ブースティングとは異なる簡易モデルです。",
            )}
          </p>
        </section>
        <section className="assistant">
          <div>
            <h3>{t("Make the decision readable.", "判断をわかりやすく。")}</h3>
            <p>
              {t(
                "Ask the browser AI to explain the measured gate. It can suggest what to inspect; it cannot promote a model or override a failed check.",
                "ブラウザAIが測定結果を説明します。確認点を提案できますが、昇格や不合格の取り消しはできません。",
              )}
            </p>
          </div>
          <div>
            <ModelControl
              state={ai.state}
              load={ai.load}
              stop={ai.stop}
              ja={ja}
            />
            <button
              onClick={explain}
              disabled={busy || !result || ai.state.phase !== "ready"}
            >
              {busy
                ? t("Writing…", "作成中…")
                : t("Explain this decision", "判断を説明する")}
            </button>
            <output>{explanation}</output>
          </div>
        </section>
        <section id="how" className="how">
          <p className="eyebrow">{t("WHAT THIS IS FOR", "何に役立つか")}</p>
          <h2>
            {t(
              "A safe rehearsal for model releases.",
              "モデルリリースのリハーサル。",
            )}
          </h2>
          <p>
            {t(
              "Imagine a retailer refreshing a demand classifier every week. New data does not automatically mean a better model. This workbench makes the release gate tangible: keep the incumbent when a challenger regresses, inspect its unseen-week accuracy, and roll back a mistaken promotion.",
              "小売店が毎週需要モデルを更新する場面を想像してください。新データがあっても改善とは限りません。候補が悪化したら現行を維持し、未知の週を確認し、必要なら元に戻します。",
            )}
          </p>
          <div className="record">
            <strong>88.2% / 85.8% / 79.5%</strong>
            <p>
              {t(
                "Recorded Python pipeline / persistence / static model, weeks 16–99. Different model and evaluation from the live browser replay.",
                "記録済み Python パイプライン／前回方向／固定モデル、16〜99週。上のブラウザとは別のモデル・評価です。",
              )}
            </p>
            <span>
              {recorded.weeks_replayed}{" "}
              {t("weeks replayed locally", "週間をローカル再生")}
            </span>
          </div>
          <a href={source.source}>Elec2 / OpenML 151 ↗</a> ·{" "}
          <a href="https://github.com/srini1998/mlops-pipeline">
            {t("Source & full methodology", "ソースと評価方法")} ↗
          </a>
          <details>
            <summary>{t("Data provenance", "データの出典")}</summary>
            <pre>{JSON.stringify(source, null, 2)}</pre>
          </details>
        </section>
      </main>
      <Trace ja={ja} entries={entries} />
    </Shell>
  );
}
