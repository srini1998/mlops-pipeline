// Deterministic checks applied to the optional cloud reviewer's explanation.
// The model may only restate the gate verdict and the measured numbers it was given.
export type GateFacts = {
  pass: boolean;
  required: number;
  current: { brier: number; accuracy: number };
  candidate: { brier: number; accuracy: number };
  matches: number;
};
export const factStrings = (g: GateFacts) => ({
  currentBrier: g.current.brier.toFixed(5),
  challengerBrier: g.candidate.brier.toFixed(5),
  improvement: (g.current.brier - g.candidate.brier).toFixed(5),
  required: g.required.toFixed(3),
  currentAccuracy: (g.current.accuracy * 100).toFixed(1) + "%",
  challengerAccuracy: (g.candidate.accuracy * 100).toFixed(1) + "%",
  matches: String(g.matches),
});
export function reviewPrompt(g: GateFacts) {
  const f = factStrings(g);
  return `A release gate compared two win/no-win soccer models on ${f.matches} held-out Premier League matches (Aug–Dec 2024).
Rule: promote only if challenger Brier error is at least ${f.required} lower AND challenger accuracy is not lower.
Current model: Brier ${f.currentBrier}, accuracy ${f.currentAccuracy}.
Challenger: Brier ${f.challengerBrier}, accuracy ${f.challengerAccuracy}. Brier improvement: ${f.improvement}.
Code verdict: ${g.pass ? "PROMOTE" : "KEEP"}.
Write the first line exactly "VERDICT: ${g.pass ? "PROMOTE" : "KEEP"}". Then at most 3 short sentences for a product manager explaining why, using only the numbers above, copied exactly.`;
}
const numberPattern = /-?\d+(?:\.\d+)?%?/g;
export function checkReview(text: string, g: GateFacts) {
  const f = factStrings(g);
  const allowed = new Set<string>([...Object.values(f), f.improvement.replace(/^-/, ""), "3", "2024", "1", "2", "0"]);
  const verdict = text.match(/VERDICT:\s*(PROMOTE|KEEP)/i)?.[1]?.toUpperCase() ?? null;
  const verdictOk = verdict === (g.pass ? "PROMOTE" : "KEEP");
  const numbers = (text.replace(/VERDICT:[^\n]*/i, "").match(numberPattern) ?? []).map((n) => n.replace(/^-/, ""));
  const unknown = numbers.filter((n) => !allowed.has(n) && !allowed.has(n.replace(/%$/, "")));
  return { verdict, verdictOk, numbers, unknown, ok: verdictOk && unknown.length === 0 };
}
