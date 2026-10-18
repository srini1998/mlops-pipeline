export type Match = {
  season: string;
  date: string;
  home: string;
  away: string;
  hg: number;
  ag: number;
};
export type Example = Match & { x: number[]; y: number };
export type FootballModel = {
  weights: number[];
  count: number;
  through: string;
  name: string;
};
const sigmoid = (v: number) =>
  1 / (1 + Math.exp(-Math.max(-30, Math.min(30, v))));
export function makeExamples(matches: Match[]): Example[] {
  const rows = [...matches].sort(
    (a, b) => a.date.localeCompare(b.date) || a.home.localeCompare(b.home),
  );
  const ratings = new Map<string, number>(),
    form = new Map<string, { points: number; gd: number }[]>();
  const out: Example[] = [];
  const avg = (team: string, key: "points" | "gd") => {
    const a = form.get(team) || [];
    return a.length ? a.reduce((s, v) => s + v[key], 0) / a.length : 0;
  };
  for (let i = 0; i < rows.length;) {
    let end = i + 1;
    while (end < rows.length && rows[end].date === rows[i].date) end++;
    // Build every match on a date before applying any result from that date.
    for (const m of rows.slice(i, end)) {
      if (
        !Number.isInteger(m.hg) ||
        !Number.isInteger(m.ag) ||
        m.hg < 0 ||
        m.ag < 0
      )
        throw Error("Invalid score");
      out.push({
        ...m,
        x: [
          1,
          ((ratings.get(m.home) ?? 1500) - (ratings.get(m.away) ?? 1500)) / 400,
          (avg(m.home, "points") - avg(m.away, "points")) / 3,
          (avg(m.home, "gd") - avg(m.away, "gd")) / 3,
        ],
        y: m.hg > m.ag ? 1 : 0,
      });
    }
    for (const m of rows.slice(i, end)) {
      const h = ratings.get(m.home) ?? 1500,
        a = ratings.get(m.away) ?? 1500,
        actual = m.hg > m.ag ? 1 : m.hg === m.ag ? 0.5 : 0,
        expected = 1 / (1 + 10 ** ((a - h - 65) / 400)),
        delta = 20 * (actual - expected);
      ratings.set(m.home, h + delta);
      ratings.set(m.away, a - delta);
      for (const [team, points, gd] of [
        [m.home, m.hg > m.ag ? 3 : m.hg === m.ag ? 1 : 0, m.hg - m.ag],
        [m.away, m.ag > m.hg ? 3 : m.hg === m.ag ? 1 : 0, m.ag - m.hg],
      ] as [string, number, number][]) {
        form.set(team, [...(form.get(team) || []), { points, gd }].slice(-5));
      }
    }
    i = end;
  }
  return out;
}
export function trainFootball(
  examples: Example[],
  before: string,
  name: string,
): FootballModel {
  const rows = examples.filter((m) => m.date < before);
  if (rows.length < 30) throw Error("Need at least 30 earlier matches");
  const weights = [0, 0, 0, 0];
  for (let epoch = 0; epoch < 250; epoch++) {
    const gradient = [0, 0, 0, 0];
    for (const r of rows) {
      const err = sigmoid(weights.reduce((s, w, j) => s + w * r.x[j], 0)) - r.y;
      for (let j = 0; j < 4; j++) gradient[j] += err * r.x[j];
    }
    for (let j = 0; j < 4; j++)
      weights[j] -=
        0.18 * (gradient[j] / rows.length + (j ? 0.015 * weights[j] : 0));
  }
  return { weights, count: rows.length, through: rows.at(-1)!.date, name };
}
export function probability(m: FootballModel, x: number[]) {
  return sigmoid(m.weights.reduce((s, w, j) => s + w * x[j], 0));
}
export function metrics(m: FootballModel, rows: Example[]) {
  if (!rows.length) throw Error("No evaluation matches");
  if (rows.some((r) => r.date <= m.through))
    throw Error("Evaluation overlaps training");
  let correct = 0,
    brier = 0;
  for (const r of rows) {
    const p = probability(m, r.x);
    correct += Number(Number(p >= 0.5) === r.y);
    brier += (p - r.y) ** 2;
  }
  return {
    n: rows.length,
    accuracy: correct / rows.length,
    brier: brier / rows.length,
  };
}
export function gateFootball(
  current: FootballModel,
  candidate: FootballModel,
  validation: Example[],
  required = 0.005,
) {
  const a = metrics(current, validation),
    b = metrics(candidate, validation);
  return {
    current: a,
    candidate: b,
    pass: b.brier <= a.brier - required && b.accuracy >= a.accuracy,
    required,
  };
}
