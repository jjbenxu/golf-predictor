"""Honest test: fit the model on 2021-2025 only, then see how well it predicts the 2026 season."""
import numpy as np
from datetime import datetime, timedelta
import build_model as b

CUTOFF = "2026-01-01"


def run(n_sims=4000, seed=1, verbose=True):
    """Rolling test: before each 2026 month, refit using only rounds played before that month."""
    fits = {}

    def model_for(date):
        cut = max(CUTOFF, date[:7] + "-01")
        if cut not in fits:
            ev_, rows_, rk_ = b.load(cutoff=cut)
            fits[cut] = b.fit(ev_, rows_, rk_, ref_date=datetime.fromisoformat(cut).toordinal())
        return fits[cut]

    all_events, all_rows, _ = b.load()
    rng = np.random.default_rng(seed)

    m = None

    def skill_of(p):
        i = m["pid"].get(p)
        return (m["skill"][i], m["sd_mult"][i]) if i is not None else (m["prior"], 1.0)

    # group 2026 rounds by event and round
    test = {}
    for r in all_rows:
        if r[4] >= CUTOFF:
            test.setdefault(r[1], {}).setdefault(r[2], []).append(r)
    rel_pred, rel_act, abs_pred, abs_act = [], [], [], []
    cut_p, cut_y, win_p, win_n, top_hit = [], [], [], [], 0
    n_ev = 0
    for ei, rounds in test.items():
        e = all_events[ei]
        m = model_for(e["date"][:10])
        cidx = m["cid"].get(e["course"]["id"])
        by_per = {}
        for x in e["rounds"]:
            by_per.setdefault(x[2], {})[x[0]] = x[4]
        for per, sc in by_per.items():
            ids = list(sc)
            sk = np.array([skill_of(p)[0] for p in ids])
            act = np.array([sc[p] for p in ids], float)
            rel_pred += list(sk - sk.mean()); rel_act += list(act - act.mean())
            day = (datetime.fromisoformat(e["date"][:10]) + timedelta(days=per - 1)).date().isoformat()
            w = e["weather"].get(day)
            if cidx is not None and w:
                diff = (m["course_base"][cidx] + m["b_wind"] * (w["wind"] - b.WIND_REF) + m["b_rain"] * min(w["rain"], 15)
                        + m["b_temp"] * (w["temp"] - b.TEMP_REF) + (m["b_wkend"] if per >= 3 else 0))
                abs_pred.append(diff + sk.mean()); abs_act.append(act.mean())
        # cut and win probabilities for standard events: one course, cut after round 2, four rounds played
        if e.get("nCourses", 1) != 1 or e.get("cutRound") != 2 or "Q-School" in e["name"] or any(k not in by_per for k in (1, 2, 3, 4)):
            continue
        ids = list(by_per[1])
        if len(ids) < 100:
            continue
        n_ev += 1
        sk = np.array([skill_of(p)[0] for p in ids]); sd = np.sqrt(m["v0"]) * np.array([skill_of(p)[1] for p in ids])
        r12 = np.rint(sk * 2 + rng.normal(0, 1, (n_sims, len(ids))) * sd * np.sqrt(2))
        line = np.sort(r12, axis=1)[:, b.cut_rule(e) - 1][:, None]
        made = r12 <= line
        r34 = np.rint(sk * 2 + rng.normal(0, 1, (n_sims, len(ids))) * sd * np.sqrt(2))
        tot = np.where(made, r12 + r34, 1e9)
        best = tot.min(axis=1, keepdims=True)
        winshare = (tot == best) / (tot == best).sum(axis=1, keepdims=True)
        pc, pw = made.mean(axis=0), winshare.mean(axis=0)
        actual_cut = np.array([p in by_per.get(3, {}) for p in ids])
        cut_p += list(pc); cut_y += list(actual_cut)
        totals = {p: sum(by_per[k][p] for k in (1, 2, 3, 4)) for p in ids if all(p in by_per[k] for k in (1, 2, 3, 4))}
        lo = min(totals.values())
        winners = [p for p, t in totals.items() if t == lo]
        win_p.append(np.mean([pw[ids.index(p)] for p in winners])); win_n.append(len(ids))
        top_hit += any(ids[j] in winners for j in np.argsort(-pw)[:10])

    rel_pred, rel_act = np.array(rel_pred), np.array(rel_act)
    cut_p, cut_y = np.array(cut_p), np.array(cut_y, float)
    res = {
        "rounds": len(rel_act),
        "rmse_model": float(np.sqrt(np.mean((rel_act - rel_pred) ** 2))),
        "rmse_naive": float(np.sqrt(np.mean(rel_act ** 2))),
        "corr": float(np.corrcoef(rel_pred, rel_act)[0, 1]),
        "slope": float(np.sum(rel_pred * rel_act) / np.sum(rel_pred ** 2)),
        "diff_mae": float(np.mean(np.abs(np.array(abs_pred) - np.array(abs_act)))),
        "diff_n": len(abs_act),
        "cut_brier": float(np.mean((cut_p - cut_y) ** 2)),
        "cut_brier_naive": float(np.mean((cut_y.mean() - cut_y) ** 2)),
        "events": n_ev, "winner_avg_prob": float(np.mean(win_p)), "winner_naive": float(np.mean(1 / np.array(win_n))),
        "winner_in_top10": top_hit,
    }
    if verbose:
        print("2026 rounds tested: %d" % res["rounds"])
        print("player vs field, per round: model error %.3f vs %.3f with no player knowledge (correlation %.2f, slope %.2f)"
              % (res["rmse_model"], res["rmse_naive"], res["corr"], res["slope"]))
        print("round scoring average (course + weather): typical miss %.2f strokes over %d rounds" % (res["diff_mae"], res["diff_n"]))
        print("make-the-cut forecasts over %d events: Brier %.4f vs %.4f naive" % (n_ev, res["cut_brier"], res["cut_brier_naive"]))
        for lo_, hi_ in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
            k = (cut_p >= lo_) & (cut_p < hi_)
            if k.sum():
                print("   predicted %2.0f-%3.0f%%: said %.0f%%, happened %.0f%%  (%d players)"
                      % (lo_ * 100, min(hi_, 1) * 100, 100 * cut_p[k].mean(), 100 * cut_y[k].mean(), k.sum()))
        print("actual winners were given %.1f%% on average (blind guess %.1f%%); winner was in model's top 10 in %d of %d events"
              % (100 * res["winner_avg_prob"], 100 * res["winner_naive"], top_hit, n_ev))
    return res


if __name__ == "__main__":
    run()
