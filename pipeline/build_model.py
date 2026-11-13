"""Fit the scoring model and write docs/model.js for the website.

Model:  score to par = course baseline + weather effect + player skill (+ player wind tendency) + luck
Run:    python3 pipeline/build_model.py            (fit on everything, write docs/model.js)
        python3 pipeline/build_model.py --backtest (fit on data before 2026, test on 2026)
"""
import json, os, sys
from datetime import datetime, timedelta
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HALF_LIFE_DAYS = 365.0   # a round one year old counts half as much as one played today
SHRINK_ROUNDS = 6.0      # players with little history are pulled toward a "newcomer" level
PRIOR = None             # skill assumed for a player with no history (None = measured newcomer average)
SPREAD = 1.0             # <1 pulls every player toward average (guards against overconfidence)
WIND_REF, TEMP_REF = 8.0, 70.0


def cut_rule(e):
    """Official cut rule (the feed only reports how many players actually survived)."""
    if not e.get("cutRound"):
        return 0
    n = e["name"]
    if "Masters" in n:
        return 50
    if "U.S. Open" in n:
        return 60
    if "PGA Championship" in n or "The Open" in n:
        return 70
    return 50 if len(e.get("field", [])) <= 80 else 65


def load(cutoff=None):
    events = json.load(open(os.path.join(ROOT, "data", "events.json")))
    rows = []  # (player, event index, round key, to_par, date, wind, rain, temp)
    rkeys = {}
    for ei, e in enumerate(events):
        off = e.get("utc_offset", 0)
        start = datetime.fromisoformat(e["date"][:10])
        per_dates = {}
        for r in e["rounds"]:
            if r[5]:
                d = (datetime.fromisoformat(r[5].replace("Z", "+00:00")) + timedelta(seconds=off)).date().isoformat()
                per_dates.setdefault(r[2], []).append(d)
        for r in e["rounds"]:
            per = r[2]
            ds = per_dates.get(per)
            day = max(set(ds), key=ds.count) if ds else (start + timedelta(days=per - 1)).date().isoformat()
            if cutoff and day >= cutoff:
                continue
            w = e["weather"].get(day)
            if not w:
                continue
            k = rkeys.setdefault((ei, per), len(rkeys))
            rows.append((r[0], ei, k, r[4], day, w["wind"], w["rain"], w["temp"], r[1]))
    return events, rows, rkeys


def fit(events, rows, rkeys, ref_date=None):
    pid = {}
    names = {}
    for r in rows:
        pid.setdefault(r[0], len(pid))
        names[r[0]] = r[8]
    P = np.array([pid[r[0]] for r in rows])
    R = np.array([r[2] for r in rows])
    E = np.array([r[1] for r in rows])
    y = np.array([r[3] for r in rows], float)
    days = np.array([datetime.fromisoformat(r[4]).toordinal() for r in rows])
    wind = np.array([r[5] for r in rows]); rain = np.array([r[6] for r in rows]); temp = np.array([r[7] for r in rows])
    nP, nR = len(pid), len(rkeys)

    # Step 1: separate "how hard was that round" from "how good is that player" (alternating averages)
    pe, re_ = np.zeros(nP), np.zeros(nR)
    cntP, cntR = np.bincount(P, minlength=nP), np.bincount(R, minlength=nR)
    for _ in range(60):
        re_ = np.bincount(R, y - pe[P], minlength=nR) / np.maximum(cntR, 1)
        pe = np.bincount(P, y - re_[R], minlength=nP) / (cntP + 3.0)
    shift = np.average(pe[P])            # zero = the average tour entrant
    pe -= shift; re_ += shift
    resid = y - re_[R]                   # strokes vs the average entrant that day (negative = better)

    # Step 2: current skill = recency-weighted average, pulled toward a newcomer level when history is thin
    ref = ref_date or days.max()
    wt = 0.5 ** ((ref - days) / HALF_LIFE_DAYS)
    newcomer = float(np.mean(resid[cntP[P] < 20])) if np.any(cntP[P] < 20) else 0.5
    sw = np.bincount(P, wt, minlength=nP)
    prior = PRIOR if PRIOR is not None else newcomer
    skill = (np.bincount(P, wt * resid, minlength=nP) + SHRINK_ROUNDS * prior) / (sw + SHRINK_ROUNDS)
    skill = skill * SPREAD

    # Step 3: what makes a round hard: the course, plus wind, rain and temperature
    r_event = np.zeros(nR, int); r_event[R] = E
    r_wind = np.zeros(nR); r_wind[R] = wind
    r_rain = np.zeros(nR); r_rain[R] = rain
    r_temp = np.zeros(nR); r_temp[R] = temp
    r_per = np.zeros(nR, int)
    for (ei, per), k in rkeys.items():
        r_per[k] = per
    cid = {}
    r_course = np.array([cid.setdefault(events[ei]["course"]["id"], len(cid)) for ei in r_event])
    nC = len(cid)
    X = np.zeros((nR, nC + 4))
    X[np.arange(nR), r_course] = 1
    X[:, nC] = r_wind - WIND_REF
    X[:, nC + 1] = np.minimum(r_rain, 15.0)
    X[:, nC + 2] = r_temp - TEMP_REF
    X[:, nC + 3] = (r_per >= 3)          # weekend rounds (tougher pins, pressure)
    ridge = np.zeros(nC + 4); ridge[:nC] = 0.05
    with np.errstate(all="ignore"):  # numpy on macOS prints spurious warnings for these products
        beta = np.linalg.solve(X.T @ (X * cntR[:, None]) + np.diag(ridge) * cntR.mean(), X.T @ (cntR * re_))
        r_pred = X @ beta
    course_base = beta[:nC]
    b_wind, b_rain, b_temp, b_wkend = beta[nC:]

    # Step 4: luck. How much do scores scatter around expectation, and does wind widen it?
    e2 = (resid - skill[P]) ** 2
    A = np.column_stack([np.ones(len(e2)), wind - WIND_REF])
    v0, v1 = np.linalg.lstsq(A, e2, rcond=None)[0]
    base_var = np.maximum(v0 + v1 * (wind - WIND_REF), 4.0)
    ratio = np.bincount(P, wt * e2 / base_var, minlength=nP)
    sd_mult = np.sqrt((ratio + 40.0) / (sw + 40.0))    # each player's own consistency, shrunk to typical

    # Step 5: does a player do better or worse than usual when it blows? (heavily shrunk)
    xw = wind - WIND_REF
    dev = resid - skill[P]
    sxx = np.bincount(P, wt * xw * xw, minlength=nP)
    sxy = np.bincount(P, wt * xw * dev, minlength=nP)
    raw = sxy / np.maximum(sxx, 1e-9)
    noise_var = v0 / np.maximum(sxx, 1e-9)
    big = sxx > np.percentile(sxx, 60)
    true_var = max(float(np.var(raw[big]) - np.mean(noise_var[big])), 0.0)
    wind_slope = raw * true_var / (true_var + noise_var)

    return dict(pid=pid, names=names, skill=skill, sd_mult=sd_mult, wind_slope=wind_slope, sw=sw, cntP=cntP,
                cid=cid, course_base=course_base, b_wind=b_wind, b_rain=b_rain, b_temp=b_temp, b_wkend=b_wkend,
                v0=v0, v1=v1, newcomer=newcomer, prior=prior * SPREAD, true_var=true_var, re_=re_, r_pred=r_pred, cntR=cntR,
                ref=ref, last_day=np.array([np.max(days[P == i]) if cntP[i] else 0 for i in range(nP)]))


def describe(m):
    print("players: %d, courses: %d" % (len(m["pid"]), len(m["cid"])))
    print("wind: %+.3f strokes per mph | rain: %+.3f per mm | temp: %+.3f per degF | weekend: %+.2f"
          % (m["b_wind"], m["b_rain"], m["b_temp"], m["b_wkend"]))
    print("round-to-round scatter (std dev): %.2f strokes at %d mph, %.2f at 20 mph"
          % (np.sqrt(m["v0"]), WIND_REF, np.sqrt(m["v0"] + m["v1"] * (20 - WIND_REF))))
    print("newcomer level: %+.2f | spread of true wind tendencies (sd, strokes per mph): %.4f"
          % (m["newcomer"], np.sqrt(m["true_var"])))
    err = m["re_"] - m["r_pred"]
    print("course+weather explains %.0f%% of round difficulty; typical miss %.2f strokes"
          % (100 * (1 - np.average(err ** 2, weights=m["cntR"]) / np.average((m["re_"] - np.average(m["re_"], weights=m["cntR"])) ** 2, weights=m["cntR"])),
             np.sqrt(np.average(err ** 2, weights=m["cntR"]))))
    inv = {v: k for k, v in m["pid"].items()}
    act = [i for i in range(len(inv)) if m["sw"][i] > 15]
    top = sorted(act, key=lambda i: m["skill"][i])[:12]
    print("best current players:", ", ".join("%s %.2f" % (m["names"][inv[i]], m["skill"][i]) for i in top))


def export(events, m):
    inv = {v: k for k, v in m["pid"].items()}
    ref = m["ref"]
    # courses on the current schedule = hosted an event in the last two seasons
    latest = {}
    for e in events:
        if e["season"] >= 2025 and e["course"]["id"] in m["cid"] and "Q-School" not in e["name"]:
            latest[e["course"]["id"]] = e
    need = set()
    for e in latest.values():
        need.update(p for p in e["field"] if p in m["pid"])
    selectable = {inv[i] for i in range(len(inv)) if m["cntP"][i] >= 12 and ref - m["last_day"][i] <= 500}
    keep = sorted(need | selectable, key=lambda p: m["skill"][m["pid"][p]])
    idx = {p: i for i, p in enumerate(keep)}
    players = [{"n": m["names"][p], "s": round(float(m["skill"][m["pid"][p]]), 3),
                "v": round(float(m["sd_mult"][m["pid"][p]]), 3), "w": round(float(m["wind_slope"][m["pid"][p]]), 4),
                "r": int(m["cntP"][m["pid"][p]]), "sel": 1 if p in selectable else 0} for p in keep]
    courses = []
    for cid, e in latest.items():
        a = e["course"]["address"]
        field = [idx[p] for p in e["field"] if p in idx]
        n_unknown = len(e["field"]) - len(field)
        courses.append({"n": e["course"]["name"], "ev": e["name"], "yr": e["season"],
                        "loc": ", ".join(x for x in (a.get("city"), a.get("state") or a.get("country")) if x),
                        "par": e["course"]["par"], "b": round(float(m["course_base"][m["cid"][cid]]), 3),
                        "cutRound": e.get("cutRound") or 0, "cutCount": cut_rule(e),
                        "major": e["major"], "f": field, "fx": n_unknown})
    courses.sort(key=lambda c: c["n"])
    out = {"asOf": datetime.fromordinal(int(ref)).date().isoformat(), "players": players, "courses": courses,
           "coef": {"wind": round(float(m["b_wind"]), 4), "rain": round(float(m["b_rain"]), 4),
                    "temp": round(float(m["b_temp"]), 4), "wkend": round(float(m["b_wkend"]), 3),
                    "v0": round(float(m["v0"]), 3), "v1": round(float(m["v1"]), 4),
                    "windRef": WIND_REF, "tempRef": TEMP_REF, "newcomer": round(m["newcomer"], 3)},
           "stats": {"events": len(events), "rounds": int(m["cntP"].sum())}}
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "model.js"), "w") as f:
        f.write("window.GOLF_MODEL = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print("wrote docs/model.js: %d players (%d selectable), %d courses" % (len(players), len(selectable), len(courses)))


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        import backtest
        backtest.run()
    else:
        ev, rows, rk = load()
        model = fit(ev, rows, rk)
        describe(model)
        export(ev, model)
