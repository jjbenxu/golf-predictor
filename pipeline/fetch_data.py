"""Download PGA Tour round scores (ESPN public feed) and matching weather (Open-Meteo).

Output: data/events.json  (one record per tournament, with rounds and daily weather)
Run:    python3 pipeline/fetch_data.py
"""
import json, math, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "events.json")
# tour code on the feed -> seasons to download (pga = PGA Tour, ntw = Korn Ferry, eur = DP World Tour, liv = LIV Golf)
SOURCES = [("pga", range(2015, 2027)), ("ntw", range(2021, 2027)), ("eur", range(2021, 2027)), ("liv", range(2022, 2027))]
UA = {"User-Agent": "Mozilla/5.0 (golf-hobby-project)"}


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.load(urllib.request.urlopen(req, timeout=45))
        except Exception as ex:  # retry on any network hiccup
            if i == tries - 1:
                print("FAILED", url, ex, file=sys.stderr)
                return None
            time.sleep(2 + 3 * i)


def to_par(disp):
    if disp in (None, "", "-"):
        return None
    if disp == "E":
        return 0
    try:
        return int(disp)
    except ValueError:
        return None


def fetch_event(job):
    try:
        return _fetch_event(job)
    except Exception as ex:      # one odd tournament should never stop the whole download
        print("skipped", job, repr(ex), file=sys.stderr)
        return None


def _fetch_event(job):
    tour, ev_id = job
    d = get(f"https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard?league={tour}&event={ev_id}")
    if not d or not d.get("events"):
        return None
    e = d["events"][0]
    t = e.get("tournament", {})
    if (t.get("scoringSystem") or {}).get("name") != "Medal":
        return None  # skip team, match-play and Stableford events
    if e["status"]["type"]["name"] != "STATUS_FINAL":
        return None
    courses = e.get("courses") or []
    host = next((c for c in courses if c.get("host")), courses[0] if courses else None)
    if not host:
        return None
    n_courses = len(courses)
    rounds, field = [], []
    for p in (e.get("competitions") or [{}])[0].get("competitors") or []:
        a = p.get("athlete")
        if not a or "id" not in a or "displayName" not in a:
            continue
        field.append(a["id"])
        for l in p.get("linescores", []):
            per, val = l.get("period"), l.get("value")
            if per is None or val is None or l.get("isPlayoff") or per > 4:
                continue
            # multi-course events: early rounds rotate across courses, so keep host-course rounds only
            if n_courses > 1 and per <= n_courses:
                continue
            nines = (l.get("inScore"), l.get("outScore"))
            if (nines[0] is None) != (nines[1] is None) or not (56 <= val <= 95):
                continue  # incomplete round (withdrawal)
            if nines[0] is None and val < 60:
                continue  # tours without nine-hole splits: treat very low totals as unfinished rounds
            tp = to_par(l.get("displayValue"))
            if tp is None:
                continue
            rounds.append([a["id"], a["displayName"], per, int(val), tp, l.get("teeTime")])
    if len(rounds) < 60:
        return None
    rounds_tour = tour
    return {
        "id": e["id"], "tour": rounds_tour, "name": e["name"], "date": e["date"], "end": e.get("endDate"),
        "season": e["season"]["year"], "major": bool(t.get("major")),
        "cutRound": t.get("cutRound"), "cutCount": t.get("cutCount"),
        "course": {"id": host["id"], "name": host["name"], "par": host.get("shotsToPar"),
                   "yards": host.get("totalYards"), "address": host.get("address") or {}},
        "rounds": rounds, "field": field, "nCourses": n_courses,
    }


_geo = {}


def geocode(addr):
    key = (addr.get("city"), addr.get("state"), addr.get("country"))
    if key in _geo:
        return _geo[key]
    res = None
    if addr.get("city"):
        d = get("https://geocoding-api.open-meteo.com/v1/search?count=20&name=" + urllib.parse.quote(addr["city"]))
        cands = (d or {}).get("results") or []
        st = (addr.get("state") or "").lower()
        ctry = (addr.get("country") or "").lower()
        def score(c):
            s = 0
            a1 = (c.get("admin1") or "").lower()
            if st and (a1 == st or STATE.get(st.upper(), "").lower() == a1):
                s += 2
            cn, cc = (c.get("country") or "").lower(), (c.get("country_code") or "").lower()
            if ctry and (ctry in cn or cn in ctry or (ctry == "usa" and cc == "us")):
                s += 1
            return s
        if cands:
            best = max(cands, key=score)
            res = (best["latitude"], best["longitude"])
    _geo[key] = res
    return res


STATE = {"AL": "Alabama", "AZ": "Arizona", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "CT": "Connecticut",
         "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
         "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
         "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
         "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
         "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
         "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
         "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
         "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
         "WY": "Wyoming"}


def add_weather(ev):
    loc = geocode(ev["course"]["address"])
    ev["latlon"] = loc
    ev["weather"] = {}
    if not loc:
        return ev
    tees = [r[5] for r in ev["rounds"] if r[5]]
    start = (min(tees) if tees else ev["date"])[:10]
    end = (max(tees) if tees else (ev["end"] or ev["date"]))[:10]
    s = (datetime.fromisoformat(start) - timedelta(days=1)).date().isoformat()
    en = (datetime.fromisoformat(end) + timedelta(days=1)).date().isoformat()
    d = get("https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=%s&end_date=%s"
            "&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,wind_direction_10m"
            "&wind_speed_unit=mph&temperature_unit=fahrenheit&timezone=auto" % (loc[0], loc[1], s, en))
    if not d or "hourly" not in d:
        return ev
    h = d["hourly"]
    ev["utc_offset"] = d.get("utc_offset_seconds", 0)
    days = {}
    for i, ts in enumerate(h["time"]):
        day, hour = ts[:10], int(ts[11:13])
        if 7 <= hour <= 18 and h["wind_speed_10m"][i] is not None:
            days.setdefault(day, []).append((h["wind_speed_10m"][i], h["wind_gusts_10m"][i] or 0,
                                             h["precipitation"][i] or 0, h["temperature_2m"][i],
                                             h["wind_direction_10m"][i] or 0))
    for day, v in days.items():
        n = len(v)
        # average direction the wind blows FROM, weighted by speed (compass degrees, 0 = north)
        vx = sum(x[0] * math.sin(math.radians(x[4])) for x in v)
        vy = sum(x[0] * math.cos(math.radians(x[4])) for x in v)
        wdir = round(math.degrees(math.atan2(vx, vy)) % 360)
        steady = round(math.hypot(vx, vy) / max(sum(x[0] for x in v), 1e-9), 2)
        ev["weather"][day] = {"dir": wdir, "steady": steady, "wind": round(sum(x[0] for x in v) / n, 1), "gust": round(sum(x[1] for x in v) / n, 1),
                              "rain": round(sum(x[2] for x in v), 1), "temp": round(sum(x[3] for x in v) / n, 1)}
    return ev


def main():
    jobs, seen = [], set()
    for tour, seasons in SOURCES:   # PGA Tour first, so shared events (majors) are filed under it
        for y in seasons:
            d = get(f"https://site.api.espn.com/apis/site/v2/sports/golf/{tour}/scoreboard?dates={y}")
            for c in ((d or {}).get("leagues") or [{}])[0].get("calendar") or []:
                if c["id"] not in seen:
                    seen.add(c["id"])
                    jobs.append((tour, c["id"]))
    print("tournaments listed:", len(jobs))
    with ThreadPoolExecutor(8) as ex:
        fetched = [e for e in ex.map(fetch_event, jobs) if e]
    events, keys = [], set()
    for e in fetched:               # the same tournament can be listed by two tours under different ids
        k = (e["course"]["id"], e["date"][:10])
        if k not in keys:
            keys.add(k)
            events.append(e)
    print("usable stroke-play tournaments:", len(events))
    for e in events:  # geocode one at a time (cached), then weather in parallel
        geocode(e["course"]["address"])
    with ThreadPoolExecutor(4) as ex:
        events = list(ex.map(add_weather, events))
    events.sort(key=lambda e: e["date"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(events, open(OUT, "w"))
    nr = sum(len(e["rounds"]) for e in events)
    nw = sum(1 for e in events if e["weather"])
    print(f"saved {len(events)} tournaments, {nr} rounds, weather for {nw}; {OUT}")
    for t in sorted({e["tour"] for e in events}):
        print("  %s: %d tournaments, %d rounds" % (t, sum(e["tour"] == t for e in events), sum(len(e["rounds"]) for e in events if e["tour"] == t)))
    print("  without weather (location unknown):", sum(1 for e in events if not e["weather"]))


if __name__ == "__main__":
    main()
