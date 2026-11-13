"""Download PGA Tour round scores (ESPN public feed) and matching weather (Open-Meteo).

Output: data/events.json  (one record per tournament, with rounds and daily weather)
Run:    python3 pipeline/fetch_data.py
"""
import json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "events.json")
SEASONS = range(2021, 2027)
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


def fetch_event(ev_id):
    d = get(f"https://site.web.api.espn.com/apis/site/v2/sports/golf/leaderboard?league=pga&event={ev_id}")
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
        if not a:
            continue
        field.append(a["id"])
        for l in p.get("linescores", []):
            per, val = l.get("period"), l.get("value")
            if per is None or val is None or l.get("isPlayoff") or per > 4:
                continue
            # multi-course events: early rounds rotate across courses, so keep host-course rounds only
            if n_courses > 1 and per <= n_courses:
                continue
            if l.get("inScore") is None or l.get("outScore") is None or not (56 <= val <= 95):
                continue  # incomplete round (withdrawal)
            tp = to_par(l.get("displayValue"))
            if tp is None:
                continue
            rounds.append([a["id"], a["displayName"], per, int(val), tp, l.get("teeTime")])
    if len(rounds) < 60:
        return None
    return {
        "id": e["id"], "name": e["name"], "date": e["date"], "end": e.get("endDate"),
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
            "&hourly=temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m"
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
                                             h["precipitation"][i] or 0, h["temperature_2m"][i]))
    for day, v in days.items():
        n = len(v)
        ev["weather"][day] = {"wind": round(sum(x[0] for x in v) / n, 1), "gust": round(sum(x[1] for x in v) / n, 1),
                              "rain": round(sum(x[2] for x in v), 1), "temp": round(sum(x[3] for x in v) / n, 1)}
    return ev


def main():
    ids = []
    for y in SEASONS:
        d = get(f"https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard?dates={y}")
        for c in d["leagues"][0]["calendar"]:
            if c["id"] not in ids:
                ids.append(c["id"])
    print("tournaments listed:", len(ids))
    with ThreadPoolExecutor(6) as ex:
        events = [e for e in ex.map(fetch_event, ids) if e]
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
    for e in events:
        if not e["weather"]:
            print("  no weather:", e["name"], e["course"]["address"])


if __name__ == "__main__":
    main()
