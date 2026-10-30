#!/usr/bin/env python3
"""Build the document-length tier corpora for bench/run_bench.py
--lengths: documents of ~500 / ~2,000 / ~10,000 / ~50,000 chars,
composed from the parity fixture notes plus a seeded inventory of
PHI-bearing and filler clinical/case-note sentences.

Fixture-derived by construction (a timing corpus, not an evaluation
set): each document is a seeded sequence of sentences with seeded
slot fills (names, DOBs, SSNs, phones, MRNs, addresses, EMP/REF
codes, card numbers), joined into paragraphs and fit to within
+/-10% of the tier target. Pure ASCII, like the canonical corpora,
so the cross-language offset semantics stay comparable. The same
generator run always produces byte-identical files (fixed seeds);
the driver re-generates each tier and checks the SHA-256 against
the file on disk, exactly as it does for bench/corpus.jsonl.

    python3 bench/gen_length_tiers.py 2000      # print one tier
    python3 bench/gen_length_tiers.py --write   # write bench/lengths/
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = os.path.join(HERE, "..", "e2e", "parity", "notes.jsonl")
OUT_DIR = os.path.join(HERE, "lengths")

SEED = 20261029
TIERS = {500: 100, 2000: 100, 10000: 50, 50000: 20}

FIRST = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer",
         "Michael", "Maria", "David", "Sarah", "Carlos", "Priya",
         "Ahmed", "Emily", "Daniel", "Rosa", "Kevin", "Laura",
         "Brian", "Nina", "Thomas", "Grace", "Henry", "Alice",
         "Frank", "Julia", "Omar", "Peter", "Susan", "Victor"]
LAST = ["Smith", "Johnson", "Garcia", "Santos", "Brown", "Miller",
        "Davis", "Wilson", "Moore", "Taylor", "Lee", "Patel",
        "Khan", "Nguyen", "Lopez", "Harris", "Clark", "Lewis",
        "Walker", "Hall", "Allen", "Young", "King", "Wright"]
PLACES = [("123 Main Street", "Harrisburg", "PA", "17101"),
          ("48 Elm Avenue", "Springfield", "IL", "62704"),
          ("9 Oak Court", "Austin", "TX", "78701"),
          ("2200 River Road", "Denver", "CO", "80202"),
          ("77 Harbor Blvd", "San Diego", "CA", "92101"),
          ("315 Pine Street", "Seattle", "WA", "98101"),
          ("602 Lake Drive", "Chicago", "IL", "60614"),
          ("18 Maple Lane", "Boston", "MA", "02118")]
AREAS = ["415", "212", "312", "617", "206", "512", "718", "303"]
PROVIDERS = ["Dr. Chen", "Dr. Alvarez", "Dr. Okafor", "Dr. Novak",
             "Dr. Fitzgerald", "Dr. Raman", "Dr. Doyle", "Dr. Kim"]
CARDS = ["4111111111111111", "5500005555555559", "378282246310005"]
MEDS = ["metformin 500 mg twice daily", "lisinopril 10 mg daily",
        "atorvastatin 20 mg at bedtime", "albuterol inhaler as needed",
        "omeprazole 20 mg daily", "gabapentin 300 mg three times daily",
        "hydrochlorothiazide 25 mg daily", "sertraline 50 mg daily"]

# Sentences carrying PHI-shaped tokens (names, dates, identifiers).
PHI_TEMPLATES = [
    "Patient {name} (DOB {dob}) presented for follow-up of hypertension and type 2 diabetes.",
    "Spoke with {name} by phone at {phone} regarding the pending referral.",
    "Intake completed on {date}. MRN {mrn}. ZIP {zip}. SSN {ssn} verified against photo ID.",
    "Please contact {name} at {email} or {phone} to reschedule the appointment.",
    "{name} reports that the pharmacy at {addr}, {city}, {st} {zip} never received the refill request.",
    "Case note: {name}, DOB {dob}, called the clinic line {phone} twice this morning.",
    "Insurance card on file ending in {card}; member ID {mrn} confirmed at check-in.",
    "Badge {emp} documented the home visit for case {mrn} on {date}.",
    "Ref code {ref} and ref code {ref2} are attached to this encounter in the chart.",
    "Forward the discharge summary to {prov} at {email} once the labs result.",
    "{name} (SSN {ssn}) authorized release of records to the county benefits office.",
    "Voicemail left for {name} at {phone}; will retry on {date}.",
    "The patient, {name}, DOB {dob}, was accompanied by his daughter {name2}.",
    "Billing flagged account {mrn}: card {card} declined, patient updated payment by phone.",
    "Address on file: {addr}, Apt. {apt}, {city}, {st} {zip}. Phone {phone}.",
    "{prov} reviewed the note and co-signed; next visit booked for {date}.",
    "Contact the caregiver, {name2}, at {phone} before the {date} procedure.",
    "Worker {emp} left a door hanger at {addr} on {date} after no answer.",
    "Remind {name} that the lab slip (MRN {mrn}) must be brought to the draw station.",
    "Email from {name} ({email}) asks whether the {date} slot can move to the afternoon.",
    "{name} confirmed identity with DOB {dob} and the last four of the SSN on file.",
    "Transportation voucher mailed to {addr}, {city}, {st} {zip} on {date}.",
    "Per {prov}, increase monitoring and recheck in two weeks; notify {name} by portal message.",
    "The consent form signed by {name} on {date} was scanned into the chart under MRN {mrn}.",
    "Reach the patient at ({area}) 555-01{line} or +1 {area}-555-02{line} after 5 pm.",
    "Guest {name2} (member {mrn}) requested a callback at {phone}.",
    "Passport and DL for {name} were copied at intake; copies filed under case {mrn}.",
    "Schedule the interpreter for {name} ({phone}) ahead of the {date} hearing.",
    "{name} at {addr} reported a change in household size effective {date}.",
    "Send the summary to {name}, {addr}, {city}, {st} {zip}, and copy {prov}.",
    "Phone screen on {date}: {name} verified DOB {dob} and ZIP {zip}.",
    "Note dictated by {prov} for patient {name}, MRN {mrn}, encounter {date}.",
    "Remittance for invoice {mrn} charged to card {card}; receipt emailed to {email}.",
    "Clerk {emp} updated the address for {name} to {addr}, {city}, {st} {zip}.",
    "{name} prefers portal messages; alternate contact {name2} at {phone}.",
    "The {date} letter to {name} at {addr} was returned undeliverable.",
    "Case {mrn}: {name} (DOB {dob}) missed the {date} appointment; outreach logged by {emp}.",
    "Forward labs to {email} and confirm receipt with {name} at {phone}.",
]

# Clinical/case-note filler without identifiers (still exercises the
# scanners: numbers, units, dates of service are filled separately).
FILLER_TEMPLATES = [
    "Blood pressure 128/82, heart rate 74, temperature 98.4 F, oxygen saturation 97 percent on room air.",
    "Lungs clear to auscultation bilaterally; no wheeze, rales, or rhonchi appreciated.",
    "The patient denies chest pain, shortness of breath, fever, or chills at this time.",
    "Continue {med}; counseled on adherence and reviewed the side-effect profile.",
    "Labs reviewed: A1c improved from 8.1 to 7.4; LDL remains above goal at 142.",
    "Plan: repeat basic metabolic panel in four weeks and reassess at the next visit.",
    "Wound edges are approximated, without erythema, drainage, or dehiscence.",
    "Patient ambulated the hallway twice without difficulty and tolerated it well.",
    "Dietary counseling provided; handout on sodium restriction given and teach-back completed.",
    "Immunizations are up to date, including influenza this season and Tdap in 2023.",
    "Mental status: alert and oriented times three, affect appropriate, speech fluent.",
    "Abdomen soft, non-tender, non-distended; bowel sounds present in all quadrants.",
    "Encouraged daily walking as tolerated and a follow-up call in one week.",
    "No new medications were started today; the current list was reconciled with the patient.",
    "Edema is trace at both ankles, improved from the last examination.",
    "The patient reports sleeping six to seven hours nightly with one awakening.",
    "Range of motion in the right shoulder is limited to about 110 degrees of abduction.",
    "Weight is stable within two pounds of the prior visit; appetite is fair.",
    "Will obtain a chest x-ray if the cough persists beyond ten days.",
    "Discussed advance directive paperwork; the patient will discuss with family first.",
    "Grip strength is symmetric; sensation intact to light touch in both hands.",
    "The rash has faded considerably since the steroid cream was started last week.",
    "Instructions were given in writing, and the patient verbalized understanding.",
    "Follow-up in three months, sooner if symptoms worsen or new concerns arise.",
    "Home safety evaluation completed; grab bars recommended for the shower.",
    "The patient completed six minutes on the treadmill without chest discomfort.",
    "Hearing is grossly intact to conversational voice in a quiet room.",
    "No falls reported since the last visit; the home has adequate lighting.",
    "Reviewed red-flag symptoms that should prompt an urgent call or visit.",
    "The splint was removed, the skin inspected, and a new splint applied.",
]


def slots(rng):
    first = rng.choice(FIRST)
    first2 = rng.choice(FIRST)
    last = rng.choice(LAST)
    last2 = rng.choice(LAST)
    street, city, st, zipc = rng.choice(PLACES)
    area = rng.choice(AREAS)
    return {
        "name": f"{first} {last}",
        "name2": f"{first2} {last2}",
        "dob": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/"
               f"{rng.randint(1948, 2015)}",
        "date": f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/"
                f"{rng.randint(2023, 2026)}",
        "ssn": f"{rng.choice(['123', '234', '321', '078', '189'])}-"
               f"{rng.randint(1, 99):02d}-{rng.randint(1, 9999):04d}",
        "phone": f"{area}-555-01{rng.randint(10, 99)}",
        "area": area,
        "line": f"{rng.randint(10, 99)}",
        "email": f"{first.lower()}.{last.lower()}@example."
                 f"{rng.choice(['com', 'org', 'net'])}",
        "mrn": f"{rng.randint(10000000, 99999999)}",
        "zip": zipc,
        "addr": street,
        "city": city,
        "st": st,
        "apt": f"{rng.randint(1, 40)}",
        "emp": f"EMP{rng.randint(100000, 999999)}",
        "ref": f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
               f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-"
               f"{rng.randint(1000, 9999)}",
        "ref2": f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
                f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-"
                f"{rng.randint(1000, 9999)}",
        "card": rng.choice(CARDS),
        "prov": rng.choice(PROVIDERS),
        "med": rng.choice(MEDS),
    }


def fixture_sentences():
    out = []
    with open(NOTES) as f:
        for line in f:
            line = line.strip()
            if line:
                note = json.loads(line)["note"]
                if note:
                    out.append(note)
    return out


def build_doc(rng, target, inventory):
    """Assemble one document within +/-10% of target chars. Length
    accounting is exact: sentences are joined by single-character
    separators (' ' or a paragraph-break newline)."""
    sentences = []
    total = 0

    def add(s):
        nonlocal total
        total += len(s) + (1 if sentences else 0)
        sentences.append(s)

    def drop_last():
        nonlocal total
        s = sentences.pop()
        total -= len(s) + (1 if sentences else 0)

    # Accumulate to just under the target (one short sentence of
    # headroom), then fit the tail against the target itself.
    while total < target - 130:
        add(rng.choice(inventory).format(**slots(rng)))
    # Fit the tail: try candidate final sentences, keep the one that
    # lands closest to the target.
    best = None
    for _ in range(60):
        cand = rng.choice(inventory).format(**slots(rng))
        new_total = total + len(cand) + 1
        if best is None or abs(new_total - target) < abs(best[0] - target):
            best = (new_total, cand)
        if target * 0.97 <= new_total <= target * 1.03:
            break
    add(best[1])
    while total > target * 1.10 and len(sentences) > 2:
        drop_last()
    assert target * 0.90 <= total <= target * 1.10, (target, total)
    # Paragraph breaks: replace some separators with newlines (same
    # 1-char accounting).
    parts = []
    for i, s in enumerate(sentences):
        if i:
            parts.append("\n" if rng.random() < 0.16 else " ")
        parts.append(s)
    doc = "".join(parts)
    assert len(doc) == total
    return doc


def build_tier(tier):
    rng = random.Random(SEED + tier)
    inventory = fixture_sentences() + PHI_TEMPLATES + FILLER_TEMPLATES
    return [build_doc(rng, tier, inventory)
            for _ in range(TIERS[tier])]


def char_stats(docs):
    lens = sorted(len(d) for d in docs)
    n = len(lens)
    return {"count": n, "min": lens[0], "median": lens[n // 2],
            "mean": round(sum(lens) / n), "max": lens[-1]}


def main() -> int:
    args = sys.argv[1:]
    if args == ["--write"]:
        os.makedirs(OUT_DIR, exist_ok=True)
        for tier in TIERS:
            docs = build_tier(tier)
            path = os.path.join(OUT_DIR, f"tier_{tier}.jsonl")
            with open(path, "w") as f:
                for d in docs:
                    f.write(json.dumps({"note": d}) + "\n")
            print(f"{path}: {char_stats(docs)}", file=sys.stderr)
        return 0
    if len(args) == 1 and int(args[0]) in TIERS:
        for d in build_tier(int(args[0])):
            print(json.dumps({"note": d}))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
