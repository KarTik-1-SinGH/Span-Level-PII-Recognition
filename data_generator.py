import json, random, os

FIRST = ["rakesh","neha","arjun","priya","amit","meera","rahul","sana","sanjay","vikas"]
LAST = ["sharma","singh","patel","kumar","gupta","nair","iyer","reddy","das","yadav"]
CITIES = ["mumbai","delhi","bangalore","pune","kolkata","chennai","hyderabad"]
LOCATIONS = ["andheri east mumbai","powai mumbai","baner pune","koramangala bangalore","salt lake kolkata"]
MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]
DIGITS = ["zero","one","two","three","four","five","six","seven","eight","nine"]
EMAIL_DOMAINS = ["gmail","yahoo","outlook","hotmail","rediffmail"]

def phone():
    return " ".join(random.choice(DIGITS) for _ in range(10))

def credit():
    return " ".join(random.choice(DIGITS) for _ in range(16))

def name():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"

def email():
    f, l = random.choice(FIRST), random.choice(LAST)
    return f"{f} {l} at {random.choice(EMAIL_DOMAINS)} dot com"

def date():
    return f"{random.randint(1,28)} {random.choice(MONTHS)} {random.randint(2015,2025)}"

def build(idx):
    label = random.choice(["CREDIT_CARD","PHONE","EMAIL","PERSON_NAME","DATE","CITY","LOCATION"])
    if label == "PHONE":
        span = phone()
        txt = f"my phone number is {span} please call me tomorrow"
    elif label == "CREDIT_CARD":
        span = credit()
        txt = f"my credit card number is {span} do not share it"
    elif label == "EMAIL":
        span = email()
        txt = f"you can contact me at email {span} for details"
    elif label == "PERSON_NAME":
        span = name()
        txt = f"my name is {span} and I work here"
    elif label == "DATE":
        span = date()
        txt = f"my appointment is on {span} next week"
    elif label == "CITY":
        span = random.choice(CITIES)
        txt = f"i am planning to move to {span} soon"
    else:
        span = random.choice(LOCATIONS)
        txt = f"the courier should be delivered to {span}"

    start = txt.index(span)
    end = start + len(span)
    return {"id": f"synth_{idx:04d}", "text": txt, "entities":[{"start":start,"end":end,"label":label}]}

def write(out, n):
    with open(out,"w",encoding="utf-8") as f:
        for i in range(n):
            json.dump(build(i), f)
            f.write("\n")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    write("data/train_synth.jsonl", 600)
    write("data/dev_synth.jsonl", 150)
    print("✔ Generated train_synth.jsonl (600) and dev_synth.jsonl (150)")
