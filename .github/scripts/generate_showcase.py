"""
Builds showcase.gif from the studio README.

1. Finds the README and extracts every game/project (AI first, regex fallback).
2. Loads each project's preview image (local file first, then raw GitHub URL).
3. Renders a splash slide + one card per project and saves an animated GIF.
4. Names the GIF by content hash (showcase-<hash>.gif) so GitHub's image cache
   always fetches the new version, deletes older versions, and updates the
   image link in the README.
"""
import hashlib
import json
import os
import re
import shutil
import tempfile
from io import BytesIO

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
STUDIO_NAME = "GLZZLEXI STUDIOS"
HUB_URL = "glzzlexi.me"
ACCENT = (255, 117, 140)
MAX_PROJECTS = 12

REPO = os.environ.get("GITHUB_REPOSITORY", "")
BRANCH = os.environ.get("BRANCH", "main")
TOKEN = os.environ.get("AI_TOKEN", "")

WIDTH, HEIGHT = 800, 450
BG = (18, 22, 30)
SKIP_SECTIONS = ["roadmap", "open source", "about", "join the team"]

# --------------------------------------------------------------------------
# 1. Locate README
# --------------------------------------------------------------------------
candidates = ["profile/README.md", ".github/profile/README.md", "README.md"]
readme_path = next((p for p in candidates if os.path.exists(p)), None)
if not readme_path:
    print("::error::Could not find README.md in profile/, .github/profile/ or root.")
    raise SystemExit(1)

readme_dir = os.path.dirname(readme_path)
out_dir = os.path.join(readme_dir, "showcase")
cache_path = os.path.join(out_dir, "projects-cache.json")
with open(readme_path, "r", encoding="utf-8") as f:
    readme_text = f.read()
print(f"Using README: {readme_path}")

# Matches any reference to our GIF: showcase.gif, showcase-1a2b3c4d.gif,
# with or without a folder / URL in front and a ?query behind.
GIF_REF = re.compile(
    r"(?<![\w.:/\-@%])(?:[\w.:/\-@%]*/)?showcase(?:-[0-9a-f]{8})?\.gif(?:\?[\w=&.\-]*)?"
)
GIF_NAME = re.compile(r"^showcase-[0-9a-f]{8}\.gif$")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clean(s):
    """Strip markdown/html/emoji so the text renders cleanly with DejaVu."""
    s = s or ""
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)      # [text](url) -> text
    s = re.sub(r"<[^>]+>", "", s)                         # html tags
    s = re.sub(r"[`*]", "", s)                            # bold / code marks
    s = re.sub(r"[\U00010000-\U0010FFFF\u200d\ufe0f]", "", s)  # emoji
    return re.sub(r"\s+", " ", s).strip()


def extract_image(section):
    """First real preview image in a section (skips badges and SVGs)."""
    pattern = r'<img[^>]+src=["\']([^"\']+)["\']|!\[[^\]]*\]\(([^)\s]+)'
    for m in re.finditer(pattern, section):
        url = m.group(1) or m.group(2)
        low = url.lower().split("?")[0]
        if GIF_REF.search(low):
            continue
        if "shields.io" in url.lower() or "badge" in low or low.endswith(".svg"):
            continue
        return url
    return None


def extract_tagline(section):
    for line in section.splitlines()[1:]:
        if line.strip().startswith(("|", "#")):
            continue
        t = clean(line)
        if len(t) < 25:
            continue
        if re.match(r"(status|genre|role|tech|stack)\s*:", t, re.I):
            continue
        return t
    return "Built with modular Luau systems on the Roblox engine."


# --------------------------------------------------------------------------
# 2. Project extraction
# --------------------------------------------------------------------------
def regex_parser(text):
    print("Parsing README with local Markdown parser...")
    projects = []
    # Split on every heading level so one section can't swallow the next
    for sec in re.split(r"\n(?=#{1,6}\s+)", text):
        lines = sec.strip().splitlines()
        if not lines or "status:" not in sec.lower():
            continue
        m = re.match(r"#{1,6}\s+(.+)", lines[0])
        if not m:
            continue
        if any(s in lines[0].lower() for s in SKIP_SECTIONS):
            continue
        title = re.sub(r"\(.*?\)", "", clean(m.group(1))).strip()
        if not title:
            continue

        status = "In Development"
        sm = re.search(r"\*\*Status:\*\*\s*`?([^`\n]+)`?", sec)
        if sm:
            status = clean(sm.group(1))

        genre = "Roblox Experience"
        gm = re.search(r"\*\*(?:Genre|Role):\*\*\s*`?([^`\n]+)`?", sec)
        if gm:
            genre = clean(gm.group(1))

        projects.append({
            "title": title,
            "status": status,
            "genre": genre,
            "image_url": extract_image(sec),
            "tagline": extract_tagline(sec),
        })
    return projects


def ai_parser(text, token):
    """Uses GitHub Models (free) -> openai/gpt-4o-mini."""
    if not token:
        print("No token available, skipping AI step.")
        return None
    print("Querying GitHub Models (openai/gpt-4o-mini)...")
    system_prompt = (
        "You analyze a game studio README. Extract every distinct game or project "
        "(live, upcoming, prototype). Exclude general sections like About, Roadmap, "
        "Open Source and Join the Team. Return ONLY a JSON object with one key "
        "'projects': an array of objects with 'title' (clean name, no markdown), "
        "'status' (e.g. 'Live', 'In Development', 'Delayed', 'Top Priority'), "
        "'genre' (short, e.g. 'Strategy / Simulation'), "
        "'image_url' (the first preview image URL EXACTLY as written in that "
        "project's section, or null), and 'tagline' (one punchy sentence, max 110 characters)."
    )
    try:
        res = requests.post(
            "https://models.github.ai/inference/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text[:12000]},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            },
            timeout=30,
        )
        if res.status_code != 200:
            print(f"AI API returned HTTP {res.status_code}: {res.text[:300]}")
            return None
        content = res.json()["choices"][0]["message"]["content"]
        items = json.loads(content).get("projects", [])
        items = [p for p in items if isinstance(p, dict) and p.get("title")]
        print(f"AI extracted {len(items)} projects.")
        return items or None
    except Exception as e:
        print(f"AI request failed: {e}")
        return None


# Fingerprint ignores our own GIF link, so updating the link never invalidates the cache.
fingerprint = hashlib.sha256(GIF_REF.sub("SHOWCASE", readme_text).encode()).hexdigest()


def read_cache():
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("fingerprint") == fingerprint and data.get("projects"):
            return data["projects"]
    except Exception:
        pass
    return None


regex_projects = regex_parser(readme_text)
projects = read_cache()
from_cache = bool(projects)
if from_cache:
    print("README unchanged since last run: reusing cached AI results.")
else:
    projects = ai_parser(readme_text, TOKEN)
ai_succeeded = bool(projects) and not from_cache

if from_cache:
    pass  # cached projects were already cleaned and merged
elif projects:
    # Never trust AI image URLs: keep only ones that really exist in the README,
    # otherwise borrow the image the regex parser found for the same project.
    for p in projects:
        p["title"] = clean(p.get("title"))
        p["status"] = clean(p.get("status")) or "In Development"
        p["genre"] = clean(p.get("genre")) or "Roblox Experience"
        p["tagline"] = clean(p.get("tagline")) or "Built with modular Luau systems."
        img = p.get("image_url")
        if not img or img not in readme_text:
            t = p["title"].lower()
            match = next(
                (r for r in regex_projects
                 if r["title"].lower() in t or t in r["title"].lower()),
                None,
            )
            p["image_url"] = match["image_url"] if match else None
else:
    projects = regex_projects

if not projects:
    print("::error::Could not extract any projects from the README.")
    raise SystemExit(1)

if ai_succeeded:
    os.makedirs(out_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "projects": projects}, f, indent=2)

projects = projects[:MAX_PROJECTS]
print("Projects:", ", ".join(p["title"] for p in projects))


# --------------------------------------------------------------------------
# 3. Image loading
# --------------------------------------------------------------------------
def load_image(src):
    if not src:
        return None
    src = src.strip()
    try:
        if not re.match(r"^(https?:)?//", src):
            rel = src.split("?")[0].split("#")[0]
            if rel.startswith("/"):
                local = os.path.normpath(rel.lstrip("/"))
            else:
                local = os.path.normpath(os.path.join(readme_dir, rel))
            if os.path.exists(local):
                im = Image.open(local)
            else:
                url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{local}"
                res = requests.get(url, timeout=15)
                res.raise_for_status()
                im = Image.open(BytesIO(res.content))
        else:
            if src.startswith("//"):
                src = "https:" + src
            blob = re.match(r"https://github\.com/([^/]+/[^/]+)/blob/(.+)", src)
            if blob:
                src = f"https://raw.githubusercontent.com/{blob.group(1)}/{blob.group(2)}"
            res = requests.get(src, timeout=15, headers={"User-Agent": "showcase-bot"})
            res.raise_for_status()
            im = Image.open(BytesIO(res.content))

        # Flatten transparency onto a dark background
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (10, 12, 16, 255))
        bg.alpha_composite(im)
        return bg.convert("RGB")
    except Exception as e:
        print(f"  Could not load image '{src}': {e}")
        return None


# --------------------------------------------------------------------------
# 4. Drawing
# --------------------------------------------------------------------------
def get_font(size, bold=False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = get_font(26, bold=True)
F_TITLE_SM = get_font(21, bold=True)
F_BADGE = get_font(13, bold=True)
F_BODY = get_font(14)
F_TAG = get_font(13)
F_SMALL = get_font(11)


def status_color(status):
    s = status.lower()
    if any(k in s for k in ["live", "released"]):
        return (46, 204, 113)
    if any(k in s for k in ["priority", "active"]):
        return (52, 152, 219)
    if any(k in s for k in ["delayed", "hold"]):
        return (231, 76, 60)
    return (241, 196, 15)


def fit_text(draw, text, font, max_w):
    if draw.textlength(text, font=font) <= max_w:
        return text
    while text and draw.textlength(text + "...", font=font) > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


def wrap_text(draw, text, font, max_w, max_lines):
    lines, cur = [], ""
    for word in text.split():
        test = f"{cur} {word}".strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = fit_text(draw, lines[-1].rstrip(".,") + "...", font, max_w)
    return lines


def draw_dots(draw, index, total, color):
    gap, r = 18, 4
    start = WIDTH // 2 - (total - 1) * gap // 2
    for i in range(total):
        x = start + i * gap
        fill = color if i == index else (45, 52, 66)
        draw.ellipse([(x - r, 418 - r), (x + r, 418 + r)], fill=fill)


total_frames = len(projects) + 1
frames = []

# ---- Splash slide ----
splash = Image.new("RGB", (WIDTH, HEIGHT), (15, 18, 24))
d = ImageDraw.Draw(splash)
d.rectangle([(0, 0), (WIDTH, 5)], fill=ACCENT)
d.text((WIDTH // 2, 155), STUDIO_NAME, fill=(255, 255, 255), font=get_font(34, True), anchor="mm")
d.text((WIDTH // 2, 205), "Official Project Pipeline & Active Releases",
       fill=(160, 170, 190), font=get_font(16), anchor="mm")
d.text((WIDTH // 2, 255), "Modular Luau Architecture  |  Roblox Engine",
       fill=(100, 110, 130), font=get_font(13), anchor="mm")
d.rounded_rectangle([(WIDTH // 2 - 95, 305), (WIDTH // 2 + 95, 342)], radius=6, fill=ACCENT)
d.text((WIDTH // 2, 323), HUB_URL, fill=(255, 255, 255), font=get_font(14, True), anchor="mm")
draw_dots(d, 0, total_frames, ACCENT)
frames.append(splash)

# ---- Project cards ----
IMG_X, IMG_Y, IMG_W, IMG_H = 40, 75, 380, 280
RIGHT_X = 450
RIGHT_W = WIDTH - 40 - RIGHT_X

for i, proj in enumerate(projects, start=1):
    print(f"Rendering card {i}/{len(projects)}: {proj['title']}")
    color = status_color(proj["status"])
    card = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(card)
    draw.rectangle([(0, 0), (WIDTH, 5)], fill=color)

    # Image frame
    draw.rounded_rectangle(
        [(IMG_X, IMG_Y), (IMG_X + IMG_W, IMG_Y + IMG_H)],
        radius=8, fill=(10, 12, 16), outline=(35, 42, 54), width=1,
    )
    shot = load_image(proj.get("image_url"))
    if shot:
        inner = (IMG_W - 2, IMG_H - 2)
        # "cover" crop so the picture fills the whole frame
        shot = ImageOps.fit(shot, inner, method=Image.LANCZOS, centering=(0.5, 0.4))
        mask = Image.new("L", inner, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [(0, 0), (inner[0] - 1, inner[1] - 1)], radius=7, fill=255
        )
        card.paste(shot, (IMG_X + 1, IMG_Y + 1), mask)
    else:
        draw.text((IMG_X + IMG_W // 2, IMG_Y + IMG_H // 2), "Preview Pending",
                  fill=(70, 80, 95), font=F_BODY, anchor="mm")

    # Right column
    draw.text((RIGHT_X, 75), f"{STUDIO_NAME} | SHOWCASE", fill=(100, 115, 135), font=F_SMALL)
    # One big line if it fits, otherwise a smaller font over up to two lines
    if draw.textlength(proj["title"], font=F_TITLE) <= RIGHT_W:
        title_lines, title_font, shift = [proj["title"]], F_TITLE, 0
    else:
        title_font = F_TITLE_SM
        title_lines = wrap_text(draw, proj["title"], title_font, RIGHT_W, 2)
        shift = 26 if len(title_lines) > 1 else 0
    for n, line in enumerate(title_lines):
        draw.text((RIGHT_X, 102 + n * 26), line, fill=(255, 255, 255), font=title_font)

    badge_txt = proj["status"].split("(")[0].strip().upper()
    badge_txt = fit_text(draw, badge_txt, F_BADGE, RIGHT_W - 20)
    badge_w = int(draw.textlength(badge_txt, font=F_BADGE)) + 20
    draw.rounded_rectangle([(RIGHT_X, 146 + shift), (RIGHT_X + badge_w, 172 + shift)], radius=4, fill=color)
    draw.text((RIGHT_X + 10, 152 + shift), badge_txt, fill=(15, 18, 24), font=F_BADGE)

    draw.text((RIGHT_X, 190 + shift),
              fit_text(draw, f"Genre: {proj['genre']}", F_BODY, RIGHT_W),
              fill=(180, 190, 205), font=F_BODY)

    for n, line in enumerate(wrap_text(draw, proj["tagline"], F_TAG, RIGHT_W, 4 if shift == 0 else 3)):
        draw.text((RIGHT_X, 228 + shift + n * 20), line, fill=(135, 146, 162), font=F_TAG)

    draw.line([(RIGHT_X, 325), (WIDTH - 40, 325)], fill=(30, 36, 48), width=1)
    draw.text((RIGHT_X, 338), f"Official hub: {HUB_URL}", fill=ACCENT, font=F_SMALL)

    draw_dots(draw, i, total_frames, color)
    frames.append(card)

# --------------------------------------------------------------------------
# 5. Save GIF under a content-hash name (beats GitHub's image cache)
# --------------------------------------------------------------------------
pal = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
durations = [3000] + [3500] * (len(pal) - 1)

tmp_gif = os.path.join(tempfile.mkdtemp(), "showcase.gif")
pal[0].save(
    tmp_gif,
    save_all=True,
    append_images=pal[1:],
    duration=durations,
    loop=0,
    disposal=2,
)

with open(tmp_gif, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()[:8]

gif_name = f"showcase-{digest}.gif"
os.makedirs(out_dir, exist_ok=True)
gif_path = os.path.join(out_dir, gif_name).replace(os.sep, "/")
shutil.copy(tmp_gif, gif_path)
print(f"Generated {gif_path} ({len(frames)} frames, {os.path.getsize(gif_path) / 1024:.0f} KB)")

# ---- Cleaner: remove every older showcase GIF ----
for fname in os.listdir(out_dir):
    if GIF_NAME.match(fname) and fname != gif_name:
        os.remove(os.path.join(out_dir, fname))
        print(f"Cleaned up old file: {fname}")

for legacy in ("showcase.gif", "profile/showcase.gif", ".github/profile/showcase.gif"):
    if os.path.isfile(legacy):
        os.remove(legacy)
        print(f"Cleaned up legacy file: {legacy}")


# ---- Point the README at the new file ----
def public_url(path):
    if REPO:
        return f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
    return os.path.relpath(path, readme_dir or ".").replace(os.sep, "/")


def update_readme(new_url):
    with open(readme_path, "r", encoding="utf-8") as f:
        text = f.read()

    mine = 0

    def swap(m):
        nonlocal mine
        tok = m.group(0)
        # Leave links that clearly point at some other repo alone
        if "://" in tok and REPO and REPO.lower() not in tok.lower():
            return tok
        mine += 1
        return new_url

    new_text, _ = GIF_REF.subn(swap, text)
    n = mine

    if n == 0:
        marker = re.compile(r"(<!--\s*SHOWCASE:START\s*-->)(.*?)(<!--\s*SHOWCASE:END\s*-->)", re.S)
        if marker.search(new_text):
            block = f'\n<img src="{new_url}" alt="{STUDIO_NAME.title()} showcase" width="100%">\n'
            new_text = marker.sub(lambda m: m.group(1) + block + m.group(3), new_text)
            n = 1

    if n == 0:
        print("::warning::No showcase link found in the README. Add this where you want the GIF:")
        print(f'  <!-- SHOWCASE:START -->\n  <!-- SHOWCASE:END -->')
        return
    if new_text != text:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(new_text)
        print(f"README updated to point at {gif_name}")
    else:
        print("README already points at the latest GIF.")


update_readme(public_url(gif_path))
