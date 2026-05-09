# Session: Install Midnight Commander + Dracula skin (Eva-01-adjacent palette) on morlok

**Timestamp (UTC):** 2026-05-09T050501Z
**Local:** 2026-05-09 00:05:01 CDT
**Host / user:** local machine, `morlok` (hostname `morlok-AM4`), user `morlok`
**Repo:** paper_processor (used only as the home for session docs; no source files changed)

---

## Goal

Install Midnight Commander (`mc`) and apply a color theme. User asked for
"Evangelion Eva-01 style or Dracula if available."

There is no canonical Eva-01 skin shipped with mc and none upstream that
I could find. Dracula (`dracula256.ini`) is the chosen substitute — its
palette (purple `#BD93F9`, pink `#FF79C6`, neon green `#50FA7B`, yellow
`#F1FA8C` on dark `#282A36`) is the closest off-the-shelf match to
Eva-01's purple body / neon-green-and-yellow accents. The user explicitly
authorized Dracula as the fallback, so no further hunt for a custom
Eva skin was undertaken.

---

## What was done

### 1. Install check — already present

```text
$ which mc
/usr/bin/mc
$ mc --version | head -1
GNU Midnight Commander 4.8.33
$ dpkg -l mc | tail -1
ii  mc  3:4.8.33-1.1build1  amd64  Midnight Commander - a powerful file manager
$ apt-cache policy mc
  Installed: 3:4.8.33-1.1build1
  Candidate: 3:4.8.33-1.1build1
  500 http://archive.ubuntu.com/ubuntu resolute/universe amd64 Packages
```

mc was already on the system from the Ubuntu Resolute `universe` repo.
No `apt-get install` required, which is a nice side-effect: passwordless
sudo is not configured for `morlok`, so an apt step would have needed
the user to run it in a real terminal (same lesson as the lite-xl and
zsh installs in this folder).

### 2. Skin survey — no Dracula or Eva variant in `/usr/share/mc/skins/`

```text
$ ls /usr/share/mc/skins/
dark.ini, darkfar.ini, default.ini, double-lines.ini,
featured.ini, featured-plus.ini, gotar.ini,
gray-green-purple256.ini, gray-orange-blue256.ini,
julia256.ini, julia256root.ini, mc46.ini,
modarcon16{,-thin,-defbg,-defbg-thin}.ini, modarcon16root{...}.ini,
modarin256{,-thin,-defbg,-defbg-thin}.ini, modarin256root{...}.ini,
nicedark.ini, sand256.ini,
seasons-{autumn,spring,summer,winter}16M.ini,
xoria256{,-thin}.ini, xoria256root-thin.ini,
yadt256{,-defbg}.ini
```

No Dracula. Closest in vibe: `gray-green-purple256` (purple+green, but
washed out), `julia256` (deep purple), `modarin256` (dark cyan/blue).
Dropping all of them in favor of upstream Dracula.

### 3. Fetch Dracula skin from upstream

Repo: `https://github.com/dracula/midnight-commander`. The skins live
in `skins/`, not the repo root — first attempt at
`raw.githubusercontent.com/.../master/dracula.ini` returned 404. Used
the GitHub contents API to discover the correct path:

```text
$ curl -fsSL https://api.github.com/repos/dracula/midnight-commander/contents/skins
dracula.ini, dracula256.ini
```

Downloaded both:

```bash
mkdir -p ~/.local/share/mc/skins ~/.config/mc
curl -fsSL -o ~/.local/share/mc/skins/dracula.ini    \
    https://raw.githubusercontent.com/dracula/midnight-commander/master/skins/dracula.ini
curl -fsSL -o ~/.local/share/mc/skins/dracula256.ini \
    https://raw.githubusercontent.com/dracula/midnight-commander/master/skins/dracula256.ini
```

Result:

```text
$ ls -l ~/.local/share/mc/skins/
-rw-rw-r-- 1 morlok morlok 3161 May 9 00:04 dracula.ini
-rw-rw-r-- 1 morlok morlok 3159 May 9 00:04 dracula256.ini
```

Section sanity-check (17 sections — complete):
`[skin] [Lines] [core] [dialog] [error] [filehighlight] [menu]`
`[popupmenu] [buttonbar] [statusbar] [help] [editor] [viewer]`
`[diffviewer] [widget-common] [widget-panel] [widget-editor]`

### 4. Pick variant — `dracula256` (truecolor terminal)

```text
$ echo $TERM       → xterm-256color
$ tput colors      → 256
$ echo $COLORTERM  → truecolor
```

Per the upstream README, `dracula.ini` (8/16-color) is "truer" only when
the terminal *profile* is configured with the exact Dracula palette;
on a generic 256-color terminal the closest-match `dracula256.ini`
gets nearer to the canonical Dracula colors. Picked `dracula256`.

### 5. Activate skin via `~/.config/mc/ini`

There was no existing mc config dir (`~/.config/mc` did not exist
before this session). Wrote the minimum to select the skin:

```ini
[Midnight-Commander]
skin=dracula256
```

mc fills in defaults for everything else on first launch.

---

## Files / system state changed

| Path | Change |
| --- | --- |
| `~/.local/share/mc/skins/dracula.ini` | new (3161 B, upstream 8/16-color variant — kept as fallback) |
| `~/.local/share/mc/skins/dracula256.ini` | new (3159 B, upstream 256-color variant — **active**) |
| `~/.config/mc/ini` | new — `[Midnight-Commander] skin=dracula256` |

No apt changes (mc was already installed). No system-level edits.
`paper_processor` source tree untouched.

---

## How to use

- **Launch:** `mc` (read+write), `mc -b` (force black-and-white),
  `mc -S <skinname>` (override skin for this run).
- **Switch skin in-app:** `F9` → Options → Appearance → pick from list.
  The two new skins show as `dracula` and `dracula256`.
- **Switch back to default:** edit `~/.config/mc/ini`, change
  `skin=dracula256` to `skin=default` (or delete the line).
- **Common keys:** `F1` help, `F3` view, `F4` edit, `F5` copy, `F6`
  rename/move, `F7` mkdir, `F8` delete, `F9` menu, `F10` quit,
  `Ctrl+O` toggle subshell, `Tab` switch panel, `Ctrl+\` directory
  hotlist.

## Why Dracula maps reasonably to "Eva-01 style"

| Eva-01 element | Closest Dracula color |
| --- | --- |
| Body purple/violet | `#BD93F9` (dracula purple) |
| Eye / neon green accents | `#50FA7B` (dracula green) |
| Horn / yellow detailing | `#F1FA8C` (dracula yellow) |
| Pilot suit pink | `#FF79C6` (dracula pink) |
| Background black-violet | `#282A36` (dracula bg) |

It is *not* a pixel-perfect Eva palette — it lacks the very saturated
neon yellow-green and the orange-red of armor highlights — but the
purple-dominant + green-accent character is recognizably similar, and
it's what the user pre-authorized as the substitute.

## Update path

```bash
cd /tmp
curl -fsSL -O https://raw.githubusercontent.com/dracula/midnight-commander/master/skins/dracula256.ini
mv dracula256.ini ~/.local/share/mc/skins/
# (likewise dracula.ini)
```

(Or symlink from a clone of the repo, per the upstream INSTALL.md.)

## Uninstall

```bash
rm -f ~/.local/share/mc/skins/dracula.ini ~/.local/share/mc/skins/dracula256.ini
# revert skin selection — set skin=default or remove ~/.config/mc/ini entirely
sed -i 's/^skin=.*/skin=default/' ~/.config/mc/ini
# to remove mc itself:
sudo apt-get autoremove --purge mc
```

## Lessons / notes

- **Already installed.** Always check `which mc` / `dpkg -l mc` before
  doing apt work — saved a sudo round-trip the harness can't perform
  anyway (no passwordless sudo on `morlok`).
- **Repo path quirk:** the Dracula mc repo keeps skins in `skins/`,
  not the repo root. The first guess at the raw URL 404'd. Hitting
  the `/contents/<dir>` API once is cheap and removes the guesswork.
- **Pick the variant the terminal can actually render.** With
  `TERM=xterm-256color` and `COLORTERM=truecolor`, `dracula256` is the
  right call — `dracula.ini` is for terminals whose 16-color palette
  has been remapped to Dracula. Both are kept on disk so either is
  selectable from the F9 menu.
- **Eva-01 is a vibes match, not an exact one.** If a closer-to-Eva
  skin is needed later, the path forward is to fork `dracula256.ini`,
  shift `core.base_color` toward `#1B0033` (deeper indigo/black-violet)
  and remap the green to a more saturated `#A6FF00`. Not done here
  because the user accepted Dracula as the fallback up front.
- **Worlock host (user `jeb`) is not in scope** — only `morlok`
  changed, despite the project files living on the `WORLOCK` mount.
