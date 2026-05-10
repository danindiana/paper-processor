# Session: Install lite-xl v2.1.8 from source on the morlok host

**Timestamp (UTC):** 2026-05-09T045435Z
**Local:** 2026-05-08 23:54:35 CDT
**Host / user:** local machine, `morlok`
**Repo:** paper_processor (used only as the home for session docs; no source files changed)

---

## Goal

Install lite-xl (https://github.com/lite-xl/lite-xl) on `morlok` from
source, with the upstream curated addons bundle and a desktop launcher.
No `lite-xl` package is in Ubuntu Resolute repos, so go straight to
upstream.

User decisions (collected in plan mode):

1. Build from source (chosen over the prebuilt tarball / AppImage).
2. Include the curated addons bundle.
3. Add a desktop-menu launcher.

---

## What was done

### 1. Build dependencies

Initial probe showed five apt packages missing: `meson`, `ninja-build`,
`libsdl2-dev`, `libpcre2-dev`, `liblua5.4-dev`. User installed them via
`sudo apt-get install -y …` in a real terminal (passwordless sudo not
available; the harness can't authenticate sudo).

**Surprise during configure:** lite-xl v2.1.8 needs **SDL3**, not SDL2.
The upstream README is outdated — it still mentions SDL2 build deps,
but `src/meson.build:62` calls `dependency('sdl3', static: true)`. The
SDL2 packages were already installed at that point; they're harmless and
unused. User did a second apt round for `libsdl3-dev` (the runtime
`libsdl3-0` 3.4.2+ds-1ubuntu1 was already present from the base
distro — only the `-dev` headers were missing).

Final installed versions:

| Package          | Version           |
| ---------------- | ----------------- |
| meson            | 1.10.1-1ubuntu2   |
| ninja-build      | 1.13.2-1          |
| libsdl3-dev      | 3.4.2+ds-1ubuntu1 |
| libpcre2-dev     | 10.46-1build1     |
| liblua5.4-dev    | 5.4.8-1build1     |
| libfreetype-dev  | 2.14.2+dfsg-1     |
| build-essential, git, pkg-config | already present |

### 2. Clone

```bash
mkdir -p ~/src
git clone --depth=1 --branch=v2.1.8 https://github.com/lite-xl/lite-xl.git ~/src/lite-xl
```

Detached HEAD at `0af4dae0` ("chore: bump versions [skip ci]"), tag
`v2.1.8`.

### 3. Configure / build / install

```bash
cd ~/src/lite-xl
meson setup build --buildtype=release --prefix="$HOME/.local" -Duse_system_lua=true
meson compile -C build
meson install -C build
```

Notes from configure:

- `-Duse_system_lua=true` was **essential** — without it, meson tries to
  pull `lua-5.4.4` as a wrap subproject and download from
  `https://www.lua.org/ftp/lua-5.4.4.tar.gz`. Since `liblua5.4-dev`
  (5.4.8) is already installed, system lua is the right choice.
- `dirmonitor_backend: inotify` was auto-selected (Linux default).
- `meson` warned about static SDL3 sub-deps (asound, pulse, wayland-*,
  egl, xkbcommon, decor, etc.) not having static libraries — Ubuntu only
  ships SDL3 as a shared library. End result: SDL3 itself is statically
  embedded in the lite-xl binary (no `libSDL3.so` in the resulting
  `ldd`), but its transitive deps are dynamically linked.

Build emitted two C warnings (unused-result on a `write()` in
`api/process.c:553`, unused-variable in `renderer.c:539`); both
upstream code, both harmless.

### 4. Curated addons

Downloaded the matching addons release tarball, extracted, copied the
addon plugins/colors into the user data dir:

```bash
URL="https://github.com/lite-xl/lite-xl/releases/download/v2.1.8/lite-xl-v2.1.8-addons-linux-x86_64-portable.tar.gz"
TMP=$(mktemp -d) && cd "$TMP"
curl -fsSL -o addons.tar.gz "$URL"
tar -xzf addons.tar.gz
mkdir -p ~/.config/lite-xl/{plugins,colors}
cp -rn lite-xl/data/plugins/* ~/.config/lite-xl/plugins/
cp -rn lite-xl/data/colors/*  ~/.config/lite-xl/colors/
```

Counts after install:

| Location                          | Plugins | Colors |
| --------------------------------- | ------- | ------ |
| `~/.local/share/lite-xl/` (core)  | 26      | 4      |
| `~/.config/lite-xl/` (addons)     | 131     | 55     |

User-config addons take precedence when names collide.

### 5. Desktop entry

Already shipped by the upstream meson install; no manual desktop-entry
authoring needed.

```
/home/morlok/.local/share/applications/org.lite_xl.lite_xl.desktop
/home/morlok/.local/share/icons/hicolor/scalable/apps/lite-xl.svg
/home/morlok/.local/share/metainfo/org.lite_xl.lite_xl.appdata.xml
```

(The plan had predicted the desktop-entry filename would be
`com.lite_xl.LiteXL.desktop` based on a binary-strings hit; upstream
actually ships it as `org.lite_xl.lite_xl.desktop`. Same effect.)

XDG cache refresh: `update-desktop-database` ran clean.
`gtk-update-icon-cache` reported "no theme index file" for
`~/.local/share/icons/hicolor` — expected and harmless on a fresh user
icon dir; KDE finds the icon via the standard XDG icon search path
regardless.

### 6. Verification

| #  | Check | Result |
| -- | ----- | ------ |
| 1  | `which lite-xl`                                    | `/home/morlok/.local/bin/lite-xl`                                                  |
| 2  | embedded version (`strings $(which lite-xl)` …)    | `2.1.8` (lite-xl has no `--version` CLI flag — it's a GUI app)                     |
| 3  | `ldd $(which lite-xl) \| grep "not found"`         | empty — all libs resolved                                                          |
| 4  | SDL3 link form                                     | statically embedded; no `libSDL3.so` in `ldd`; transitive deps dynamic             |
| 5  | desktop entry installed                            | `org.lite_xl.lite_xl.desktop` present, `Exec=lite-xl %F`, `Categories=Development;IDE;` |
| 6  | icon installed                                     | `lite-xl.svg` at scalable/apps                                                     |
| 7  | addons present                                     | 131 plugins + 55 colors in `~/.config/lite-xl/`                                    |
| 8  | smoke launch (with display)                        | process alive at +1s, 7.5 MB RSS; killed cleanly by `timeout`                      |
| 9  | smoke launch (without DISPLAY/WAYLAND_DISPLAY)     | hung waiting on SDL init then killed by timeout — expected                         |
| 10 | install footprint                                  | binary 5.4 MB, core data 1.2 MB, user addons 1.2 MB, source clone 18 MB            |

---

## Files / system state changed

| Path                                                              | Change                                                            |
| ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| apt: `meson`, `ninja-build`, `libsdl2-dev`, `libsdl3-dev`, `libpcre2-dev`, `liblua5.4-dev` | installed (`libsdl2-dev` is unused — leave or remove later)       |
| `~/src/lite-xl/`                                                  | shallow clone at tag `v2.1.8` (18 MB)                             |
| `~/.local/bin/lite-xl`                                            | new binary (5.4 MB, SDL3 statically linked)                       |
| `~/.local/share/lite-xl/`                                         | core Lua scripts, fonts, builtin plugins (1.2 MB)                 |
| `~/.local/share/icons/hicolor/scalable/apps/lite-xl.svg`          | new                                                               |
| `~/.local/share/applications/org.lite_xl.lite_xl.desktop`         | new                                                               |
| `~/.local/share/metainfo/org.lite_xl.lite_xl.appdata.xml`         | new                                                               |
| `~/.local/share/doc/lite-xl/licenses.md`                          | new                                                               |
| `~/.config/lite-xl/plugins/` (131 files), `~/.config/lite-xl/colors/` (55 files) | new — curated addon bundle                                        |
| `docs/sessions/2026-05-09T045435Z_install-lite-xl.md`             | new — this document                                               |

`~/.bashrc`, `~/.zshrc`, the `paper_processor` source tree: untouched.

---

## How to use

- **From a terminal:** `lite-xl` (or `lite-xl path/to/file` /
  `lite-xl path/to/dir`).
- **From the KDE menu / KRunner:** search "Lite XL".
- **Configure:** edit `~/.config/lite-xl/init.lua`. Plugins are loaded
  by name from `~/.config/lite-xl/plugins/`; in-app, open the command
  palette (`ctrl+shift+p`) and search "core: load plugin" to enable
  a specific one, or `core: open user module` to edit `init.lua`.
- **Pick a theme:** `ctrl+shift+p` → "core: change color theme".

## Update path

```bash
cd ~/src/lite-xl
git fetch --tags
git checkout vNEXT      # e.g. v2.1.9 when released
meson compile -C build
meson install -C build
```

Re-run the addons step (with the matching tarball URL) if v$NEXT bumps
its addons too.

## Uninstall

```bash
rm -rf \
    ~/src/lite-xl \
    ~/.local/bin/lite-xl \
    ~/.local/share/lite-xl \
    ~/.local/share/icons/hicolor/scalable/apps/lite-xl.svg \
    ~/.local/share/applications/org.lite_xl.lite_xl.desktop \
    ~/.local/share/metainfo/org.lite_xl.lite_xl.appdata.xml \
    ~/.local/share/doc/lite-xl \
    ~/.config/lite-xl
update-desktop-database ~/.local/share/applications 2>/dev/null || true
```

The apt build deps can stay (cheap, useful for other source builds) or
go: `sudo apt-get autoremove --purge meson ninja-build libsdl3-dev
libsdl2-dev libpcre2-dev liblua5.4-dev`.

## Lessons / notes

- **Upstream README is out of date for v2.1.8** — it lists SDL2 as a
  build dep but the build actually requires SDL3. The cost was one
  extra apt round and a stale `libsdl2-dev` package on disk. For future
  third-party-from-source installs, do `meson setup` against current
  HEAD before promising a final dep list to the user.
- **Always pass `-Duse_system_lua=true`** if `liblua5.4-dev` is
  installed. Otherwise meson silently downloads lua-5.4.4 from
  `lua.org`, which (a) wastes time, (b) ties the build to network
  reachability, (c) embeds a different lua version than the rest of the
  system.
- **SDL3 statically embedded** — the `static: true` flag in lite-xl's
  meson is honored when SDL3 ships as `.a`, otherwise meson falls back
  but in this case static linking did succeed for libSDL3 itself. The
  binary works on this exact distro; if you copy the binary to another
  machine, the *dynamic* deps still need to be present there.
- **Wayland**: SDL3 talks Wayland natively; the smoke launch worked
  under `XDG_SESSION_TYPE=wayland`. No env-var tweaks needed.
- **Sudo + harness**: same lesson as the zsh install — `morlok` does
  not have passwordless sudo, the harness is non-TTY, so any `sudo`
  step has to be run by the user in a real terminal.
- **Worlock host (user `jeb`)** is **not in scope** — only `morlok`
  changed.
