# Install five XTree-style file managers on Ubuntu 26.04

**Host:** morlok-AM4
**User:** morlok
**Distro:** Ubuntu 26.04 LTS, x86_64, GCC 15.2.0, ncurses 6.6, glibc current
**Date:** 2026-05-09 06:55 UTC
**Sources:** unixtree.org, xtreefanpage.org/x64linux.htm, stahlke.org/dan/lxt/

## Goal

Install five classic XTreeGold-style ncurses file managers side-by-side
under `/usr/local`. URL #2 (xtreefanpage.org) turned out to be a hub page
hosting three projects, so the actual install set was: UnixTree, utree,
XTC, lxt 1.3a, lxt 1.3c.

## Final state

```
/usr/local/bin/xt           UnixTree 3.0.4 build 2064  (1178 KB)
/usr/local/bin/xtx          UnixTree 3.0.4 build 2064  (1374 KB, X11 variant)
/usr/local/bin/lxt          linuXtree 1.3c (Dan Stahlke, BSD)  (592 KB)
/usr/local/bin/lxt-1.3a     linuXtree 1.3a (renamed)            (597 KB)
/usr/local/bin/xtc          XTC 0.1.9 (GPL, dev status)         (691 KB)
/usr/local/bin/utree        utree 3.04a-um (1992)               (188 KB)
/usr/local/bin/utree.prlist utree print formatter                (18 KB)
```

All sources kept under `~/src/xtree-clones/` for future `make install`/uninstall.

## Build patches needed (per project)

GCC 15.2 / glibc-current promoted many legacy patterns to hard errors. The
cumulative patch set:

### UnixTree (github.com/dokakod/unixtree)

Custom build system: `. build -r linux && make`. INS_DIR defaults to
`/usr/local/xt`; we installed via `install -m 0755` directly to
`/usr/local/bin/{xt,xtx}` to avoid an extra subdir.

`env/linux.env` extended:
```
CDEFS="$CDEFS -Wno-error=implicit-int -Wno-error=implicit-function-declaration \
              -Wno-error=int-conversion -Wno-error=incompatible-pointer-types \
              -Wno-error=strict-prototypes -Wno-error=old-style-definition \
              -fcommon"
```

### lxt 1.3c (stahlke.org/dan/lxt/lxt-1.3c.tar.gz)

`./configure --prefix=/usr/local && make CFLAGS="-O2 -g -std=gnu17 -Wno-error" && sudo make install`.
The `-std=gnu17` avoids C23 `strchr` returning const-qualified pointer
(line `*strchr(buf,'\n')=0;` in readconfig.c).

### lxt 1.3a (stahlke.org/dan/lxt/lxt-1.3a.tar.gz — fanpage's link to
modzer0.cs.uaf.edu was stale; fetched from Stahlke's main site)

Same CFLAGS as 1.3c plus `-DNCURSES_OPAQUE=0 -DNCURSES_INTERNALS=1`
(18 sites in gui.c/commands.c/readconfig.c access `WINDOW->_maxx/_maxy`
directly — modern ncurses makes WINDOW opaque). `make install` skipped
(would clobber 1.3c); binary copied to `/usr/local/bin/lxt-1.3a`.

### XTC 0.1.9 (xtreefanpage.org/download/xtc-0.1.9.tar.gz)

```
CFLAGS="-O2 -g -std=gnu17 -Wno-error -Wno-error=implicit-int \
        -Wno-error=implicit-function-declaration -fcommon \
        -DNCURSES_OPAQUE=0 -DNCURSES_INTERNALS=1" ./configure --prefix=/usr/local
```
The configure test program (`main(){return(0);}`) is K&R-style; needs
relaxed CFLAGS at configure time too. `-fcommon` reverses GCC 10's
`-fno-common` default to fix the multi-defined-globals link error
(globals declared in headers without `extern`).

Source patches:
- `info_window.c:22`: `static int info_window_drawn;` → `int info_window_drawn;`
  (header says `extern int info_window_drawn` — the static was a true bug).
- `modechange.c:43-44`: removed redundant `char *malloc ();` redecl
  (conflicts with `<stdlib.h>` prototype).

Makefile has no `install` target; `xtc` binary copied manually to
`/usr/local/bin/`.

### utree 3.04a-um (xtreefanpage.org/download/utree3.04-src.tar.gz)

Largest port — a 1992 SVR3/BSD codebase. Steps:

1. Picked sys/Makefile.V.4 + sys/conf.h.V.4 as starting templates.
2. **Makefile bug** (suffix rule): the original
   ```
   $(OBJS): $(DEFS)
       $(CC) $(CFLAGS) -c $<
   ```
   uses `$<` as the first prereq — always `conf.h`, not the .c file.
   Replaced with explicit `.SUFFIXES: .c .o` + `.c.o:` rule and a
   dependency-only `$(OBJS): $(DEFS)`. Also added `.PHONY: utree
   utree.prlist all clean strip` so make's built-in `%: %.c` rule does
   not double-compile `utree.prlist`.
3. **`<termio.h>` removed in glibc**: `<termio.h>` → `<termios.h>` in
   `defs.h`; `struct termio` → `struct termios` in `term.c`;
   `ioctl(...,TCGETA,...)` → `tcgetattr(...)` and `ioctl(...,TCSETAW,...)`
   → `tcsetattr(...,TCSADRAIN,...)`.
4. **`<varargs.h>` removed in GCC 14+**: `defs.h` swap to `<stdarg.h>`,
   plus four functions rewritten from K&R varargs to ANSI stdarg:
   `putfxy`, `putecho`, `puthelp` in echo.c; `putpage` in page.c.
   (Pattern: `func(args, va_alist) va_dcl { va_start(ap); fmt =
   va_arg(ap, char *); … }` → `func(args, char *fmt, ...) {
   va_start(ap, fmt); … }`).
5. **POSIX `getline()` collision**: utree had a function named `getline()`
   — same name as glibc's POSIX `<stdio.h>` `getline()` but different
   signature. Renamed to `ut_getline` across 43 sites in 9 files.
6. CFLAGS: `-O -DSIGNL=void -std=gnu17 -Wno-error -Wno-error=implicit-int
   -Wno-error=implicit-function-declaration -fcommon -DNCURSES_OPAQUE=0
   -DNCURSES_INTERNALS=1`.
7. utree's Makefile install target hard-codes SVR4 paths; binary +
   man page copied manually.
8. **Two runtime crashes fixed during smoke test** (the binary built
   but segfaulted on launch):
   - `getenv()` truncation: `defs.h` did not include `<stdlib.h>`, so
     `getenv` defaulted to `int` return type.  The `(char *)` cast in
     `home = (char *) getenv("HOME")` then sign-extended a 32-bit
     truncation back into a 64-bit pointer like `0xffffffffffffe371`.
     `pathname()` strcpy'd through it and crashed.  Added
     `<stdlib.h>`, `<unistd.h>`, `<string.h>` to `defs.h`.
   - Read-only string literal write in `vars.c:initvariables`: original
     code did `VARVAL(i) = VARDEF(i)` so the per-variable `value` field
     pointed at a `.rodata` literal like `"3"`.  Code paths in
     `setvideomode`, `settreeindent`, `setfilelines` then do
     `*VARVAL(V_VM/V_TI/V_FL) = digit + '0'` — segfault on Linux where
     literals are read-only.  Patch: `strdup(VARDEF(i))` for
     `VARTYP(i) == VT_N` (numeric) only.  Booleans use `VB_ON` =
     `(char *)1` and `VB_OFF` = `(char *)0` as sentinel pointers, which
     would crash `strdup` — kept the original assignment for those.

## Runtime quirks discovered after install

### xt / xtx — terminal database miss on `xterm-256color`

UnixTree ships its own `.trm` files (xterm, linux, vt100/220/320, ansi, …)
and looks them up by literal `$TERM`. Modern terminals advertise
`xterm-256color`, which is not in the bundled set, so `xt` exits with:

```
No term file or database entry found for term "xterm-256color"
xt: cannot initialize terminal.
```

Workaround applied: shell aliases in `~/.zshrc:43-44`:

```
alias xt='xt -t xterm'
alias xtx='xtx -t xterm'
```

Both then work in any modern terminal.

### xtx — X11-aware but still TTY-rendered

`xtx` links `libX11` (`XOpenDisplay`, `XCreateSimpleWindow`,
`XAllocColor`, font load), so I expected a graphical window. It does
**not** open one; the file-manager UI is drawn in the host terminal via
xterm OSC + CSI sequences (visible in capture: `OSC ]0;UnixTree`,
`CSI 25;44m` etc.). The X11 connection is used as a side channel for
named-color allocation, custom cursor/border colors, font handles, and
all-motion mouse reporting. Without `$DISPLAY`, `xtx` still runs and
renders fine — it just degrades to the same featureset as `xt`.

### utree — runs cleanly only with a real keyboard tty

Under `script(1)` (PTY) with a closed-stdin pipe, utree's main
`poll → read → write` loop tight-spins (~2700 polls/sec) because the
EOF on the pipe satisfies poll without blocking. In a real interactive
terminal this does not happen — kernel tty read blocks on keypress, so
CPU is idle until input arrives. `utree` from a regular zsh prompt is
fine.

## How to drive each binary

```
xt        # UnixTree TTY (alias adds -t xterm)
xtx       # UnixTree TTY + X11 colors/mouse (alias adds -t xterm)
lxt       # linuXtree 1.3c
lxt-1.3a  # linuXtree 1.3a (kept for archaeology)
xtc       # XTC 0.1.9 (closest to literal XTreeGold key bindings; dev/incomplete)
utree     # utree 3.04a-um from 1992
```

Common keys (XTreeGold lineage): arrows / pgup-pgdn navigate, `Tab`
toggles tree↔file pane, `Enter` views, `T`/`U` tag/untag, `q` (often
followed by `y`) or `Esc` to quit, `F1`/`?` for help, `F10` cancel.

## Workflow notes

- `xtreefanpage.org/x64linux.htm` link to `modzer0.cs.uaf.edu/~dan/lxt/lxt-1.3a.tar.gz`
  is dead (no archive in Wayback either). Fetched from
  `stahlke.org/dan/lxt/lxt-1.3a.tar.gz` instead — Stahlke keeps old
  versions on his main site.
- sudo on this box requires password (no NOPASSWD). Final installs done
  via two `sudo bash <script>` invocations:
  `~/src/xtree-clones/install.sh` (UnixTree, lxt-1.3c, lxt-1.3a) and
  `install-rest.sh` (XTC, utree). utree was reinstalled a third time
  after the runtime crash fixes:
  `sudo install -m 0755 ~/src/xtree-clones/utree.um/bin/utree /usr/local/bin/utree`.
- Smoke-test technique that worked for all five: drive the binary
  through `script -q -e -c "TERM=xterm <bin>" /tmp/cap.typescript`
  while feeding `q`/`Esc` from a `(sleep; printf …)` pipeline; strip
  ANSI/charset-shift escapes with
  `sed -E 's/\x1B\[[0-9;?]*[a-zA-Z]//g; s/\x1B[()][AB012]//g; …'` to
  read the painted UI as plain text.  When that segfaulted, switching
  to `gdb -batch -ex run -ex bt` against the same input gave usable
  stack traces (`pathname()` → `__strcpy_chk` for the truncated-pointer
  bug; `initvariables` → `__strdup` for the read-only-literal bug).

## Verification

```
$ for n in xt xtx lxt lxt-1.3a xtc utree utree.prlist; do command -v $n; done
/usr/local/bin/xt
/usr/local/bin/xtx
/usr/local/bin/lxt
/usr/local/bin/lxt-1.3a
/usr/local/bin/xtc
/usr/local/bin/utree
/usr/local/bin/utree.prlist

$ xt -ZV         → "UnixTree Version 3.0.4 Build 2064 …"
$ xtx -ZV        → "UnixTree Version 3.0.4 Build 2064 …"
$ lxt -V         → "linuXtree 1.3c copyright (c) 2008 Dan Stahlke …"
$ lxt-1.3a -V    → "linuXtree 1.3a copyright (c) 2008 Dan Stahlke …"
$ utree.prlist -? → prints usage banner ("Usage: utree.prlist [-Vr]…")
$ utree </dev/null → "utree: Not attached to a terminal" (exits 0;
                     binary functional, requires a tty for full UI)
$ xtc            → launches ncurses UI (no --version/--help flag in 0.1.9)
```

## ZTreeWin (ztree.com) — considered, skipped

After the five clones above, ztree.com was evaluated. Only Windows
binaries are distributed:
- `ztw22x86.exe` — Windows 32-bit, v2.2.19
- `ztw22x64.exe` — Windows 64-bit, v2.2.19
- `ztw22.zip`   — same, zipped

License: 30-day free trial, then $29.95 (1–4 copies). No native Linux
build, no source. Linux path would be Wine. Skipped — the five native
clones already cover the use case.

## Files touched (this session)

- `~/src/xtree-clones/` — full source trees + patches, retained
  (`unixtree/`, `lxt-1.3c/`, `lxt-1.3a/`, `xtc-0.1.9/`, `utree.um/`,
  plus the two `install*.sh` scripts).
- `/usr/local/bin/{xt,xtx,lxt,lxt-1.3a,xtc,utree,utree.prlist}` — installed.
- `/usr/local/share/man/man1/{utree,utree.prlist}.1` — utree man pages.
- `~/.zshrc:43-44` — `alias xt='xt -t xterm'` and
  `alias xtx='xtx -t xterm'` (so plain `xt`/`xtx` work in
  `xterm-256color` terminals).
- This document.

## Re-build / uninstall

To rebuild any project from scratch with the patches applied, the source
trees under `~/src/xtree-clones/` are ready — each was built with the
exact CFLAGS recorded above and (where applicable) kept its `Makefile`
fixes in tree.

Uninstall is straightforward — each binary is a single file in
`/usr/local/bin/`. Example: `sudo rm /usr/local/bin/{xt,xtx}` removes
UnixTree; `cd ~/src/xtree-clones/lxt-1.3c && sudo make uninstall`
removes lxt 1.3c via its autotools target.
