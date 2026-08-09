# Global Claude Code Instructions

## Primary Languages and Frameworks

I work primarily in **Python**, **C**, and **C++**.  I prefer CMake-based build systems for all projects.  I always want to have *BOTH* Python bindings and C/C++ libraries for projects and APIs.  When I work on web frontends, I work in **JavaScript/HTML**.

My applications need to run on Linux, OSX and Windows 11.  Always create build targets for all of these platforms and bootstrap and install scripts for the platforms.

Match the language of the existing project, but warn me if it is not compatible with my preferences.

### Database Preferences

- Use SQLite databases for most projects, but provide hooks for switching to Postgres database servers

### Documentation Preferences

- Always generate man pages for all scripts and executables
- Always generate man pages for APIs
- Always generate instructions in markdown for how to install, bootstrap, build and run a project
- Always create a README.md file for projects which can be used with GitHub
- For web applications always create an About page which documents the project dependances
- For web applications always create an API page that documents the API and routes that the application supports.
- For web sites and other web applictions always create a sitemap page that documents their structure.

### Commandline Utilities

- All scripts and applications should support commandline options
- All scripts and applications should have their commandline options override configuration parameters provided by files or environment variables.

### Python Preferences

- For Linux systems maintain compatiblity with Python 3.9
- For graphical interfaces in Python use a QT based framework (PyQt6 or PySide6). PyQt5 is acceptable for compatiblity with older platforms.  For simple projects tkinter is acceptable.
- Use **PyYAML** for config files — YAML is the standard config format across projects
- Use JSON as a format for dumping and restoring data
- Use **Flask** for simple web applications that do not require user logins
- Use **DJango** for complex web applications that require user logins or have more complex database setups
- **Litestar** is acceptable if asynchronous performance is needed
- Use **SQLAlchemy** for database access; prefer async sessions in Litestar apps, sync in daemon threads
- Use **pyzmq** for inter-process messaging when needed
- Use pip and Python virtual environments for package management.
- Each project should have a Python virtual environment setup in the directory "venv"
- Always generate and update the requirements.txt file for depedencies.
- Always generate a bootstrap script which sets up the project and updates packages
- Prefer `pyproject.toml` for new project packaging over `setup.py`
- Use **PyTest** for test suites

### C++ Preferences

- Use QT as the framework for graphical applications
- Maintain compatiblity with the C++17 standard
- **OpenMP** should be used for parallelism
- Use C++ builtin constructs and STL containers and algorithms when possible
- Use **Boost** libraries when possible 
- Use **CPPunit** for unit tests

### Web Frontend Preferences

- Use **HTMX** + **Jinja2** templates for partial page updates — avoid heavy JS frameworks
- Use **Tailwind CSS** for styling
- Prefer vanilla JavaScript for client-side logic; keep it minimal
- Server-Sent Events (SSE) for real-time push to the browser

### Threading and Concurrency

- Background workers run as **daemon threads** (so they don't block clean shutdown)
- In Qt apps, daemon threads communicate back to the UI thread via **Qt signals** — never update UI directly from a background thread
- In Litestar apps, use async SQLAlchemy for route handlers, sync DB operations from background threads

## Project Structure

- CMake files at top level of directory 
- Build in `build/`
- Python source under `src/<packagename>/`
- C/C++ source under `src/cpp/`
- Include headers in `src/include/`
- Manpages in `man/`
- Webpages in `web/`
- Config files in `config/` as YAML
- Core dumps in `core/`
- Tests under `tests/` using **pytest**; use `pytest-qt` for Qt app tests
- Startup scripts as `start-<applicationname>.sh` (creates venv, installs and updates deps, launches app)
- Stop scripts as `stop-<applicationname>.sh` (stops app cleanly, kills app if needed, cleans up temp files)
- Static data (lookup tables, item databases) in `data/`

## Design Principles

- **Config-driven**: behavior should be externally configurable via YAML — avoid hardcoding values that operators might need to change.  Provide commandline and environment variables overrides for configuration parameters.  Order of priority should be 1) commandline 2) environment 3) config file 4) hard coded or default values.
- **Minimal dependencies**: prefer stdlib or a small focused library over pulling in a large framework; single-file solutions are fine for simple utilities
- **Infrastructure first**: prioritize correctness of the underlying system over UI polish
- **No linting configured by default** — don't add linting config unless explicitly asked
- Add stand alone helper utilities for diagnostics and debugging.  Each helper should have a manpage for how to use it.
- Don't add abstractions or refactoring beyond what the task requires

## CLAUDE.md for New Projects

When initializing a new project's CLAUDE.md, use this structure:
1. Project Overview (one paragraph: what it does and why)
2. Setup and Running (exact commands to install deps and launch)
3. Architecture (modules, threading model, data flow)
4. Configuration Schema (YAML example with comments)
5. Testing (how to run tests, or note if none are configured)

## Domain Context

Many projects fall into two domains:

**Mu2e / NOvA DAQ systems** — real-time monitoring and control software for particle physics experiments at Fermilab. These are production infrastructure tools; correctness and reliability matter more than convenience.

**Personal and Game related tools** — game utilities including raid loot management, log parsing, web UIs, spawn timers, map tools, and DKP tracking. These are personal/community tools; pragmatism over perfection is fine.
