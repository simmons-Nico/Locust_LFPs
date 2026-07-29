from __future__ import annotations

import calendar
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
import tkinter.font as tkfont

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


APP_NAME = "Daily Notes Library"
APP_DIR = Path.home() / "Documents" / APP_NAME
NOTES_DIR_NAME = "notes"
SCRIPT_DIR = Path(__file__).resolve().parent
ICON_PATH = SCRIPT_DIR / "daily_notes_icon.ico"
LEGACY_NOTE_ID = "main"
MONTH_NAMES = list(calendar.month_name)[1:]
RICH_TEXT_TAGS = {"bold", "italic", "underline", "mono", "heading", "table"}
CHECKBOX_EMPTY = chr(0x2610)
CHECKBOX_CHECKED = chr(0x2611)
CHECKLIST_MARK_RE = re.compile(r"[\u2610\u2611]|\[[ xX]\]")
CHECKLIST_LINE_RE = re.compile(r"^(\s*)(?:[-*]\s*)?([\u2610\u2611]|\[[ xX]\])\s+")
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def is_separator_row(cells: list[str]) -> bool:
    clean_cells = [cell.replace(" ", "") for cell in cells if cell.strip()]
    return bool(clean_cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) for cell in clean_cells
    )


def split_table_line(line: str, allow_single_space: bool = False) -> list[str] | None:
    stripped = line.strip()
    if not stripped:
        return None

    if stripped.count("|") >= 2:
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    elif "\t" in stripped:
        cells = [cell.strip() for cell in stripped.split("\t")]
    elif stripped.count(",") >= 1:
        cells = [cell.strip() for cell in stripped.split(",")]
    elif stripped.count(";") >= 1:
        cells = [cell.strip() for cell in stripped.split(";")]
    elif re.search(r"\S\s{2,}\S", stripped):
        cells = [cell.strip() for cell in re.split(r"\s{2,}", stripped)]
    elif allow_single_space:
        cells = [cell.strip() for cell in re.split(r"\s+", stripped)]
    else:
        return None

    meaningful_cells = [cell for cell in cells if cell]
    if len(meaningful_cells) < 2:
        return None
    return cells


def normalize_table_rows(
    lines: list[str],
    allow_single_space: bool = False,
) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        cells = split_table_line(line, allow_single_space=allow_single_space)
        if cells is None or is_separator_row(cells):
            continue
        rows.append(cells)

    if not rows:
        return []

    column_count = max(len(row) for row in rows)
    return [row + [""] * (column_count - len(row)) for row in rows]


def table_rows_to_text(rows: list[list[str]]) -> str:
    return "\n".join("\t".join(cell.strip() for cell in row) for row in rows)

COLORS = {
    "bg": "#edf3f8",
    "panel": "#f9fcff",
    "sidebar": "#e4edf5",
    "surface": "#f1f6fb",
    "border": "#c4d0dc",
    "text": "#172033",
    "muted": "#66758a",
    "button": "#edf4fb",
    "button_hover": "#e0ebf6",
    "button_pressed": "#cfdeec",
    "button_shadow": "#b7c5d5",
    "button_light": "#ffffff",
    "accent": "#2f66d9",
    "accent_dark": "#244fb4",
    "soft_accent": "#dce8ff",
    "format": "#405a9c",
    "format_dark": "#31487f",
    "format_soft": "#e7ecff",
    "attach": "#0e837c",
    "attach_dark": "#0a6964",
    "attach_soft": "#d8f3ef",
    "table": "#2f6f9d",
    "table_dark": "#255b80",
    "table_soft": "#e2f0fb",
    "table_header": "#d7e8f7",
    "table_alt": "#f3f8fd",
    "teal": "#0e837c",
    "soft_teal": "#d8f3ef",
    "danger": "#b4233a",
    "soft_danger": "#ffe4e8",
}


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def today() -> date:
    return date.today()


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def format_date_long(day: date) -> str:
    return day.strftime("%A, %B %d, %Y")


def format_month_label(year: int, month: int) -> str:
    return f"{calendar.month_name[month]} {year}"


def safe_filename(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip()
    clean = re.sub(r"\s+", " ", clean)
    return clean or "attachment"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def open_with_default_app(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))

    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def powershell_single_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@dataclass(frozen=True)
class SearchMatch:
    day: date
    note_id: str
    title: str
    preview: str


class NotesStore:
    def __init__(self, root: Path = APP_DIR) -> None:
        self.root = root
        self.notes_dir = root / NOTES_DIR_NAME
        self.ensure()

    def ensure(self) -> None:
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def day_folder(self, day: date) -> Path:
        return self.notes_dir / f"{day.year:04d}" / f"{day.month:02d}"

    def new_note_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def note_path(self, day: date, note_id: str = LEGACY_NOTE_ID) -> Path:
        if note_id == LEGACY_NOTE_ID:
            return self.day_folder(day) / f"{day.isoformat()}.json"
        return self.day_folder(day) / f"{day.isoformat()}__{safe_filename(note_id)}.json"

    def asset_dir(self, day: date, note_id: str = LEGACY_NOTE_ID) -> Path:
        if note_id == LEGACY_NOTE_ID:
            return self.day_folder(day) / f"{day.isoformat()}_files"
        return self.day_folder(day) / f"{day.isoformat()}__{safe_filename(note_id)}_files"

    def note_id_from_path(self, path: Path) -> tuple[date, str] | None:
        stem = path.stem
        if "__" in stem:
            date_part, note_id = stem.split("__", 1)
        else:
            date_part, note_id = stem, LEGACY_NOTE_ID
        try:
            return parse_date(date_part), note_id
        except ValueError:
            return None

    def empty_note(self, day: date, note_id: str | None = None) -> dict:
        stamp = now_iso()
        return {
            "id": note_id or self.new_note_id(),
            "date": day.isoformat(),
            "title": "",
            "body": "",
            "content_ops": [],
            "attachments": [],
            "created_at": stamp,
            "updated_at": stamp,
        }

    def load(self, day: date, note_id: str | None = None) -> dict:
        if note_id is None:
            note_id = self.default_note_id(day) or self.new_note_id()
        path = self.note_path(day, note_id)
        if not path.exists():
            return self.empty_note(day, note_id)

        try:
            with path.open("r", encoding="utf-8") as handle:
                note = json.load(handle)
        except (json.JSONDecodeError, OSError):
            backup = path.with_suffix(".broken.json")
            try:
                shutil.copy2(path, unique_path(backup))
            except OSError:
                pass
            return self.empty_note(day, note_id)

        note.setdefault("id", note_id)
        note.setdefault("date", day.isoformat())
        note.setdefault("title", "")
        note.setdefault("body", "")
        note.setdefault("content_ops", [])
        note.setdefault("attachments", [])
        note.setdefault("created_at", now_iso())
        note.setdefault("updated_at", now_iso())
        return note

    def save(self, note: dict) -> None:
        day = parse_date(note["date"])
        note_id = note.setdefault("id", self.new_note_id())
        folder = self.day_folder(day)
        folder.mkdir(parents=True, exist_ok=True)
        note["updated_at"] = now_iso()

        destination = self.note_path(day, note_id)
        temp_path = destination.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(note, handle, indent=2, ensure_ascii=False)
        temp_path.replace(destination)

    def delete_note(self, day: date, note_id: str) -> None:
        note_file = self.note_path(day, note_id)
        files_dir = self.asset_dir(day, note_id)
        if note_file.exists():
            note_file.unlink()
        if files_dir.exists() and files_dir.resolve().is_relative_to(self.notes_dir.resolve()):
            shutil.rmtree(files_dir)

    def copied_attachment_path(self, item: dict) -> Path:
        return self.root / item.get("path", "")

    def attachment_path(self, item: dict) -> Path:
        if item.get("mode") == "copied":
            return self.copied_attachment_path(item)
        return Path(item.get("path", ""))

    def all_note_dates(self) -> set[date]:
        dates: set[date] = set()
        if not self.notes_dir.exists():
            return dates

        for path in self.notes_dir.glob("*/*/*.json"):
            parsed = self.note_id_from_path(path)
            if parsed:
                dates.add(parsed[0])
        return dates

    def note_ids_for_day(self, day: date) -> list[str]:
        ids: list[str] = []
        folder = self.day_folder(day)
        if not folder.exists():
            return ids
        for path in folder.glob(f"{day.isoformat()}*.json"):
            parsed = self.note_id_from_path(path)
            if parsed and parsed[0] == day:
                ids.append(parsed[1])

        def sort_key(note_id: str) -> tuple[str, str]:
            note = self.load(day, note_id)
            return note.get("created_at", ""), note_id

        return sorted(set(ids), key=sort_key)

    def default_note_id(self, day: date) -> str | None:
        ids = self.note_ids_for_day(day)
        if not ids:
            return None
        notes = [(self.load(day, note_id), note_id) for note_id in ids]
        notes.sort(key=lambda pair: pair[0].get("updated_at", ""), reverse=True)
        return notes[0][1]

    def all_notes(self) -> list[tuple[date, str, dict]]:
        notes: list[tuple[date, str, dict]] = []
        for day in sorted(self.all_note_dates(), reverse=True):
            for note_id in reversed(self.note_ids_for_day(day)):
                notes.append((day, note_id, self.load(day, note_id)))
        return notes

    def search(self, query: str, limit: int = 100) -> list[SearchMatch]:
        words = [word.lower() for word in query.split() if word.strip()]
        if not words:
            return []

        matches: list[SearchMatch] = []
        for day, note_id, note in self.all_notes():
            attachment_text = " ".join(
                attachment.get("name", "") for attachment in note.get("attachments", [])
            )
            haystack = " ".join(
                [
                    note.get("title", ""),
                    note.get("body", ""),
                    attachment_text,
                    day.isoformat(),
                    format_date_long(day),
                ]
            ).lower()
            if not all(word in haystack for word in words):
                continue

            body_preview = " ".join(note.get("body", "").split())
            if len(body_preview) > 90:
                body_preview = body_preview[:87] + "..."
            matches.append(
                SearchMatch(
                    day=day,
                    note_id=note_id,
                    title=note.get("title", "").strip() or "Untitled note",
                    preview=body_preview,
                )
            )
            if len(matches) >= limit:
                break
        return matches


class DailyNotesApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.store = NotesStore()
        self.current_date = today()
        self.current_note = self.store.load(self.current_date)
        self.current_note_id = self.current_note["id"]
        self.note_dates = self.store.all_note_dates()
        self.dirty = False
        self.loading = False
        self.autosave_job: str | None = None
        self.search_job: str | None = None
        self.table_style_job: str | None = None
        self.thumbnail_refs: list[ImageTk.PhotoImage] = [] if ImageTk else []
        self.inline_image_refs: dict[str, ImageTk.PhotoImage] = {} if ImageTk else {}
        self.inline_image_name_to_attachment: dict[str, str] = {}
        self.table_widgets: dict[str, dict[str, object]] = {}
        self.table_window_to_id: dict[str, str] = {}
        self.selected_attachment_id: str | None = None
        self.search_results: list[SearchMatch] = []

        self.title_var = tk.StringVar()
        self.status_var = tk.StringVar(value=f"Notes folder: {self.store.root}")
        self.search_var = tk.StringVar()
        self.library_filter_var = tk.StringVar(value="Recent")
        self.month_name_var = tk.StringVar()
        self.year_var = tk.IntVar(value=self.current_date.year)

        self.title(APP_NAME)
        self.geometry("1240x780")
        self.minsize(1040, 680)
        self.configure(bg=COLORS["bg"])
        if ICON_PATH.exists():
            try:
                self.iconbitmap(default=str(ICON_PATH))
            except tk.TclError:
                pass

        self.configure_styles()
        self.build_menu()
        self.build_layout()
        self.bind_shortcuts()
        self.load_date(self.current_date, first_load=True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = ("Segoe UI", 10)
        style.configure(".", font=base_font, foreground=COLORS["text"])
        style.configure("App.TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Sidebar.TFrame", background=COLORS["sidebar"])
        style.configure("Muted.TLabel", foreground=COLORS["muted"], background=COLORS["panel"])
        style.configure("Sidebar.TLabel", background=COLORS["sidebar"], foreground=COLORS["text"])
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure(
            "TEntry",
            fieldbackground=COLORS["panel"],
            bordercolor=COLORS["button_shadow"],
            lightcolor=COLORS["button_light"],
            darkcolor=COLORS["button_shadow"],
            relief="sunken",
            borderwidth=2,
        )
        style.configure(
            "TCombobox",
            fieldbackground=COLORS["panel"],
            background=COLORS["button"],
            bordercolor=COLORS["button_shadow"],
            lightcolor=COLORS["button_light"],
            darkcolor=COLORS["button_shadow"],
            relief="sunken",
            borderwidth=2,
        )
        style.configure(
            "TButton",
            foreground=COLORS["text"],
            background=COLORS["button"],
            bordercolor=COLORS["button_shadow"],
            lightcolor=COLORS["button_light"],
            darkcolor=COLORS["button_shadow"],
            relief="raised",
            borderwidth=2,
            padding=(8, 5),
        )
        style.map(
            "TButton",
            background=[
                ("pressed", COLORS["button_pressed"]),
                ("active", COLORS["button_hover"]),
            ],
            relief=[("pressed", "sunken"), ("active", "raised")],
        )
        style.configure(
            "Title.TLabel",
            background=COLORS["sidebar"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 15),
        )
        style.configure(
            "Date.TLabel",
            background=COLORS["bg"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 17),
        )
        style.configure(
            "Section.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Accent.TButton",
            foreground="#ffffff",
            background=COLORS["accent"],
            bordercolor=COLORS["accent_dark"],
            lightcolor="#7ea5ff",
            darkcolor=COLORS["accent_dark"],
            relief="raised",
            borderwidth=2,
            padding=(8, 5),
        )
        style.map(
            "Accent.TButton",
            background=[("active", COLORS["accent_dark"]), ("pressed", COLORS["accent_dark"])],
            relief=[("pressed", "sunken"), ("active", "raised")],
        )
        style.configure(
            "Format.TButton",
            foreground=COLORS["format_dark"],
            background=COLORS["format_soft"],
            bordercolor="#b8c4ee",
            lightcolor="#ffffff",
            darkcolor="#b8c4ee",
        )
        style.map(
            "Format.TButton",
            background=[("active", "#d8e1ff"), ("pressed", "#c3cff7")],
            relief=[("pressed", "sunken"), ("active", "raised")],
        )
        style.configure(
            "Attach.TButton",
            foreground=COLORS["attach_dark"],
            background=COLORS["attach_soft"],
            bordercolor="#9acfc8",
            lightcolor="#ffffff",
            darkcolor="#9acfc8",
        )
        style.map(
            "Attach.TButton",
            background=[("active", "#c8ebe6"), ("pressed", "#aadbd4")],
            relief=[("pressed", "sunken"), ("active", "raised")],
        )
        style.configure(
            "Danger.TButton",
            foreground=COLORS["danger"],
            background=COLORS["soft_danger"],
            bordercolor="#f2a8b5",
            lightcolor="#ffffff",
            darkcolor="#f2a8b5",
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#ffd5dc"), ("pressed", "#f8bac7")],
            relief=[("pressed", "sunken"), ("active", "raised")],
        )
        style.configure("Treeview", rowheight=26)
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))

    def tk_button_options(self, role: str = "neutral") -> dict:
        palettes = {
            "accent": ("#ffffff", COLORS["accent"], COLORS["accent_dark"], COLORS["accent_dark"]),
            "format": (COLORS["format_dark"], COLORS["format_soft"], "#d8e1ff", "#b8c4ee"),
            "attach": (COLORS["attach_dark"], COLORS["attach_soft"], "#c8ebe6", "#9acfc8"),
            "table": (COLORS["table_dark"], COLORS["table_soft"], "#d3e8f8", "#9ec6e3"),
            "danger": (COLORS["danger"], COLORS["soft_danger"], "#ffd5dc", "#f2a8b5"),
            "sidebar": (COLORS["text"], COLORS["button"], COLORS["button_hover"], COLORS["button_shadow"]),
            "neutral": (COLORS["text"], COLORS["button"], COLORS["button_hover"], COLORS["button_shadow"]),
        }
        fg, bg, active_bg, border = palettes.get(role, palettes["neutral"])
        return {
            "fg": fg,
            "bg": bg,
            "activeforeground": fg,
            "activebackground": active_bg,
            "relief": "raised",
            "bd": 2,
            "highlightthickness": 1,
            "highlightbackground": border,
            "highlightcolor": border,
            "cursor": "hand2",
        }

    def make_tk_button(
        self,
        parent: tk.Misc,
        text: str,
        command: object,
        role: str = "neutral",
        **kwargs: object,
    ) -> tk.Button:
        options = self.tk_button_options(role)
        options.update(kwargs)
        return tk.Button(parent, text=text, command=command, **options)

    def build_menu(self) -> None:
        menu_bar = tk.Menu(self)

        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="Save", command=self.save_current_note, accelerator="Ctrl+S")
        file_menu.add_command(label="Open notes folder", command=self.open_notes_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menu_bar.add_cascade(label="File", menu=file_menu)

        note_menu = tk.Menu(menu_bar, tearoff=False)
        note_menu.add_command(label="Insert time stamp", command=self.insert_time_stamp)
        note_menu.add_command(label="Insert check box", command=self.insert_checklist_item)
        note_menu.add_command(label="Insert table", command=self.insert_table)
        note_menu.add_command(label="Make table from text", command=self.apply_table_interpretation)
        note_menu.add_separator()
        note_menu.add_command(label="Delete current note", command=self.delete_current_note)
        menu_bar.add_cascade(label="Note", menu=note_menu)

        navigate_menu = tk.Menu(menu_bar, tearoff=False)
        navigate_menu.add_command(label="Today", command=lambda: self.load_date(today()))
        navigate_menu.add_command(label="Previous day", command=lambda: self.shift_day(-1))
        navigate_menu.add_command(label="Next day", command=lambda: self.shift_day(1))
        menu_bar.add_cascade(label="Navigate", menu=navigate_menu)

        self.config(menu=menu_bar)

    def build_layout(self) -> None:
        self.main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(self.main_pane, style="Sidebar.TFrame", width=320)
        self.editor_panel = ttk.Frame(self.main_pane, style="App.TFrame")
        self.attachment_panel = ttk.Frame(self.main_pane, style="Panel.TFrame", width=310)

        self.main_pane.add(self.sidebar, weight=0)
        self.main_pane.add(self.editor_panel, weight=1)
        self.main_pane.add(self.attachment_panel, weight=0)

        self.build_sidebar()
        self.build_editor()
        self.build_attachments_panel()

    def build_sidebar(self) -> None:
        self.sidebar.columnconfigure(0, weight=1)

        title = ttk.Label(self.sidebar, text=APP_NAME, style="Title.TLabel")
        title.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 4))
        subtitle = ttk.Label(
            self.sidebar,
            text="Browse by year, month, and day",
            style="Sidebar.TLabel",
        )
        subtitle.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))

        search_label = ttk.Label(self.sidebar, text="Search", style="Sidebar.TLabel")
        search_label.grid(row=2, column=0, sticky="w", padx=16, pady=(0, 4))

        self.search_entry = ttk.Entry(self.sidebar, textvariable=self.search_var)
        self.search_entry.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.search_var.trace_add("write", self.on_search_changed)

        self.search_list = tk.Listbox(
            self.sidebar,
            height=4,
            activestyle="none",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            selectbackground=COLORS["accent"],
            selectforeground="#ffffff",
            font=("Segoe UI", 9),
        )
        self.search_list.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 14))
        self.search_list.bind("<Double-Button-1>", self.on_search_open)
        self.search_list.bind("<Return>", self.on_search_open)

        month_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        month_frame.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 8))
        month_frame.columnconfigure(1, weight=1)

        ttk.Button(month_frame, text="<", width=3, command=lambda: self.shift_month(-1)).grid(
            row=0, column=0, sticky="w"
        )
        self.month_combo = ttk.Combobox(
            month_frame,
            textvariable=self.month_name_var,
            values=MONTH_NAMES,
            state="readonly",
            width=12,
        )
        self.month_combo.grid(row=0, column=1, sticky="ew", padx=6)
        self.month_combo.bind("<<ComboboxSelected>>", lambda event: self.on_calendar_month_changed())

        self.year_spin = ttk.Spinbox(
            month_frame,
            from_=1900,
            to=2200,
            textvariable=self.year_var,
            width=6,
            command=self.on_calendar_month_changed,
        )
        self.year_spin.grid(row=0, column=2, sticky="e", padx=(0, 6))
        self.year_spin.bind("<Return>", lambda event: self.on_calendar_month_changed())
        self.year_spin.bind("<FocusOut>", lambda event: self.on_calendar_month_changed())
        ttk.Button(month_frame, text=">", width=3, command=lambda: self.shift_month(1)).grid(
            row=0, column=3, sticky="e"
        )

        self.calendar_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.calendar_frame.grid(row=6, column=0, sticky="ew", padx=16, pady=(0, 14))
        for col in range(7):
            self.calendar_frame.columnconfigure(col, weight=1, uniform="day")
        self.day_buttons: list[tk.Button] = []
        for col, name in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
            label = ttk.Label(
                self.calendar_frame,
                text=name,
                anchor="center",
                style="Sidebar.TLabel",
                font=("Segoe UI Semibold", 8),
            )
            label.grid(row=0, column=col, sticky="ew", pady=(0, 4))

        for row in range(6):
            for col in range(7):
                button = self.make_tk_button(
                    self.calendar_frame,
                    text="",
                    command=lambda: None,
                    role="sidebar",
                    width=4,
                    height=1,
                    font=("Segoe UI", 9),
                )
                button.grid(row=row + 1, column=col, sticky="nsew", padx=1, pady=1)
                self.day_buttons.append(button)

        today_button = ttk.Button(
            self.sidebar,
            text="Today",
            style="Accent.TButton",
            command=lambda: self.load_date(today()),
        )
        today_button.grid(row=7, column=0, sticky="ew", padx=16, pady=(0, 14))

        library_header = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        library_header.grid(row=8, column=0, sticky="ew", padx=16, pady=(0, 6))
        library_header.columnconfigure(0, weight=1)
        ttk.Label(library_header, text="Library", style="Sidebar.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.library_count_label = ttk.Label(library_header, text="", style="Sidebar.TLabel")
        self.library_count_label.grid(row=0, column=1, sticky="e", padx=(8, 0))

        library_filter_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        library_filter_frame.grid(row=9, column=0, sticky="ew", padx=16, pady=(0, 8))
        library_filter_frame.columnconfigure(0, weight=1)
        self.library_filter = ttk.Combobox(
            library_filter_frame,
            textvariable=self.library_filter_var,
            values=["Recent", "Shown month", "Shown year", "All notes"],
            state="readonly",
            width=14,
        )
        self.library_filter.grid(row=0, column=0, sticky="ew")
        self.library_filter.bind("<<ComboboxSelected>>", lambda event: self.refresh_library_tree())

        library_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        library_frame.grid(row=10, column=0, sticky="nsew", padx=0, pady=(0, 16))
        library_frame.columnconfigure(0, weight=1)
        library_frame.rowconfigure(0, weight=1)
        self.sidebar.rowconfigure(10, weight=1)

        self.library_canvas = tk.Canvas(
            library_frame,
            bg=COLORS["sidebar"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.library_canvas.grid(row=0, column=0, sticky="nsew")
        library_scroll = ttk.Scrollbar(
            library_frame,
            orient=tk.VERTICAL,
            command=self.library_canvas.yview,
        )
        library_scroll.grid(row=0, column=1, sticky="ns")
        self.library_canvas.configure(yscrollcommand=library_scroll.set)
        self.library_inner = ttk.Frame(self.library_canvas, style="Sidebar.TFrame")
        self.library_window = self.library_canvas.create_window(
            (0, 0), window=self.library_inner, anchor="nw"
        )
        self.library_inner.bind(
            "<Configure>",
            lambda event: self.library_canvas.configure(
                scrollregion=self.library_canvas.bbox("all")
            ),
        )
        self.library_canvas.bind(
            "<Configure>",
            lambda event: self.library_canvas.itemconfigure(
                self.library_window, width=event.width
            ),
        )
        self.library_canvas.bind("<MouseWheel>", self.on_library_mousewheel)

    def build_editor(self) -> None:
        self.editor_panel.columnconfigure(0, weight=1)
        self.editor_panel.rowconfigure(3, weight=1)

        header = ttk.Frame(self.editor_panel, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 8))
        header.columnconfigure(1, weight=1)

        ttk.Button(header, text="<", width=3, command=lambda: self.shift_day(-1)).grid(
            row=0, column=0, sticky="w"
        )
        self.date_label = ttk.Label(header, text="", style="Date.TLabel")
        self.date_label.grid(row=0, column=1, sticky="w", padx=12)
        ttk.Button(header, text=">", width=3, command=lambda: self.shift_day(1)).grid(
            row=0, column=2, sticky="e", padx=(0, 8)
        )
        ttk.Button(header, text="Save", style="Accent.TButton", command=self.save_current_note).grid(
            row=0, column=3, sticky="e"
        )

        title_frame = ttk.Frame(self.editor_panel, style="App.TFrame")
        title_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))
        title_frame.columnconfigure(1, weight=1)
        ttk.Label(title_frame, text="Title", background=COLORS["bg"]).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.title_entry = ttk.Entry(title_frame, textvariable=self.title_var, font=("Segoe UI", 11))
        self.title_entry.grid(row=0, column=1, sticky="ew")
        self.title_var.trace_add("write", lambda *_: self.mark_dirty())

        toolbar = ttk.Frame(self.editor_panel, style="App.TFrame")
        toolbar.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 8))
        self.make_tk_button(
            toolbar,
            text="B",
            command=lambda: self.toggle_text_tag("bold"),
            role="format",
            width=3,
            font=("Segoe UI Semibold", 10),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.make_tk_button(
            toolbar,
            text="I",
            command=lambda: self.toggle_text_tag("italic"),
            role="format",
            width=3,
            font=("Segoe UI Italic", 10),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.make_tk_button(
            toolbar,
            text="U",
            command=lambda: self.toggle_text_tag("underline"),
            role="format",
            width=3,
            font=("Segoe UI", 10, "underline"),
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(toolbar, text="Mono", style="Format.TButton", command=lambda: self.toggle_text_tag("mono")).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(toolbar, text="Heading", style="Format.TButton", command=lambda: self.toggle_text_tag("heading")).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(toolbar, text="Check box", style="Format.TButton", command=self.insert_checklist_item).pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(toolbar, text="Table", style="Format.TButton", command=self.insert_table).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(toolbar, text="Make table", style="Format.TButton", command=self.apply_table_interpretation).pack(
            side=tk.LEFT, padx=(0, 10)
        )
        ttk.Button(toolbar, text="Insert time", style="Format.TButton", command=self.insert_time_stamp).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(toolbar, text="Delete note", style="Danger.TButton", command=self.delete_current_note).pack(
            side=tk.RIGHT
        )

        editor_frame = ttk.Frame(self.editor_panel, style="App.TFrame")
        editor_frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 8))
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(0, weight=1)

        self.body_text = tk.Text(
            editor_frame,
            wrap="word",
            undo=True,
            borderwidth=2,
            relief="sunken",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent"],
            padx=14,
            pady=12,
            font=("Segoe UI", 11),
            bg=COLORS["panel"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
        )
        self.body_text.grid(row=0, column=0, sticky="nsew")
        text_scroll = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self.body_text.yview)
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.body_text.configure(yscrollcommand=text_scroll.set)
        self.configure_rich_text_tags()
        self.body_text.bind("<<Modified>>", self.on_body_modified)
        self.body_text.bind("<Return>", self.on_body_return)

        self.status_label = ttk.Label(
            self.editor_panel,
            textvariable=self.status_var,
            anchor="w",
            background=COLORS["bg"],
            foreground=COLORS["muted"],
        )
        self.status_label.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 12))

    def configure_rich_text_tags(self) -> None:
        self.rich_fonts = {
            "body": tkfont.Font(family="Segoe UI", size=11),
            "bold": tkfont.Font(family="Segoe UI", size=11, weight="bold"),
            "italic": tkfont.Font(family="Segoe UI", size=11, slant="italic"),
            "underline": tkfont.Font(family="Segoe UI", size=11, underline=True),
            "mono": tkfont.Font(family="Consolas", size=10),
            "heading": tkfont.Font(family="Segoe UI", size=15, weight="bold"),
        }
        self.body_text.tag_configure("bold", font=self.rich_fonts["bold"])
        self.body_text.tag_configure("italic", font=self.rich_fonts["italic"])
        self.body_text.tag_configure("underline", font=self.rich_fonts["underline"])
        self.body_text.tag_configure("mono", font=self.rich_fonts["mono"])
        self.body_text.tag_configure(
            "heading",
            font=self.rich_fonts["heading"],
            spacing1=8,
            spacing3=6,
        )
        self.body_text.tag_configure(
            "table",
            font=self.rich_fonts["body"],
            background="#f1f5f9",
            foreground=COLORS["text"],
            borderwidth=1,
            relief="solid",
            lmargin1=10,
            lmargin2=10,
            rmargin=10,
            spacing1=4,
            spacing3=4,
            tabs=(150, 300, 450, 600, 750),
        )
        self.body_text.tag_configure(
            "checklist_box",
            foreground=COLORS["accent"],
            font=self.rich_fonts["bold"],
        )
        self.body_text.tag_bind("checklist_box", "<Button-1>", self.on_checklist_click)
        self.body_text.tag_bind(
            "checklist_box",
            "<Enter>",
            lambda _event: self.body_text.configure(cursor="hand2"),
        )
        self.body_text.tag_bind(
            "checklist_box",
            "<Leave>",
            lambda _event: self.body_text.configure(cursor="xterm"),
        )
        self.body_text.tag_raise("sel")

    def build_attachments_panel(self) -> None:
        self.attachment_panel.columnconfigure(0, weight=1)
        self.attachment_panel.rowconfigure(3, weight=1)

        header = ttk.Frame(self.attachment_panel, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(16, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Attachments", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.attachment_count = ttk.Label(header, text="", style="Muted.TLabel")
        self.attachment_count.grid(row=0, column=1, sticky="e")

        note_actions = ttk.Frame(self.attachment_panel, style="Panel.TFrame")
        note_actions.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        note_actions.columnconfigure(0, weight=1)
        ttk.Button(
            note_actions,
            text="Add note",
            style="Accent.TButton",
            command=self.add_note_for_current_day,
        ).grid(row=0, column=0, sticky="ew")

        button_grid = ttk.Frame(self.attachment_panel, style="Panel.TFrame")
        button_grid.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        for col in range(2):
            button_grid.columnconfigure(col, weight=1)
        ttk.Button(
            button_grid,
            text="Add image",
            style="Attach.TButton",
            command=self.add_image,
        ).grid(
            row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6)
        )
        ttk.Button(
            button_grid,
            text="Attach copy",
            style="Attach.TButton",
            command=self.attach_file_copy,
        ).grid(
            row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 6)
        )
        ttk.Button(
            button_grid,
            text="Link file",
            style="Attach.TButton",
            command=self.link_file,
        ).grid(
            row=1, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(
            button_grid,
            text="Open file",
            style="Attach.TButton",
            command=self.open_selected_attachment,
        ).grid(
            row=1, column=1, sticky="ew", padx=(4, 0)
        )

        list_frame = ttk.Frame(self.attachment_panel, style="Panel.TFrame")
        list_frame.grid(row=3, column=0, sticky="nsew", padx=0, pady=0)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.attach_canvas = tk.Canvas(
            list_frame,
            bg=COLORS["panel"],
            highlightthickness=0,
            borderwidth=0,
        )
        self.attach_canvas.grid(row=0, column=0, sticky="nsew")
        attach_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.attach_canvas.yview)
        attach_scroll.grid(row=0, column=1, sticky="ns")
        self.attach_canvas.configure(yscrollcommand=attach_scroll.set)

        self.attach_inner = ttk.Frame(self.attach_canvas, style="Panel.TFrame")
        self.attach_window = self.attach_canvas.create_window(
            (0, 0), window=self.attach_inner, anchor="nw"
        )
        self.attach_inner.bind(
            "<Configure>",
            lambda event: self.attach_canvas.configure(scrollregion=self.attach_canvas.bbox("all")),
        )
        self.attach_canvas.bind(
            "<Configure>",
            lambda event: self.attach_canvas.itemconfigure(self.attach_window, width=event.width),
        )

        footer = ttk.Frame(self.attachment_panel, style="Panel.TFrame")
        footer.grid(row=4, column=0, sticky="ew", padx=14, pady=(8, 14))
        footer.columnconfigure(0, weight=1)
        ttk.Button(footer, text="Remove selected", style="Danger.TButton", command=self.remove_selected_attachment).grid(
            row=0, column=0, sticky="ew"
        )

    def bind_shortcuts(self) -> None:
        self.bind("<Control-s>", lambda event: (self.save_current_note(), "break")[1])
        self.bind("<Control-f>", lambda event: (self.focus_search(), "break")[1])
        self.bind("<Control-b>", lambda event: (self.toggle_text_tag("bold"), "break")[1])
        self.bind("<Control-i>", lambda event: (self.toggle_text_tag("italic"), "break")[1])
        self.bind("<Control-u>", lambda event: (self.toggle_text_tag("underline"), "break")[1])
        self.bind("<Alt-Left>", lambda event: (self.shift_day(-1), "break")[1])
        self.bind("<Alt-Right>", lambda event: (self.shift_day(1), "break")[1])
        self.bind("<Control-t>", lambda event: (self.insert_time_stamp(), "break")[1])
        self.bind("<Control-Shift-C>", lambda event: (self.insert_checklist_item(), "break")[1])

    def focus_search(self) -> None:
        self.search_entry.focus_set()
        self.search_entry.selection_range(0, tk.END)

    def on_body_modified(self, _event: tk.Event) -> None:
        if not self.body_text.edit_modified():
            return
        self.body_text.edit_modified(False)
        self.mark_dirty()
        if self.table_style_job is not None:
            self.after_cancel(self.table_style_job)
        self.table_style_job = self.after(450, self.refresh_editor_tags)

    def selected_range_or_line(self, tag: str) -> tuple[str, str] | None:
        try:
            return self.body_text.index("sel.first"), self.body_text.index("sel.last")
        except tk.TclError:
            if tag in {"heading", "table"}:
                return self.body_text.index("insert linestart"), self.body_text.index("insert lineend")

            start = self.body_text.index("insert wordstart")
            end = self.body_text.index("insert wordend")
            if start != end:
                return start, end
        return None

    def toggle_text_tag(self, tag: str) -> None:
        range_pair = self.selected_range_or_line(tag)
        if range_pair is None:
            self.status_var.set("Select text first, then choose a format.")
            return

        start, end = range_pair
        if tag in self.body_text.tag_names(start):
            self.body_text.tag_remove(tag, start, end)
        else:
            self.body_text.tag_add(tag, start, end)
        self.mark_dirty()

    def insert_checklist_item(self) -> None:
        try:
            start = self.body_text.index("sel.first")
            end = self.body_text.index("sel.last")
        except tk.TclError:
            start = ""
            end = ""

        if start and end:
            start_line = int(start.split(".")[0])
            end_line = int(self.body_text.index(f"{end} - 1c").split(".")[0])
            for line_number in range(end_line, start_line - 1, -1):
                line_start = f"{line_number}.0"
                line_text = self.body_text.get(line_start, f"{line_number}.end")
                if CHECKLIST_LINE_RE.match(line_text):
                    continue
                self.body_text.insert(line_start, f"{CHECKBOX_EMPTY} ")
        else:
            insert_at = self.body_text.index(tk.INSERT)
            if insert_at != "1.0" and self.body_text.get(f"{insert_at} - 1c") != "\n":
                self.body_text.insert(tk.INSERT, "\n")
            self.body_text.insert(tk.INSERT, CHECKBOX_EMPTY, ("checklist_box",))
            self.body_text.insert(tk.INSERT, " ")

        self.refresh_checklist_tags()
        self.body_text.focus_set()
        self.mark_dirty()

    def checklist_range_at_index(self, index: str) -> tuple[str, str] | None:
        ranges = self.body_text.tag_ranges("checklist_box")
        for range_index in range(0, len(ranges), 2):
            start = str(ranges[range_index])
            end = str(ranges[range_index + 1])
            if self.body_text.compare(start, "<=", index) and self.body_text.compare(index, "<", end):
                return start, end
        return None

    def on_checklist_click(self, event: tk.Event) -> str | None:
        index = self.body_text.index(f"@{event.x},{event.y}")
        range_pair = self.checklist_range_at_index(index)
        if range_pair is None:
            return None

        start, end = range_pair
        current = self.body_text.get(start, end)
        if current == CHECKBOX_EMPTY:
            replacement = CHECKBOX_CHECKED
        elif current == CHECKBOX_CHECKED:
            replacement = CHECKBOX_EMPTY
        elif current.lower() == "[x]":
            replacement = "[ ]"
        else:
            replacement = "[x]"

        self.body_text.delete(start, end)
        self.body_text.insert(start, replacement, ("checklist_box",))
        self.body_text.mark_set(tk.INSERT, f"{start} + {len(replacement)} chars")
        self.mark_dirty()
        return "break"

    def on_body_return(self, _event: tk.Event) -> str | None:
        if not self.body_text.compare(tk.INSERT, "==", "insert lineend"):
            return None

        line_text = self.body_text.get("insert linestart", "insert lineend")
        match = CHECKLIST_LINE_RE.match(line_text)
        if match is None:
            return None

        marker = match.group(2)
        next_marker = "[ ]" if marker.startswith("[") else CHECKBOX_EMPTY
        self.body_text.insert(tk.INSERT, "\n" + match.group(1))
        self.body_text.insert(tk.INSERT, next_marker, ("checklist_box",))
        self.body_text.insert(tk.INSERT, " ")
        self.refresh_checklist_tags()
        self.mark_dirty()
        return "break"

    def insert_table(self) -> None:
        rows = [
            ["Column 1", "Column 2", "Column 3"],
            ["", "", ""],
            ["", "", ""],
        ]
        insert_at = self.body_text.index(tk.INSERT)
        if insert_at != "1.0" and self.body_text.get(f"{insert_at} - 1c") != "\n":
            self.body_text.insert(tk.INSERT, "\n")
        table_id = self.insert_table_widget(rows)
        self.body_text.insert(tk.INSERT, "\n")
        self.after_idle(lambda current_id=table_id: self.focus_table_cell(current_id, 0, 0))
        self.mark_dirty()

    def create_table_editor(self, rows: list[list[str]], table_id: str) -> tk.Frame:
        frame = tk.Frame(
            self.body_text,
            bg=COLORS["panel"],
            highlightthickness=1,
            highlightbackground=COLORS["button_shadow"],
            bd=1,
            relief="raised",
        )
        frame.columnconfigure(0, weight=1)

        header = tk.Frame(frame, bg=COLORS["table_soft"], padx=6, pady=4)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Table",
            bg=COLORS["table_soft"],
            fg=COLORS["table_dark"],
            font=("Segoe UI Semibold", 9),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        self.make_tk_button(
            header,
            text="+ Row",
            command=lambda current_id=table_id: self.add_table_row(current_id),
            role="table",
            font=("Segoe UI", 8),
            padx=6,
            pady=1,
        ).grid(row=0, column=1, sticky="e", padx=(6, 0))
        self.make_tk_button(
            header,
            text="+ Column",
            command=lambda current_id=table_id: self.add_table_column(current_id),
            role="table",
            font=("Segoe UI", 8),
            padx=6,
            pady=1,
        ).grid(row=0, column=2, sticky="e", padx=(6, 0))

        grid = tk.Frame(
            frame,
            bg=COLORS["border"],
            padx=1,
            pady=1,
        )
        grid.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        self.table_widgets[table_id] = {
            "frame": frame,
            "grid": grid,
            "vars": [],
            "entries": [],
        }

        for row in rows:
            self.add_table_row(table_id, row, mark_change=False)
        return frame

    def table_cell_background(self, row_index: int) -> str:
        if row_index == 0:
            return COLORS["table_header"]
        return COLORS["table_alt"] if row_index % 2 == 0 else COLORS["panel"]

    def table_column_count(self, table_id: str) -> int:
        data = self.table_widgets.get(table_id)
        if data is None:
            return 0
        rows = data.get("vars", [])
        if not isinstance(rows, list) or not rows:
            return 0
        first_row = rows[0]
        return len(first_row) if isinstance(first_row, list) else 0

    def add_table_row(
        self,
        table_id: str,
        values: list[str] | None = None,
        mark_change: bool = True,
    ) -> None:
        data = self.table_widgets.get(table_id)
        if data is None:
            return

        frame = data["frame"]
        if not isinstance(frame, tk.Frame):
            return

        grid = data.get("grid")
        if not isinstance(grid, tk.Frame):
            return

        rows = data["vars"]
        entries = data["entries"]
        if not isinstance(rows, list) or not isinstance(entries, list):
            return

        column_count = self.table_column_count(table_id)
        if values is not None:
            column_count = max(column_count, len(values))
        column_count = column_count or 3
        values = (values or []) + [""] * (column_count - len(values or []))
        row_index = len(rows)
        row_vars: list[tk.StringVar] = []
        row_entries: list[tk.Entry] = []

        for column_index in range(column_count):
            grid.columnconfigure(column_index, weight=1)
            var = tk.StringVar(value=values[column_index])
            var.trace_add("write", lambda *_args: self.mark_dirty())
            entry = self.create_table_cell(table_id, row_index, column_index, var)
            row_vars.append(var)
            row_entries.append(entry)

        rows.append(row_vars)
        entries.append(row_entries)
        if mark_change:
            self.mark_dirty()

    def create_table_cell(
        self,
        table_id: str,
        row_index: int,
        column_index: int,
        var: tk.StringVar,
    ) -> tk.Entry:
        data = self.table_widgets.get(table_id)
        if data is None:
            raise KeyError(table_id)

        grid = data.get("grid")
        if not isinstance(grid, tk.Frame):
            raise KeyError(table_id)

        cell_bg = self.table_cell_background(row_index)
        entry = tk.Entry(
            grid,
            textvariable=var,
            relief="flat",
            bd=0,
            width=20,
            font=("Segoe UI Semibold", 10) if row_index == 0 else ("Segoe UI", 10),
            bg=cell_bg,
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            selectbackground=COLORS["table"],
            selectforeground="#ffffff",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["table"],
        )
        entry.grid(row=row_index, column=column_index, sticky="nsew", padx=1, pady=1, ipady=5)
        entry.bind(
            "<FocusIn>",
            lambda event: event.widget.configure(
                bg="#ffffff",
                highlightbackground=COLORS["table"],
            ),
        )
        entry.bind(
            "<FocusOut>",
            lambda event, row=row_index: event.widget.configure(
                bg=self.table_cell_background(row),
                highlightbackground=COLORS["border"],
            ),
        )
        entry.bind(
            "<Tab>",
            lambda event, row=row_index, column=column_index, current_id=table_id: self.on_table_tab(
                event,
                current_id,
                row,
                column,
                1,
            ),
        )
        entry.bind(
            "<ISO_Left_Tab>",
            lambda event, row=row_index, column=column_index, current_id=table_id: self.on_table_tab(
                event,
                current_id,
                row,
                column,
                -1,
            ),
        )
        entry.bind(
            "<Shift-Tab>",
            lambda event, row=row_index, column=column_index, current_id=table_id: self.on_table_tab(
                event,
                current_id,
                row,
                column,
                -1,
            ),
        )
        entry.bind(
            "<Return>",
            lambda event, row=row_index, column=column_index, current_id=table_id: self.on_table_return(
                event,
                current_id,
                row,
                column,
            ),
        )
        return entry

    def add_table_column(self, table_id: str, mark_change: bool = True) -> None:
        data = self.table_widgets.get(table_id)
        if data is None:
            return

        rows = data.get("vars", [])
        entries = data.get("entries", [])
        grid = data.get("grid")
        if not isinstance(rows, list) or not isinstance(entries, list) or not isinstance(grid, tk.Frame):
            return
        if not rows:
            self.add_table_row(table_id, ["Column 1"], mark_change=False)
            return

        column_index = self.table_column_count(table_id)
        grid.columnconfigure(column_index, weight=1)
        for row_index, row in enumerate(rows):
            if not isinstance(row, list):
                continue
            value = f"Column {column_index + 1}" if row_index == 0 else ""
            var = tk.StringVar(value=value)
            var.trace_add("write", lambda *_args: self.mark_dirty())
            entry = self.create_table_cell(table_id, row_index, column_index, var)
            row.append(var)
            if row_index < len(entries) and isinstance(entries[row_index], list):
                entries[row_index].append(entry)

        if mark_change:
            self.mark_dirty()

    def on_table_tab(
        self,
        _event: tk.Event,
        table_id: str,
        row: int,
        column: int,
        direction: int,
    ) -> str:
        column_count = self.table_column_count(table_id)
        if column_count <= 0:
            return "break"

        next_row = row
        next_column = column + direction
        if next_column >= column_count:
            next_row += 1
            next_column = 0
        elif next_column < 0:
            next_row -= 1
            next_column = column_count - 1

        self.focus_table_cell(table_id, next_row, next_column)
        return "break"

    def on_table_return(
        self,
        _event: tk.Event,
        table_id: str,
        row: int,
        column: int,
    ) -> str:
        self.focus_table_cell(table_id, row + 1, column)
        return "break"

    def focus_table_cell(self, table_id: str, row: int, column: int) -> None:
        if row < 0:
            return
        data = self.table_widgets.get(table_id)
        if data is None:
            return

        entries = data.get("entries", [])
        if not isinstance(entries, list):
            return

        while row >= len(entries):
            self.add_table_row(table_id)

        if row < len(entries) and column < len(entries[row]):
            entries[row][column].focus_set()
            entries[row][column].selection_range(0, tk.END)

    def get_table_rows(self, table_id: str) -> list[list[str]]:
        data = self.table_widgets.get(table_id)
        if data is None:
            return []

        rows = data.get("vars", [])
        if not isinstance(rows, list):
            return []

        values: list[list[str]] = []
        for row in rows:
            if isinstance(row, list):
                values.append([cell.get() for cell in row if isinstance(cell, tk.StringVar)])
        return values

    def insert_table_widget(
        self,
        rows: list[list[str]],
        index: str | None = None,
        table_id: str | None = None,
    ) -> str:
        table_id = table_id or uuid.uuid4().hex[:12]
        rows = [
            ["" if cell is None else str(cell) for cell in row]
            for row in rows
            if isinstance(row, list)
        ]
        if not rows:
            rows = [["Column 1", "Column 2", "Column 3"], ["", "", ""], ["", "", ""]]
        column_count = max(len(row) for row in rows)
        rows = [row + [""] * (column_count - len(row)) for row in rows]
        frame = self.create_table_editor(rows, table_id)
        self.body_text.window_create(index or tk.INSERT, window=frame, padx=2, pady=6)
        self.table_window_to_id[str(frame)] = table_id
        return table_id

    def is_table_line(self, line: str) -> bool:
        return split_table_line(line) is not None

    def table_blocks(self) -> list[tuple[int, int, list[str]]]:
        end_line = int(self.body_text.index("end-1c").split(".")[0])
        blocks: list[tuple[int, int, list[str]]] = []
        block_start: int | None = None
        block_lines: list[str] = []

        for line_number in range(1, end_line + 1):
            line_text = self.body_text.get(f"{line_number}.0", f"{line_number}.end")
            if self.is_table_line(line_text):
                if block_start is None:
                    block_start = line_number
                block_lines.append(line_text)
                continue

            if block_start is not None and len(block_lines) >= 2:
                blocks.append((block_start, line_number - 1, block_lines))
            block_start = None
            block_lines = []

        if block_start is not None and len(block_lines) >= 2:
            blocks.append((block_start, end_line, block_lines))
        return blocks

    def refresh_table_tags(self) -> None:
        if self.loading:
            return
        self.table_style_job = None
        self.body_text.tag_remove("table", "1.0", "end")
        for start_line, end_line, _lines in self.table_blocks():
            self.body_text.tag_add("table", f"{start_line}.0", f"{end_line}.end")

    def refresh_checklist_tags(self) -> None:
        if self.loading:
            return
        self.body_text.tag_remove("checklist_box", "1.0", "end")
        end_line = int(self.body_text.index("end-1c").split(".")[0])
        for line_number in range(1, end_line + 1):
            line_text = self.body_text.get(f"{line_number}.0", f"{line_number}.end")
            for match in CHECKLIST_MARK_RE.finditer(line_text):
                self.body_text.tag_add(
                    "checklist_box",
                    f"{line_number}.{match.start()}",
                    f"{line_number}.{match.end()}",
                )

    def refresh_editor_tags(self) -> None:
        self.refresh_table_tags()
        self.refresh_checklist_tags()

    def selected_table_rows(self) -> tuple[str, str, list[list[str]]] | None:
        try:
            start = self.body_text.index("sel.first")
            end = self.body_text.index("sel.last")
        except tk.TclError:
            return None

        selected_text = self.body_text.get(start, end)
        rows = normalize_table_rows(selected_text.splitlines(), allow_single_space=True)
        if not rows:
            return None
        return start, end, rows

    def apply_table_interpretation(self) -> None:
        selected_rows = self.selected_table_rows()
        if selected_rows is not None:
            start, end, rows = selected_rows
            self.body_text.delete(start, end)
            self.body_text.mark_set(tk.INSERT, start)
            table_id = self.insert_table_widget(rows)
            self.body_text.insert(tk.INSERT, "\n")
            self.mark_dirty()
            self.status_var.set("Converted selection to a table")
            self.after_idle(lambda current_id=table_id: self.focus_table_cell(current_id, 0, 0))
            return

        blocks = self.table_blocks()
        if not blocks:
            self.status_var.set(
                "No table-like text found. Use pipes, tabs, commas, or aligned spaces."
            )
            return

        converted = 0
        focus_table_id: str | None = None
        for start_line, end_line, lines in reversed(blocks):
            rows = normalize_table_rows(lines)
            if not rows:
                continue
            start_index = f"{start_line}.0"
            self.body_text.delete(start_index, f"{end_line}.end")
            self.body_text.mark_set(tk.INSERT, start_index)
            focus_table_id = self.insert_table_widget(rows)
            self.body_text.insert(tk.INSERT, "\n")
            converted += 1

        self.refresh_editor_tags()
        self.mark_dirty()
        self.status_var.set(
            f"Converted {converted} table block" + ("" if converted == 1 else "s")
        )
        if focus_table_id is not None:
            self.after_idle(lambda current_id=focus_table_id: self.focus_table_cell(current_id, 0, 0))

    def mark_dirty(self) -> None:
        if self.loading:
            return
        self.dirty = True
        self.status_var.set("Unsaved changes...")
        if self.autosave_job is not None:
            self.after_cancel(self.autosave_job)
        self.autosave_job = self.after(1200, self.save_current_note)

    def get_calendar_month(self) -> tuple[int, int]:
        try:
            year = int(self.year_var.get())
        except (tk.TclError, ValueError):
            year = self.current_date.year
            self.year_var.set(year)

        month_name = self.month_name_var.get()
        if month_name not in MONTH_NAMES:
            month = self.current_date.month
            self.month_name_var.set(calendar.month_name[month])
        else:
            month = MONTH_NAMES.index(month_name) + 1
        return year, month

    def set_calendar_month(self, year: int, month: int) -> None:
        self.year_var.set(year)
        self.month_name_var.set(calendar.month_name[month])
        self.refresh_calendar()

    def on_calendar_month_changed(self) -> None:
        self.refresh_calendar()
        if hasattr(self, "library_inner"):
            self.refresh_library_tree()

    def shift_month(self, offset: int) -> None:
        year, month = self.get_calendar_month()
        month += offset
        while month < 1:
            month += 12
            year -= 1
        while month > 12:
            month -= 12
            year += 1
        self.set_calendar_month(year, month)
        self.refresh_library_tree()

    def shift_day(self, offset: int) -> None:
        self.load_date(self.current_date + timedelta(days=offset))

    def refresh_calendar(self) -> None:
        year, month = self.get_calendar_month()
        month_days = calendar.Calendar(firstweekday=0).monthdayscalendar(year, month)
        while len(month_days) < 6:
            month_days.append([0, 0, 0, 0, 0, 0, 0])

        flat_days = [day for week in month_days for day in week]
        today_day = today()
        for button, day_number in zip(self.day_buttons, flat_days):
            if day_number == 0:
                button.configure(
                    text="",
                    state=tk.DISABLED,
                    command=lambda: None,
                    bg=COLORS["sidebar"],
                    activebackground=COLORS["sidebar"],
                    cursor="arrow",
                    relief="flat",
                    bd=0,
                    highlightthickness=0,
                )
                continue

            day = date(year, month, day_number)
            has_note = day in self.note_dates
            is_selected = day == self.current_date
            is_today = day == today_day

            if is_selected:
                bg = COLORS["accent"]
                fg = "#ffffff"
            elif has_note and is_today:
                bg = COLORS["soft_teal"]
                fg = COLORS["text"]
            elif has_note:
                bg = COLORS["soft_accent"]
                fg = COLORS["text"]
            elif is_today:
                bg = "#fef3c7"
                fg = COLORS["text"]
            else:
                bg = COLORS["panel"]
                fg = COLORS["text"]

            button.configure(
                text=str(day_number),
                state=tk.NORMAL,
                command=lambda selected=day: self.load_date(selected),
                bg=bg,
                fg=fg,
                activebackground=COLORS["soft_teal"] if not is_selected else COLORS["accent_dark"],
                activeforeground=fg,
                cursor="hand2",
                relief="raised",
                bd=2,
                highlightthickness=1,
                highlightbackground=COLORS["accent"] if is_selected else COLORS["button_shadow"],
            )

    def update_date_header(self) -> None:
        note_ids = self.store.note_ids_for_day(self.current_date)
        if self.current_note_id in note_ids:
            note_text = f" - Note {note_ids.index(self.current_note_id) + 1} of {len(note_ids)}"
        elif note_ids:
            note_text = f" - {len(note_ids)} saved note" + ("" if len(note_ids) == 1 else "s")
        else:
            note_text = " - New note"
        self.date_label.configure(text=f"{format_date_long(self.current_date)}{note_text}")

    def refresh_library_tree(self) -> None:
        if not hasattr(self, "library_inner"):
            return

        for child in self.library_inner.winfo_children():
            child.destroy()

        entries = self.library_entries()
        count = len(entries)
        self.library_count_label.configure(
            text=f"{count} note" if count == 1 else f"{count} notes"
        )

        if not entries:
            self.add_library_empty_state()
            return

        last_month: tuple[int, int] | None = None
        for day, note_id, note, note_number, note_total in entries:
            month_key = (day.year, day.month)
            if month_key != last_month:
                self.add_library_month_header(day)
                last_month = month_key
            self.add_library_card(day, note_id, note, note_number, note_total)

    def library_entries(self) -> list[tuple[date, str, dict, int, int]]:
        mode = self.library_filter_var.get()
        shown_year, shown_month = self.get_calendar_month()
        entries: list[tuple[date, str, dict, int, int]] = []

        for day in sorted(self.note_dates, reverse=True):
            if mode == "Shown month" and (day.year != shown_year or day.month != shown_month):
                continue
            if mode == "Shown year" and day.year != shown_year:
                continue

            note_ids = self.store.note_ids_for_day(day)
            note_total = len(note_ids)
            for note_id in reversed(note_ids):
                note = self.store.load(day, note_id)
                note_number = note_ids.index(note_id) + 1
                entries.append((day, note_id, note, note_number, note_total))

        entries.sort(
            key=lambda item: (
                item[0],
                item[2].get("updated_at", item[2].get("created_at", "")),
            ),
            reverse=True,
        )
        if mode == "Recent":
            entries = entries[:50]
        return entries

    def add_library_empty_state(self) -> None:
        mode = self.library_filter_var.get()
        text = "No notes in this view."
        if mode in {"Shown month", "Shown year"}:
            text = "No notes for the shown date range."
        empty = tk.Label(
            self.library_inner,
            text=text,
            bg=COLORS["sidebar"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
            justify="left",
            wraplength=260,
        )
        empty.pack(fill=tk.X, padx=16, pady=12)

    def add_library_month_header(self, day: date) -> None:
        header = tk.Label(
            self.library_inner,
            text=day.strftime("%B %Y"),
            bg=COLORS["sidebar"],
            fg=COLORS["muted"],
            anchor="w",
            font=("Segoe UI Semibold", 9),
        )
        header.pack(fill=tk.X, padx=16, pady=(10, 5))

    def add_library_card(
        self,
        day: date,
        note_id: str,
        note: dict,
        note_number: int,
        note_total: int,
    ) -> None:
        is_selected = day == self.current_date and note_id == self.current_note_id
        card_bg = COLORS["soft_accent"] if is_selected else COLORS["panel"]
        date_bg = COLORS["accent"] if is_selected else COLORS["soft_teal"]
        date_fg = "#ffffff" if is_selected else COLORS["teal"]
        border = COLORS["accent"] if is_selected else COLORS["border"]

        card = tk.Frame(
            self.library_inner,
            bg=card_bg,
            highlightbackground=border,
            highlightthickness=1,
            relief="raised",
            bd=1,
            padx=8,
            pady=8,
            cursor="hand2",
        )
        card.pack(fill=tk.X, padx=16, pady=(0, 8))
        card.columnconfigure(1, weight=1)

        date_pill = tk.Frame(
            card,
            bg=date_bg,
            width=46,
            height=54,
            relief="raised",
            bd=1,
            highlightbackground=border,
            highlightthickness=1,
        )
        date_pill.grid(row=0, column=0, rowspan=3, sticky="nw", padx=(0, 9))
        date_pill.grid_propagate(False)
        tk.Label(
            date_pill,
            text=day.strftime("%d"),
            bg=date_bg,
            fg=date_fg,
            font=("Segoe UI Semibold", 15),
        ).pack(pady=(5, 0))
        tk.Label(
            date_pill,
            text=day.strftime("%a"),
            bg=date_bg,
            fg=date_fg,
            font=("Segoe UI", 8),
        ).pack()

        title = note.get("title", "").strip() or f"Untitled note {note_number}"
        tk.Label(
            card,
            text=title,
            bg=card_bg,
            fg=COLORS["text"],
            anchor="w",
            justify="left",
            wraplength=205,
            font=("Segoe UI Semibold", 9),
        ).grid(row=0, column=1, sticky="ew")

        preview = self.note_preview(note)
        tk.Label(
            card,
            text=preview or "No text yet",
            bg=card_bg,
            fg=COLORS["muted"],
            anchor="w",
            justify="left",
            wraplength=205,
            font=("Segoe UI", 8),
        ).grid(row=1, column=1, sticky="ew", pady=(3, 0))

        meta = self.note_meta(note, note_number, note_total)
        tk.Label(
            card,
            text=meta,
            bg=card_bg,
            fg=COLORS["muted"],
            anchor="w",
            justify="left",
            wraplength=205,
            font=("Segoe UI", 8),
        ).grid(row=2, column=1, sticky="ew", pady=(4, 0))

        self.bind_library_card(card, day, note_id)
        for child in card.winfo_children():
            self.bind_library_card(child, day, note_id)
            for grandchild in child.winfo_children():
                self.bind_library_card(grandchild, day, note_id)

    def bind_library_card(self, widget: tk.Widget, day: date, note_id: str) -> None:
        widget.configure(cursor="hand2")
        widget.bind("<Button-1>", lambda _event, selected_day=day, selected_id=note_id: self.load_date(selected_day, selected_id))
        widget.bind("<MouseWheel>", self.on_library_mousewheel)

    def note_preview(self, note: dict) -> str:
        text = re.sub(r"\|?\s*-{3,}\s*\|?", " ", note.get("body", ""))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 86:
            return text[:83] + "..."
        return text

    def note_meta(self, note: dict, note_number: int, note_total: int) -> str:
        parts: list[str] = []
        if note_total > 1:
            parts.append(f"Note {note_number} of {note_total}")

        attachment_count = len(note.get("attachments", []))
        if attachment_count:
            parts.append(f"{attachment_count} file" if attachment_count == 1 else f"{attachment_count} files")

        updated = note.get("updated_at", "")
        try:
            updated_dt = datetime.fromisoformat(updated)
            parts.append(f"Saved {updated_dt.strftime('%b %d, %H:%M')}")
        except ValueError:
            parts.append("Saved")
        return " | ".join(parts)

    def on_library_mousewheel(self, event: tk.Event) -> None:
        if not hasattr(self, "library_canvas"):
            return
        delta = int(-1 * (event.delta / 120))
        if delta:
            self.library_canvas.yview_scroll(delta, "units")

    def refresh_search_results(self) -> None:
        query = self.search_var.get().strip()
        self.search_list.delete(0, tk.END)
        self.search_results = []
        if not query:
            self.search_list.insert(tk.END, "Type to search notes")
            self.search_list.itemconfig(0, foreground=COLORS["muted"])
            return

        self.search_results = self.store.search(query)
        if not self.search_results:
            self.search_list.insert(tk.END, "No matches")
            self.search_list.itemconfig(0, foreground=COLORS["muted"])
            return

        for match in self.search_results:
            preview = f" - {match.preview}" if match.preview else ""
            self.search_list.insert(tk.END, f"{match.day.isoformat()}  {match.title}{preview}")

    def on_search_changed(self, *_args: object) -> None:
        if self.search_job is not None:
            self.after_cancel(self.search_job)
        self.search_job = self.after(250, self.refresh_search_results)

    def on_search_open(self, _event: tk.Event) -> None:
        selection = self.search_list.curselection()
        if not selection:
            return
        index = selection[0]
        if index >= len(self.search_results):
            return
        match = self.search_results[index]
        self.load_date(match.day, match.note_id)

    def serialize_editor_content(self) -> list[dict]:
        ops: list[dict] = []
        for key, value, _index in self.body_text.dump(
            "1.0",
            "end-1c",
            text=True,
            tag=True,
            image=True,
            window=True,
        ):
            if key == "text":
                ops.append({"kind": "text", "text": value})
            elif key in {"tagon", "tagoff"} and value in RICH_TEXT_TAGS:
                ops.append({"kind": key, "tag": value})
            elif key == "image":
                attachment_id = self.inline_image_name_to_attachment.get(value)
                if attachment_id:
                    ops.append({"kind": "image", "attachment_id": attachment_id})
            elif key == "window":
                table_id = self.table_window_to_id.get(value)
                if table_id:
                    ops.append(
                        {
                            "kind": "table",
                            "table_id": table_id,
                            "rows": self.get_table_rows(table_id),
                        }
                    )
        return ops

    def plain_text_from_ops(self, ops: list[dict]) -> str:
        pieces: list[str] = []
        for op in ops:
            kind = op.get("kind")
            if kind == "text":
                pieces.append(op.get("text", ""))
            elif kind == "table":
                rows = op.get("rows", [])
                if isinstance(rows, list):
                    pieces.append("\n" + table_rows_to_text(rows) + "\n")
        return "".join(pieces)

    def render_note_content(self, note: dict) -> None:
        self.body_text.delete("1.0", tk.END)
        self.inline_image_refs.clear()
        self.inline_image_name_to_attachment.clear()
        self.table_widgets.clear()
        self.table_window_to_id.clear()

        ops = note.get("content_ops") or []
        if not ops:
            self.body_text.insert("1.0", note.get("body", ""))
            self.refresh_editor_tags()
            return

        active_tags: list[str] = []
        for op in ops:
            kind = op.get("kind")
            if kind == "text":
                self.body_text.insert(tk.INSERT, op.get("text", ""), tuple(active_tags))
            elif kind == "tagon":
                tag = op.get("tag")
                if tag in RICH_TEXT_TAGS and tag not in active_tags:
                    active_tags.append(tag)
            elif kind == "tagoff":
                tag = op.get("tag")
                if tag in active_tags:
                    active_tags.remove(tag)
            elif kind == "image":
                item = self.find_attachment(op.get("attachment_id"))
                if item is not None:
                    self.insert_inline_attachment_image(
                        item,
                        tk.INSERT,
                        mark_change=False,
                        add_spacing=False,
                    )
            elif kind == "table":
                rows = op.get("rows", [])
                if isinstance(rows, list):
                    self.insert_table_widget(
                        rows,
                        tk.INSERT,
                        table_id=op.get("table_id") or uuid.uuid4().hex[:12],
                    )

        self.refresh_editor_tags()

    def make_inline_photo(self, path: Path) -> ImageTk.PhotoImage | None:
        if Image is None or ImageTk is None:
            return None
        try:
            with Image.open(path) as image:
                image.thumbnail((520, 360), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(image.copy())
        except Exception:
            return None

    def insert_inline_attachment_image(
        self,
        item: dict,
        index: str | None = None,
        mark_change: bool = True,
        add_spacing: bool = True,
    ) -> None:
        path = self.store.attachment_path(item)
        insert_at = index or tk.INSERT
        if item.get("kind") != "image":
            return

        photo = self.make_inline_photo(path)
        if photo is None:
            label = f"[Image unavailable: {item.get('name', 'image')}]"
            self.body_text.insert(insert_at, label, ("mono",))
            if mark_change:
                self.mark_dirty()
            return

        name = f"img_{uuid.uuid4().hex}"
        if add_spacing:
            self.body_text.insert(insert_at, "\n")
        self.body_text.image_create(
            tk.INSERT,
            image=photo,
            name=name,
            padx=6,
            pady=6,
            align="center",
        )
        if add_spacing:
            self.body_text.insert(tk.INSERT, "\n")
        self.inline_image_refs[name] = photo
        self.inline_image_name_to_attachment[name] = item.get("id", "")
        if mark_change:
            self.mark_dirty()

    def load_date(self, day: date, note_id: str | None = None, first_load: bool = False) -> None:
        requested_id = note_id or self.store.default_note_id(day)
        if (
            not first_load
            and day == self.current_date
            and (requested_id is None or requested_id == self.current_note_id)
            and not self.dirty
        ):
            return

        if not first_load and self.dirty:
            self.save_current_note()

        self.loading = True
        self.current_date = day
        self.current_note = self.store.load(day, requested_id)
        self.current_note_id = self.current_note["id"]
        self.selected_attachment_id = None
        self.title_var.set(self.current_note.get("title", ""))
        self.render_note_content(self.current_note)
        self.body_text.edit_modified(False)
        self.loading = False
        self.refresh_editor_tags()
        self.dirty = False
        self.autosave_job = None

        self.set_calendar_month(day.year, day.month)
        self.note_dates = self.store.all_note_dates()
        self.update_date_header()
        self.refresh_calendar()
        self.refresh_library_tree()
        self.refresh_attachments()
        self.refresh_search_results()
        updated = self.current_note.get("updated_at", "")
        if updated:
            self.status_var.set(f"Loaded {day.isoformat()}   Last saved: {updated}")
        else:
            self.status_var.set(f"Loaded {day.isoformat()}")

    def note_from_editor(self) -> dict:
        ops = self.serialize_editor_content()
        self.current_note["id"] = self.current_note_id
        self.current_note["date"] = self.current_date.isoformat()
        self.current_note["title"] = self.title_var.get().strip()
        self.current_note["body"] = self.plain_text_from_ops(ops)
        self.current_note["content_ops"] = ops
        self.current_note.setdefault("attachments", [])
        self.current_note.setdefault("created_at", now_iso())
        return self.current_note

    def save_current_note(self) -> None:
        if self.loading:
            return
        if self.autosave_job is not None:
            try:
                self.after_cancel(self.autosave_job)
            except tk.TclError:
                pass
            self.autosave_job = None

        note = self.note_from_editor()
        self.store.save(note)
        self.current_note = note
        self.current_note_id = note["id"]
        self.dirty = False
        self.note_dates = self.store.all_note_dates()
        self.update_date_header()
        self.refresh_calendar()
        self.refresh_library_tree()
        self.refresh_search_results()
        self.status_var.set(f"Saved {self.current_date.isoformat()} at {datetime.now().strftime('%H:%M:%S')}")

    def add_note_for_current_day(self) -> None:
        if self.dirty:
            self.save_current_note()

        note = self.store.empty_note(self.current_date)
        note["title"] = "Untitled note"
        self.store.save(note)
        self.note_dates = self.store.all_note_dates()
        self.load_date(self.current_date, note["id"])
        self.title_entry.focus_set()
        self.title_entry.selection_range(0, tk.END)

    def insert_time_stamp(self) -> None:
        stamp = datetime.now().strftime("[%H:%M]")
        insert_at = self.body_text.index(tk.INSERT)
        prefix = "" if insert_at == "1.0" else "\n"
        self.body_text.insert(tk.INSERT, f"{prefix}{stamp}\n")
        self.body_text.focus_set()
        self.mark_dirty()

    def add_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.add_copied_attachment(Path(path), kind="image", inline=True)

    def attach_file_copy(self) -> None:
        path = filedialog.askopenfilename(title="Choose a file to copy into this note")
        if path:
            source = Path(path)
            kind = "image" if is_image_path(source) else "file"
            self.add_copied_attachment(source, kind=kind, inline=kind == "image")

    def link_file(self) -> None:
        path = filedialog.askopenfilename(title="Choose a file to link")
        if not path:
            return
        source = Path(path)
        kind = "image" if is_image_path(source) else "file"
        item = {
            "id": str(uuid.uuid4()),
            "kind": kind,
            "mode": "linked",
            "name": source.name,
            "path": str(source),
            "added_at": now_iso(),
        }
        self.current_note.setdefault("attachments", []).append(item)
        if kind == "image":
            self.insert_inline_attachment_image(item)
        self.mark_dirty()
        self.save_current_note()
        self.refresh_attachments()

    def add_copied_attachment(self, source: Path, kind: str, inline: bool = False) -> None:
        if not source.exists():
            messagebox.showerror(APP_NAME, f"File not found:\n{source}")
            return

        assets = self.store.asset_dir(self.current_date, self.current_note_id)
        assets.mkdir(parents=True, exist_ok=True)
        destination = unique_path(assets / safe_filename(source.name))
        try:
            shutil.copy2(source, destination)
        except OSError as exc:
            messagebox.showerror(APP_NAME, f"Could not copy the file:\n{exc}")
            return

        try:
            relative = destination.relative_to(self.store.root)
        except ValueError:
            relative = destination

        item = {
            "id": str(uuid.uuid4()),
            "kind": kind,
            "mode": "copied",
            "name": source.name,
            "path": relative.as_posix(),
            "added_at": now_iso(),
        }
        self.current_note.setdefault("attachments", []).append(item)
        if inline and kind == "image":
            self.insert_inline_attachment_image(item)
        self.mark_dirty()
        self.save_current_note()
        self.refresh_attachments()

    def refresh_attachments(self) -> None:
        for child in self.attach_inner.winfo_children():
            child.destroy()
        self.thumbnail_refs.clear()

        attachments = self.current_note.get("attachments", [])
        count = len(attachments)
        self.attachment_count.configure(text=f"{count} file" if count == 1 else f"{count} files")

        if not attachments:
            empty = tk.Label(
                self.attach_inner,
                text="No attachments for this day.",
                bg=COLORS["panel"],
                fg=COLORS["muted"],
                font=("Segoe UI", 9),
                justify="left",
                wraplength=250,
            )
            empty.pack(fill=tk.X, padx=14, pady=14)
            return

        for item in attachments:
            self.add_attachment_card(item)

    def add_attachment_card(self, item: dict) -> None:
        item_id = item.get("id", "")
        selected = item_id == self.selected_attachment_id
        card_bg = COLORS["soft_accent"] if selected else COLORS["panel"]
        border = COLORS["accent"] if selected else COLORS["border"]
        path = self.store.attachment_path(item)
        exists = path.exists()
        mode = item.get("mode", "linked")
        kind = item.get("kind", "file")

        card = tk.Frame(
            self.attach_inner,
            bg=card_bg,
            highlightbackground=border,
            highlightthickness=1,
            relief="raised",
            bd=1,
            padx=8,
            pady=8,
        )
        card.pack(fill=tk.X, padx=12, pady=(0, 8))
        card.columnconfigure(1, weight=1)

        thumb = self.make_thumbnail(path) if kind == "image" and exists else None
        if thumb is not None:
            image_label = tk.Label(card, image=thumb, bg=card_bg, width=74, height=74)
            image_label.grid(row=0, column=0, rowspan=3, sticky="nw", padx=(0, 8))
            self.thumbnail_refs.append(thumb)
        else:
            badge_text = "IMG" if kind == "image" else "FILE"
            if not exists:
                badge_text = "MISS"
            image_label = tk.Label(
                card,
                text=badge_text,
                width=8,
                height=4,
                bg=COLORS["soft_danger"] if not exists else "#e2e8f0",
                fg=COLORS["danger"] if not exists else COLORS["text"],
                font=("Segoe UI Semibold", 8),
            )
            image_label.grid(row=0, column=0, rowspan=3, sticky="nw", padx=(0, 8))

        name = item.get("name", "Attachment")
        name_label = tk.Label(
            card,
            text=name,
            bg=card_bg,
            fg=COLORS["text"],
            anchor="w",
            justify="left",
            wraplength=190,
            font=("Segoe UI Semibold", 9),
        )
        name_label.grid(row=0, column=1, sticky="ew")

        detail_text = "Copied into note" if mode == "copied" else "Linked file"
        if not exists:
            detail_text += " - missing"
        detail_label = tk.Label(
            card,
            text=detail_text,
            bg=card_bg,
            fg=COLORS["muted"] if exists else COLORS["danger"],
            anchor="w",
            font=("Segoe UI", 8),
        )
        detail_label.grid(row=1, column=1, sticky="ew", pady=(2, 0))

        path_label = tk.Label(
            card,
            text=str(path),
            bg=card_bg,
            fg=COLORS["muted"],
            anchor="w",
            justify="left",
            wraplength=190,
            font=("Segoe UI", 8),
        )
        path_label.grid(row=2, column=1, sticky="ew", pady=(2, 0))

        for widget in (card, image_label, name_label, detail_label, path_label):
            widget.bind("<Button-1>", lambda _event, value=item_id: self.select_attachment(value))
            widget.bind("<Double-Button-1>", lambda _event, value=item_id: self.open_attachment(value))

    def make_thumbnail(self, path: Path) -> ImageTk.PhotoImage | None:
        if Image is None or ImageTk is None:
            return None

        try:
            with Image.open(path) as image:
                image.thumbnail((76, 76), Image.Resampling.LANCZOS)
                canvas = Image.new("RGBA", (76, 76), (255, 255, 255, 0))
                x = (76 - image.width) // 2
                y = (76 - image.height) // 2
                canvas.paste(image.convert("RGBA"), (x, y))
                return ImageTk.PhotoImage(canvas)
        except Exception:
            return None

    def select_attachment(self, attachment_id: str) -> None:
        self.selected_attachment_id = attachment_id
        self.refresh_attachments()

    def find_attachment(self, attachment_id: str | None) -> dict | None:
        if not attachment_id:
            return None
        for item in self.current_note.get("attachments", []):
            if item.get("id") == attachment_id:
                return item
        return None

    def open_selected_attachment(self) -> None:
        self.open_attachment(self.selected_attachment_id)

    def open_attachment(self, attachment_id: str | None) -> None:
        item = self.find_attachment(attachment_id)
        if item is None:
            messagebox.showinfo(APP_NAME, "Select an attachment first.")
            return
        path = self.store.attachment_path(item)
        try:
            open_with_default_app(path)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not open the attachment:\n{exc}")

    def remove_inline_images_for_attachment(self, attachment_id: str | None) -> None:
        if not attachment_id:
            return
        for image_name, mapped_id in list(self.inline_image_name_to_attachment.items()):
            if mapped_id != attachment_id:
                continue
            try:
                self.body_text.delete(image_name)
            except tk.TclError:
                pass
            self.inline_image_name_to_attachment.pop(image_name, None)
            self.inline_image_refs.pop(image_name, None)

    def remove_selected_attachment(self) -> None:
        item = self.find_attachment(self.selected_attachment_id)
        if item is None:
            messagebox.showinfo(APP_NAME, "Select an attachment first.")
            return

        name = item.get("name", "this attachment")
        if not messagebox.askyesno(APP_NAME, f"Remove {name} from this note?"):
            return

        attachments = self.current_note.get("attachments", [])
        self.current_note["attachments"] = [
            attachment
            for attachment in attachments
            if attachment.get("id") != self.selected_attachment_id
        ]
        self.remove_inline_images_for_attachment(self.selected_attachment_id)
        if item.get("mode") == "copied":
            path = self.store.attachment_path(item)
            try:
                if path.exists() and path.resolve().is_relative_to(self.store.root.resolve()):
                    path.unlink()
            except OSError:
                pass

        self.selected_attachment_id = None
        self.mark_dirty()
        self.save_current_note()
        self.refresh_attachments()

    def delete_current_note(self) -> None:
        title = self.title_var.get().strip() or "this note"
        if not messagebox.askyesno(
            APP_NAME,
            f"Delete {title} and its copied files for {self.current_date.isoformat()}?",
        ):
            return

        self.store.delete_note(self.current_date, self.current_note_id)
        self.note_dates = self.store.all_note_dates()
        next_note_id = self.store.default_note_id(self.current_date)
        self.current_note = self.store.load(self.current_date, next_note_id)
        self.current_note_id = self.current_note["id"]
        self.loading = True
        self.title_var.set(self.current_note.get("title", ""))
        self.body_text.delete("1.0", tk.END)
        self.render_note_content(self.current_note)
        self.loading = False
        self.refresh_editor_tags()
        self.body_text.edit_modified(False)
        self.dirty = False
        self.selected_attachment_id = None
        self.update_date_header()
        self.refresh_calendar()
        self.refresh_library_tree()
        self.refresh_attachments()
        self.refresh_search_results()
        self.status_var.set(f"Deleted note for {self.current_date.isoformat()}")

    def open_notes_folder(self) -> None:
        self.store.ensure()
        try:
            open_with_default_app(self.store.root)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not open the notes folder:\n{exc}")

    def on_close(self) -> None:
        if self.dirty:
            self.save_current_note()
        self.destroy()


def install_desktop_shortcut() -> Path:
    if sys.platform != "win32":
        raise RuntimeError("Desktop shortcut creation is only available on Windows.")

    desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut_path = desktop / f"{APP_NAME}.lnk"
    script_path = Path(__file__).resolve()
    pythonw = shutil.which("pythonw")
    if pythonw is None:
        candidate = Path(sys.executable).with_name("pythonw.exe")
        pythonw = str(candidate if candidate.exists() else sys.executable)

    icon_location = f"{ICON_PATH},0" if ICON_PATH.exists() else r"%SystemRoot%\System32\shell32.dll,70"
    command = "\n".join(
        [
            "$shell = New-Object -ComObject WScript.Shell",
            f"$shortcut = $shell.CreateShortcut({powershell_single_quote(str(shortcut_path))})",
            f"$shortcut.TargetPath = {powershell_single_quote(str(pythonw))}",
            f"$shortcut.Arguments = {powershell_single_quote(str(script_path))}",
            f"$shortcut.WorkingDirectory = {powershell_single_quote(str(script_path.parent))}",
            f"$shortcut.IconLocation = {powershell_single_quote(icon_location)}",
            "$shortcut.WindowStyle = 1",
            "$shortcut.Save()",
        ]
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        check=True,
    )
    return shortcut_path


def run_self_test() -> None:
    temp_root = Path(tempfile.mkdtemp(prefix="daily-notes-test-"))
    try:
        store = NotesStore(temp_root)
        test_day = date(2026, 7, 9)
        note = store.load(test_day)
        note["title"] = "Test note"
        note["body"] = "A searchable body."
        note["attachments"].append(
            {
                "id": "test",
                "kind": "file",
                "mode": "linked",
                "name": "example.txt",
                "path": str(temp_root / "example.txt"),
                "added_at": now_iso(),
            }
        )
        store.save(note)

        loaded = store.load(test_day)
        assert loaded["title"] == "Test note"
        assert loaded["body"] == "A searchable body."

        second = store.empty_note(test_day)
        second["title"] = "Second note"
        second["body"] = "Another searchable entry."
        store.save(second)

        note_ids = store.note_ids_for_day(test_day)
        assert len(note_ids) == 2
        assert test_day in store.all_note_dates()
        matches = store.search("Another")
        assert matches and matches[0].day == test_day and matches[0].note_id == second["id"]

        pipe_rows = normalize_table_rows(
            ["| Name | Dose |", "| --- | --- |", "| A | 10 |"]
        )
        assert pipe_rows == [["Name", "Dose"], ["A", "10"]]
        aligned_rows = normalize_table_rows(["Name  Dose", "A     10"])
        assert aligned_rows == [["Name", "Dose"], ["A", "10"]]
        selected_rows = normalize_table_rows(["Name Dose", "A 10"], allow_single_space=True)
        assert selected_rows == [["Name", "Dose"], ["A", "10"]]
        assert CHECKLIST_MARK_RE.search(CHECKBOX_EMPTY)
        assert table_rows_to_text(pipe_rows) == "Name\tDose\nA\t10"
        print("Self-test passed.")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return
    if "--install-shortcut" in sys.argv:
        shortcut = install_desktop_shortcut()
        print(f"Created shortcut: {shortcut}")
        return

    app = DailyNotesApp()
    app.mainloop()


if __name__ == "__main__":
    main()
