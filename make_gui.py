#!/usr/bin/env python3
"""Tkinter-based launcher for remote vehicle helper scripts."""
from __future__ import annotations

import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT_DIR = Path(__file__).resolve().parent
REMOTE_DIR = (ROOT_DIR / "remote").resolve()

DEFAULT_VEHICLE_ID = "A2"
DEFAULT_USERNAME = ""

@dataclass
class CommandSpec:
    label: str
    command: str | None = None
    log_key: str | None = None
    requires_vehicle: bool = False
    requires_username: bool = False
    stop_before: bool = False
    note: str | None = None
    formatter: Optional[Callable[[str, str], str]] = None
    kind: str = "command"  # command, stop, stop_all

    def render(self, vehicle_id: str, username: str) -> str:
        if self.kind != "command":
            return ""
        if self.formatter:
            return self.formatter(vehicle_id, username)
        values = {
            "vehicle_id": vehicle_id,
            "username": username,
        }
        assert self.command is not None
        return self.command.format(**values)

COMMANDS: List[CommandSpec] = [
    CommandSpec(
        label="Start Zenoh",
        command="./connect_zenoh.bash {vehicle_id}",
        log_key="zenoh",
        requires_vehicle=True,
        note="指定した Vehicle ID の zenoh-bridge へ接続します。",
    ),
    CommandSpec(
        label="Stop Zenoh",
        log_key="zenoh",
        kind="stop",
        requires_vehicle=True,
        note="GUI で起動した Zenoh プロセスを終了します (Ctrl+C 相当)。",
    ),
    CommandSpec(
        label="Restart Zenoh",
        command="./restart.bash {vehicle_id}",
        log_key="zenoh",
        requires_vehicle=True,
        stop_before=True,
        note="既存プロセス停止後に zenoh bridge を再接続します。",
    ),
    CommandSpec(
        label="Start RViz",
        command="./rviz.bash",
        log_key="rviz",
        note="RViz 用コンテナを起動します。",
    ),
    CommandSpec(
        label="Stop RViz",
        command="./rviz.bash down",
        log_key="rviz",
        stop_before=True,
        note="RViz コンテナを停止します。",
    ),
    CommandSpec(
        label="Restart RViz",
        command="./rviz.bash restart",
        log_key="rviz",
        note="RViz コンテナを再起動します。",
    ),
    CommandSpec(
        label="Start Joy",
        command="./joy.bash",
        log_key="joy",
        note="ゲームパッドノードを起動します。",
    ),
    CommandSpec(
        label="Stop Joy",
        log_key="joy",
        kind="stop",
        note="GUI で起動した joy プロセスを終了します (Ctrl+C 相当)。",
    ),
    CommandSpec(
        label="Restart Joy",
        command="bash -lc \"pkill -f 'ros2 run joy joy_node' || true; ./joy.bash\"",
        log_key="joy",
        stop_before=True,
        note="joy ノードを再起動します。",
    ),
]

SPEC_MAP: Dict[str, CommandSpec] = {spec.label: spec for spec in COMMANDS}

ROW_LAYOUT = [
    ("Zenoh", ["Start Zenoh", "Stop Zenoh", "Restart Zenoh"]),
    ("RViz", ["Start RViz", "Stop RViz", "Restart RViz"]),
    ("Joy", ["Start Joy", "Stop Joy", "Restart Joy"]),
]

STOP_ALL_LABEL = "Stop All"

LOG_AREAS = {
    "zenoh": "Zenoh Log",
    "rviz": "RViz Log",
    "joy": "Joy Log",
}

class RemoteGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Remote Vehicle Helper")

        if not REMOTE_DIR.exists():
            messagebox.showerror(
                "Configuration error",
                f"Remote directory not found: {REMOTE_DIR}",
            )
            raise SystemExit(1)

        self.vehicle_id_var = tk.StringVar(value=DEFAULT_VEHICLE_ID)
        self.username_var = tk.StringVar(value=DEFAULT_USERNAME)

        self.processes: Dict[str, subprocess.Popen[str]] = {}
        self.process_threads: Dict[str, threading.Thread] = {}
        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()

        self._build_ui()
        self.root.after(100, self._poll_log_queue)

    def _build_ui(self) -> None:
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=12, pady=8)

        ttk.Label(top_frame, text="Vehicle ID:").pack(side=tk.LEFT)
        vehicle_entry = ttk.Entry(top_frame, textvariable=self.vehicle_id_var, width=12)
        vehicle_entry.pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top_frame, text="SSH User:").pack(side=tk.LEFT)
        username_entry = ttk.Entry(top_frame, textvariable=self.username_var, width=14)
        username_entry.pack(side=tk.LEFT, padx=(4, 12))

        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(fill=tk.X, padx=12, pady=(0, 8))

        self.directory_label = ttk.Label(preview_frame, text="Directory: -")
        self.directory_label.pack(anchor=tk.W)
        self.command_label = ttk.Label(preview_frame, text="Command: -")
        self.command_label.pack(anchor=tk.W)
        self.note_label = ttk.Label(preview_frame, text="Note: -", foreground="#555555")
        self.note_label.pack(anchor=tk.W)

        button_container = ttk.Frame(self.root)
        button_container.pack(fill=tk.X, padx=12, pady=4)

        for label, buttons in ROW_LAYOUT:
            row_frame = ttk.Frame(button_container)
            row_frame.pack(fill=tk.X, pady=4)
            ttk.Label(row_frame, text=label, width=8).pack(side=tk.LEFT, padx=(0, 8))
            for btn_label in buttons:
                spec = SPEC_MAP[btn_label]
                ttk.Button(
                    row_frame,
                    text=spec.label,
                    command=lambda s=spec: self._handle_command(s),
                    width=16,
                ).pack(side=tk.LEFT, padx=4)

        stop_all_spec = SPEC_MAP.get(STOP_ALL_LABEL)
        if stop_all_spec:
            stop_all_frame = ttk.Frame(button_container)
            stop_all_frame.pack(fill=tk.X, pady=4)
            ttk.Label(stop_all_frame, text="All", width=8).pack(side=tk.LEFT, padx=(0, 8))
            ttk.Button(
                stop_all_frame,
                text=stop_all_spec.label,
                command=lambda s=stop_all_spec: self._handle_command(s),
                width=16,
            ).pack(side=tk.LEFT, padx=4)

        logs_frame = ttk.Frame(self.root)
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(4, 12))

        self.log_widgets: Dict[str, ScrolledText] = {}
        for idx, (key, title) in enumerate(LOG_AREAS.items()):
            frame = ttk.LabelFrame(logs_frame, text=title)
            frame.grid(row=0, column=idx, padx=4, pady=4, sticky=tk.NSEW)
            logs_frame.columnconfigure(idx, weight=1)

            text_widget = ScrolledText(frame, height=18, width=40, state=tk.DISABLED)
            text_widget.pack(fill=tk.BOTH, expand=True)
            self.log_widgets[key] = text_widget

        logs_frame.rowconfigure(0, weight=1)

    def _handle_stop_single(self, log_key: str) -> None:
        if not self._process_running(log_key):
            self._append_log(log_key, '[no running process]\n')
            return
        self._append_log(log_key, '[stop requested]\n')
        self._stop_process(log_key)

    def _stop_all_processes(self) -> None:
        for log_key in list(self.processes.keys()):
            self._append_log(log_key, '[stop requested]\n')
            self._stop_process(log_key)

    def _handle_command(self, spec: CommandSpec) -> None:
        vehicle_id = self.vehicle_id_var.get().strip()
        username = self.username_var.get().strip()

        if spec.requires_vehicle and not vehicle_id:
            messagebox.showwarning("入力不足", "Vehicle ID を指定してください。")
            return

        if spec.requires_username and not username:
            messagebox.showwarning("入力不足", "SSH User を指定してください。")
            return

        if spec.kind == "stop_all":
            self._update_preview(REMOTE_DIR, "[Stop All]", spec.note or "")
            self._stop_all_processes()
            return

        if spec.kind == "stop":
            if not spec.log_key:
                return
            self._update_preview(REMOTE_DIR, f"[Stop] {spec.log_key}", spec.note or "")
            self._handle_stop_single(spec.log_key)
            return

        command_text = spec.render(vehicle_id, username)
        working_dir = REMOTE_DIR
        note = spec.note or ""
        self._update_preview(working_dir, command_text, note)

        log_key = spec.log_key
        if log_key is None:
            return

        if spec.stop_before:
            self._stop_process(log_key)

        if self._process_running(log_key):
            messagebox.showinfo(
                "Process running",
                f"{LOG_AREAS.get(log_key, log_key)} でコマンドが実行中です。先に停止してください。",
            )
            return

        try:
            process = subprocess.Popen(
                ["bash", "-lc", command_text],
                cwd=str(working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            messagebox.showerror("Command error", f"bash が見つかりませんでした。")
            return
        except Exception as exc:  # pragma: no cover - defensive
            messagebox.showerror("Command error", str(exc))
            return

        thread = threading.Thread(
            target=self._stream_output,
            args=(log_key, process),
            daemon=True,
        )
        self.processes[log_key] = process
        self.process_threads[log_key] = thread
        thread.start()
        self._append_log(log_key, f"$ {command_text}\n")

    def _stream_output(self, log_key: str, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in iter(process.stdout.readline, ""):
            self.log_queue.put((log_key, line))
        process.wait()
        exit_msg = f"[process exited with code {process.returncode}]\n"
        self.log_queue.put((log_key, exit_msg))
        self.processes.pop(log_key, None)
        self.process_threads.pop(log_key, None)

    def _stop_process(self, log_key: str) -> None:
        process = self.processes.pop(log_key, None)
        self.process_threads.pop(log_key, None)
        if not process:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        self._append_log(log_key, "[process terminated]\n")

    def _process_running(self, log_key: str) -> bool:
        proc = self.processes.get(log_key)
        return proc is not None and proc.poll() is None

    def _update_preview(self, working_dir: Path, command: str, note: str) -> None:
        self.directory_label.config(text=f"Directory: {working_dir}")
        self.command_label.config(text=f"Command: {command}")
        self.note_label.config(text=f"Note: {note}" if note else "Note: -")

    def _append_log(self, log_key: str, text: str) -> None:
        widget = self.log_widgets.get(log_key)
        if not widget:
            return
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, text)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _poll_log_queue(self) -> None:
        while True:
            try:
                log_key, line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log(log_key, line)
        self.root.after(100, self._poll_log_queue)

def main() -> None:
    root = tk.Tk()
    app = RemoteGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
