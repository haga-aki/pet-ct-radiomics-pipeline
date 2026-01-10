#!/usr/bin/env python
"""
PET/CT Radiomics Analysis GUI Launcher
======================================
A simple GUI for running radiomics analysis
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import threading
from pathlib import Path
import sys

# Project base directory
BASE_DIR = Path(__file__).parent
# Use current Python interpreter
PYTHON_PATH = sys.executable


class RadiomicsLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PET/CT Radiomics Analysis Tool")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # Style configuration
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Big.TButton', font=('Helvetica', 11), padding=10)

        self.create_widgets()
        self.process = None

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="PET/CT Radiomics Integrated Analysis Tool",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # Full analysis button
        self.full_btn = ttk.Button(
            button_frame,
            text="Full Analysis (Automated)",
            style='Big.TButton',
            command=self.run_full_analysis
        )
        self.full_btn.pack(fill=tk.X, pady=5)

        # Visualization only button
        self.viz_btn = ttk.Button(
            button_frame,
            text="Visualization Only",
            style='Big.TButton',
            command=self.run_visualization_only
        )
        self.viz_btn.pack(fill=tk.X, pady=5)

        # Force reprocess button
        self.force_btn = ttk.Button(
            button_frame,
            text="Force Reprocess (All Data)",
            style='Big.TButton',
            command=self.run_force_reprocess
        )
        self.force_btn.pack(fill=tk.X, pady=5)

        # Open results folder button
        self.open_btn = ttk.Button(
            button_frame,
            text="Open Results Folder",
            style='Big.TButton',
            command=self.open_results_folder
        )
        self.open_btn.pack(fill=tk.X, pady=5)

        # Separator
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=15)

        # Log display area
        log_label = ttk.Label(main_frame, text="Execution Log:", style='Header.TLabel')
        log_label.pack(anchor=tk.W)

        self.log_area = scrolledtext.ScrolledText(
            main_frame,
            height=15,
            font=('Monaco', 10),
            bg='#1e1e1e',
            fg='#d4d4d4',
            insertbackground='white'
        )
        self.log_area.pack(fill=tk.BOTH, expand=True, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Idle")
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(status_frame, text="Status: ").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT)

    def log(self, message):
        """Add message to log area"""
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.root.update_idletasks()

    def set_buttons_state(self, state):
        """Change button states"""
        self.full_btn['state'] = state
        self.viz_btn['state'] = state
        self.force_btn['state'] = state

    def run_script(self, script_name, args=None):
        """Execute a script"""
        def execute():
            self.set_buttons_state('disabled')
            self.progress.start()
            self.status_var.set("Running...")

            script_path = BASE_DIR / script_name
            cmd = [PYTHON_PATH, str(script_path)]
            if args:
                cmd.extend(args)

            self.log(f"\n{'='*50}")
            self.log(f"Executing: {script_name} {' '.join(args or [])}")
            self.log(f"{'='*50}\n")

            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                for line in iter(self.process.stdout.readline, ''):
                    self.log(line.rstrip())

                self.process.wait()

                if self.process.returncode == 0:
                    self.log("\nProcessing completed successfully")
                    self.status_var.set("Complete")
                    messagebox.showinfo("Complete", "Processing completed successfully!")
                else:
                    self.log(f"\nError occurred (code: {self.process.returncode})")
                    self.status_var.set("Error")

            except Exception as e:
                self.log(f"\nException: {e}")
                self.status_var.set("Error")
                messagebox.showerror("Error", f"An error occurred during execution:\n{e}")
            finally:
                self.progress.stop()
                self.set_buttons_state('normal')
                self.process = None

        # Execute in background thread
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()

    def run_full_analysis(self):
        """Run full analysis"""
        self.log_area.delete(1.0, tk.END)
        self.run_script("run_full_analysis.py")

    def run_visualization_only(self):
        """Run visualization only"""
        self.log_area.delete(1.0, tk.END)
        self.run_script("run_full_analysis.py", ["--visualize-only"])

    def run_force_reprocess(self):
        """Force reprocess all data"""
        if messagebox.askyesno("Confirm", "Force reprocess all data?\nThis may take a while."):
            self.log_area.delete(1.0, tk.END)
            self.run_script("run_full_analysis.py", ["--force"])

    def open_results_folder(self):
        """Open results folder"""
        results_dir = BASE_DIR / "analysis_results"
        if results_dir.exists():
            subprocess.run(["open", str(results_dir)])
        else:
            messagebox.showwarning("Warning", "Results folder does not exist yet.\nPlease run the analysis first.")


def main():
    root = tk.Tk()
    app = RadiomicsLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
