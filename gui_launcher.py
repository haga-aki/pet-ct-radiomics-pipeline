#!/Users/akira/miniforge3/envs/med_ai/bin/python
"""
PET/CT Radiomics 解析 GUI ランチャー
====================================
シンプルなGUIで解析を実行できます
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import subprocess
import threading
from pathlib import Path
import sys

# プロジェクトのベースディレクトリ
BASE_DIR = Path(__file__).parent
PYTHON_PATH = "/Users/akira/miniforge3/envs/med_ai/bin/python"


class RadiomicsLauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PET/CT Radiomics 解析ツール")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # スタイル設定
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Big.TButton', font=('Helvetica', 11), padding=10)

        self.create_widgets()
        self.process = None

    def create_widgets(self):
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # タイトル
        title_label = ttk.Label(
            main_frame,
            text="PET/CT Radiomics 統合解析ツール",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))

        # ボタンフレーム
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        # フル解析ボタン
        self.full_btn = ttk.Button(
            button_frame,
            text="🔬 フル解析（全自動）",
            style='Big.TButton',
            command=self.run_full_analysis
        )
        self.full_btn.pack(fill=tk.X, pady=5)

        # 可視化のみボタン
        self.viz_btn = ttk.Button(
            button_frame,
            text="📊 可視化のみ",
            style='Big.TButton',
            command=self.run_visualization_only
        )
        self.viz_btn.pack(fill=tk.X, pady=5)

        # 強制再処理ボタン
        self.force_btn = ttk.Button(
            button_frame,
            text="🔄 強制再処理（全データ）",
            style='Big.TButton',
            command=self.run_force_reprocess
        )
        self.force_btn.pack(fill=tk.X, pady=5)

        # 結果フォルダを開くボタン
        self.open_btn = ttk.Button(
            button_frame,
            text="📁 結果フォルダを開く",
            style='Big.TButton',
            command=self.open_results_folder
        )
        self.open_btn.pack(fill=tk.X, pady=5)

        # セパレータ
        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=15)

        # ログ表示エリア
        log_label = ttk.Label(main_frame, text="実行ログ:", style='Header.TLabel')
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

        # ステータスバー
        self.status_var = tk.StringVar(value="待機中")
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(status_frame, text="状態: ").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT)

        # プログレスバー
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT)

    def log(self, message):
        """ログエリアにメッセージを追加"""
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.root.update_idletasks()

    def set_buttons_state(self, state):
        """ボタンの状態を変更"""
        self.full_btn['state'] = state
        self.viz_btn['state'] = state
        self.force_btn['state'] = state

    def run_script(self, script_name, args=None):
        """スクリプトを実行"""
        def execute():
            self.set_buttons_state('disabled')
            self.progress.start()
            self.status_var.set("実行中...")

            script_path = BASE_DIR / script_name
            cmd = [PYTHON_PATH, str(script_path)]
            if args:
                cmd.extend(args)

            self.log(f"\n{'='*50}")
            self.log(f"実行: {script_name} {' '.join(args or [])}")
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
                    self.log("\n✅ 処理が正常に完了しました")
                    self.status_var.set("完了")
                    messagebox.showinfo("完了", "処理が正常に完了しました！")
                else:
                    self.log(f"\n❌ エラーが発生しました (code: {self.process.returncode})")
                    self.status_var.set("エラー")

            except Exception as e:
                self.log(f"\n❌ 例外: {e}")
                self.status_var.set("エラー")
                messagebox.showerror("エラー", f"実行中にエラーが発生しました:\n{e}")
            finally:
                self.progress.stop()
                self.set_buttons_state('normal')
                self.process = None

        # バックグラウンドスレッドで実行
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()

    def run_full_analysis(self):
        """フル解析を実行"""
        self.log_area.delete(1.0, tk.END)
        self.run_script("run_full_analysis.py")

    def run_visualization_only(self):
        """可視化のみ実行"""
        self.log_area.delete(1.0, tk.END)
        self.run_script("run_full_analysis.py", ["--visualize-only"])

    def run_force_reprocess(self):
        """強制再処理"""
        if messagebox.askyesno("確認", "全データを強制的に再処理しますか？\nこれには時間がかかる場合があります。"):
            self.log_area.delete(1.0, tk.END)
            self.run_script("run_full_analysis.py", ["--force"])

    def open_results_folder(self):
        """結果フォルダを開く"""
        results_dir = BASE_DIR / "analysis_results"
        if results_dir.exists():
            subprocess.run(["open", str(results_dir)])
        else:
            messagebox.showwarning("警告", "結果フォルダがまだ存在しません。\nまず解析を実行してください。")


def main():
    root = tk.Tk()
    app = RadiomicsLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
