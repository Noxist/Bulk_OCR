import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import importlib.util

OpenAI = None
if importlib.util.find_spec("openai"):
    from openai import OpenAI


class AiTextRefinerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Refiner (Post-Processing)")
        self.root.geometry("800x650")
        self.api_key_var = tk.StringVar()
        self.model_var = tk.StringVar(value="gpt-4o-mini")
        self.status_var = tk.StringVar(value="Ready")
        self.input_file_path = None
        self.is_running = False
        self._setup_ui()

    def _setup_ui(self):
        config_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(config_frame, text="API Key:").pack(side="left")
        ttk.Entry(config_frame, textvariable=self.api_key_var, show="*", width=30).pack(
            side="left", padx=5
        )
        ttk.Label(config_frame, text="Model:").pack(side="left", padx=(10, 0))
        ttk.Entry(config_frame, textvariable=self.model_var, width=15).pack(
            side="left", padx=5
        )

        file_frame = ttk.LabelFrame(self.root, text="Input File (Raw Text)", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side="left", fill="x", expand=True)
        ttk.Button(file_frame, text="Select File...", command=self.select_file).pack(
            side="right"
        )

        prompt_frame = ttk.LabelFrame(self.root, text="System Prompt", padding=10)
        prompt_frame.pack(fill="both", expand=True, padx=10, pady=5)
        default_prompt = (
            "You are a professional editor. Your task is FORMATTING ONLY.\n"
            "Rules:\n"
            "1. Merge lines that were incorrectly split by page breaks.\n"
            "2. Mark real paragraphs with a blank line.\n"
            "3. Mark headings with Markdown (#).\n"
            "4. DO NOT change the content or spelling.\n"
            "5. If text is unreadable/unsure, insert [[??]].\n"
            "6. Return only the formatted text."
        )
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=8)
        self.prompt_text.insert("1.0", default_prompt)
        self.prompt_text.pack(fill="both", expand=True)

        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill="x")
        self.progress = ttk.Progressbar(action_frame, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.start_btn = ttk.Button(
            action_frame, text="START PROCESSING", command=self.start_processing
        )
        self.start_btn.pack(side="right")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x"
        )

    def select_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            self.input_file_path = path
            self.file_label.config(text=path)

    def start_processing(self):
        if not self.api_key_var.get() or not self.input_file_path:
            messagebox.showerror("Error", "Missing API Key or File")
            return
        self.is_running = True
        self.start_btn.config(state="disabled")
        threading.Thread(
            target=self._worker,
            args=(self.prompt_text.get("1.0", "end-1c"),),
            daemon=True,
        ).start()

    def _smart_split(self, text, target_chars=12000):
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        current_char_count = 0
        sentence_endings = (".", "!", "?", '"', "”", "»")

        for line in lines:
            line_len = len(line) + 1
            if current_char_count + line_len > target_chars:
                # Soft limit reached, check for sentence ending or use hard limit buffer
                if not current_chunk:
                    chunks.append(line)
                    continue

                # Check if previous line ended a sentence or if we are still within hard limit buffer (2000 chars)
                is_sentence_end = current_chunk[-1].strip().endswith(sentence_endings)
                within_hard_limit = current_char_count + line_len < target_chars + 2000

                if not is_sentence_end and within_hard_limit:
                    current_chunk.append(line)
                    current_char_count += line_len
                    continue

                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_char_count = line_len
            else:
                current_chunk.append(line)
                current_char_count += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    def _worker(self, system_prompt):
        try:
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK is not installed.")
            client = OpenAI(api_key=self.api_key_var.get().strip())
            with open(self.input_file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            self._update_status("Smart Chunking...")
            chunks = self._smart_split(raw_text)
            refined_chunks = []

            for i, chunk in enumerate(chunks, 1):
                if not self.is_running:
                    break
                self._update_status(f"Processing chunk {i}/{len(chunks)}...")
                response = client.chat.completions.create(
                    model=self.model_var.get(),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk},
                    ],
                    temperature=0.3,
                )
                refined_chunks.append(response.choices[0].message.content)
                self.progress.configure(value=(i / len(chunks)) * 100)

            if self.is_running:
                save_path = self.input_file_path.replace(".txt", "_refined.txt")
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(refined_chunks))
                messagebox.showinfo("Success", f"Saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.start_btn.config(state="normal"))
            self._update_status("Ready")

    def _update_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))


if __name__ == "__main__":
    root = tk.Tk()
    app = AiTextRefinerApp(root)
    root.mainloop()
