import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import importlib.util
import json
import base64
import os

OpenAI = None
if importlib.util.find_spec("openai"):
    from openai import OpenAI

class AiTextRefinerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Text Refiner & Batch Preparer")
        self.root.geometry("900x750")
        
        self.api_key_var = tk.StringVar()
        self.model_var = tk.StringVar(value="gpt-4o-mini")
        self.status_var = tk.StringVar(value="Bereit")
        
        self.input_file_path = None
        self.toc_file_paths = []  # Liste für TOC Bilder
        self.is_running = False
        
        self._setup_ui()

    def _setup_ui(self):
        # --- Config ---
        config_frame = ttk.LabelFrame(self.root, text="Konfiguration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(config_frame, text="API Key:").pack(side="left")
        ttk.Entry(config_frame, textvariable=self.api_key_var, show="*", width=30).pack(side="left", padx=5)
        
        ttk.Label(config_frame, text="Modell:").pack(side="left", padx=(10, 0))
        ttk.Entry(config_frame, textvariable=self.model_var, width=15).pack(side="left", padx=5)

        # --- Input Files (Text & TOC) ---
        files_frame = ttk.LabelFrame(self.root, text="Eingabedateien", padding=10)
        files_frame.pack(fill="x", padx=10, pady=5)
        
        # Text File
        file_row = ttk.Frame(files_frame)
        file_row.pack(fill="x", pady=2)
        ttk.Button(file_row, text="1. Textdatei wählen (.txt)", command=self.select_text_file).pack(side="left")
        self.file_label = ttk.Label(file_row, text="Keine Datei ausgewählt", foreground="gray")
        self.file_label.pack(side="left", padx=10)

        # TOC Files
        toc_row = ttk.Frame(files_frame)
        toc_row.pack(fill="x", pady=2)
        ttk.Button(toc_row, text="2. TOC Bilder wählen (optional)", command=self.select_toc_files).pack(side="left")
        self.toc_label = ttk.Label(toc_row, text="0 Bilder ausgewählt", foreground="gray")
        self.toc_label.pack(side="left", padx=10)

        # --- Prompt ---
        prompt_frame = ttk.LabelFrame(self.root, text="System Prompt (Anweisungen)", padding=10)
        prompt_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        default_prompt = (
            "Du bist ein professioneller Lektor. Deine Aufgabe ist REINES FORMATTIEREN.\n"
            "Regeln:\n"
            "1. Verbinde Zeilen, die durch Seitenumbrüche falsch getrennt wurden.\n"
            "2. Markiere echte Absätze mit einer Leerzeile.\n"
            "3. Markiere Überschriften mit Markdown (#).\n"
            "4. Ändere NICHT den Inhalt oder die Rechtschreibung.\n"
            "5. Falls Text unleserlich ist, schreibe [[??]].\n"
            "6. Gib nur den formatierten Text zurück."
        )
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=10)
        self.prompt_text.insert("1.0", default_prompt)
        self.prompt_text.pack(fill="both", expand=True)

        # --- Actions ---
        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill="x")
        
        self.progress = ttk.Progressbar(action_frame, mode="determinate")
        self.progress.pack(side="top", fill="x", pady=(0, 10))
        
        btn_container = ttk.Frame(action_frame)
        btn_container.pack(fill="x")

        # Live Processing Button
        self.start_btn = ttk.Button(
            btn_container, 
            text="DIREKT STARTEN (Live API)", 
            command=lambda: self.start_processing(mode="live")
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # Batch JSONL Button
        self.batch_btn = ttk.Button(
            btn_container, 
            text="ALS BATCH JSONL SPEICHERN (für Upload)", 
            command=lambda: self.start_processing(mode="batch")
        )
        self.batch_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Status Bar
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x")

    def select_text_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            self.input_file_path = path
            self.file_label.config(text=os.path.basename(path), foreground="black")

    def select_toc_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if paths:
            self.toc_file_paths = list(paths)
            self.toc_label.config(text=f"{len(paths)} Bilder ausgewählt", foreground="black")

    def start_processing(self, mode="live"):
        if not self.api_key_var.get() or not self.input_file_path:
            messagebox.showerror("Fehler", "API Key und Textdatei werden benötigt.")
            return
        
        self.is_running = True
        self._toggle_buttons(disabled=True)
        
        threading.Thread(
            target=self._worker,
            args=(self.prompt_text.get("1.0", "end-1c"), mode),
            daemon=True,
        ).start()

    def _toggle_buttons(self, disabled=False):
        state = "disabled" if disabled else "normal"
        self.start_btn.config(state=state)
        self.batch_btn.config(state=state)

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_toc_context(self, client):
        """Liest die TOC Bilder und gibt eine Liste der Kapitel zurück."""
        if not self.toc_file_paths:
            return ""
        
        self._update_status("Analysiere Inhaltsverzeichnis...")
        try:
            content = [
                {"type": "text", "text": "Extrahiere nur die Kapiteltitel aus diesen Bildern. Gib sie als einfache Liste zurück."}
            ]
            
            for img_path in self.toc_file_paths:
                base64_image = self._encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                max_tokens=1000
            )
            extracted_toc = response.choices[0].message.content
            return f"\n\nHIER IST DAS INHALTSVERZEICHNIS (Nutze dies, um Überschriften zu erkennen):\n{extracted_toc}"
        except Exception as e:
            print(f"TOC Error: {e}")
            return ""

    def _smart_split(self, text, target_chars=10000):
        # Größere Chunks für Batch (Kontext besser erhalten)
        lines = text.splitlines()
        chunks = []
        current_chunk = []
        current_char_count = 0
        
        # Etwas Puffer für Überlappung oder saubere Schnitte
        for line in lines:
            line_len = len(line) + 1
            if current_char_count + line_len > target_chars:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_char_count = line_len
            else:
                current_chunk.append(line)
                current_char_count += line_len
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    def _worker(self, base_system_prompt, mode):
        try:
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK nicht installiert.")
            
            client = OpenAI(api_key=self.api_key_var.get().strip())
            
            # 1. TOC Context holen (falls Bilder da sind)
            toc_context = self._get_toc_context(client)
            full_system_prompt = base_system_prompt + toc_context
            
            # 2. Text lesen und splitten
            with open(self.input_file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            self._update_status("Text wird segmentiert...")
            chunks = self._smart_split(raw_text)
            
            if mode == "live":
                self._run_live_mode(client, chunks, full_system_prompt)
            else:
                self._run_batch_mode(chunks, full_system_prompt)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Fehler", str(e)))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self._toggle_buttons(disabled=False))
            self._update_status("Bereit")

    def _run_live_mode(self, client, chunks, system_prompt):
        refined_chunks = []
        for i, chunk in enumerate(chunks, 1):
            if not self.is_running: break
            
            self._update_status(f"Verarbeite Teil {i}/{len(chunks)} (Live)...")
            try:
                response = client.chat.completions.create(
                    model=self.model_var.get(),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk}
                    ],
                    temperature=0.3
                )
                refined_chunks.append(response.choices[0].message.content)
                self.progress.configure(value=(i / len(chunks)) * 100)
            except Exception as e:
                print(f"Error in chunk {i}: {e}")
                refined_chunks.append(f"[[ERROR PROCESSING CHUNK {i}]]\n{chunk}")

        if self.is_running:
            save_path = self.input_file_path.replace(".txt", "_refined.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(refined_chunks))
            self.root.after(0, lambda: messagebox.showinfo("Fertig", f"Gespeichert unter:\n{save_path}"))

    def _run_batch_mode(self, chunks, system_prompt):
        batch_lines = []
        base_name = os.path.splitext(os.path.basename(self.input_file_path))[0]
        
        self._update_status("Erstelle Batch-Datei...")
        
        for i, chunk in enumerate(chunks, 1):
            # Custom ID mit Padding für korrekte Sortierung (part_0001, part_0002...)
            custom_id = f"{base_name}_part_{i:04d}"
            
            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_var.get(),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chunk}
                    ],
                    "temperature": 0.3
                }
            }
            batch_lines.append(json.dumps(req))
            self.progress.configure(value=(i / len(chunks)) * 100)

        save_path = self.input_file_path.replace(".txt", "_batch_request.jsonl")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(batch_lines))
            
        self.root.after(0, lambda: messagebox.showinfo(
            "Batch Erstellt", 
            f"JSONL Datei erstellt:\n{save_path}\n\n"
            "Lade diese nun im 'Upload JSONL Batches' Tool hoch."
        ))

    def _update_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

if __name__ == "__main__":
    root = tk.Tk()
    app = AiTextRefinerApp(root)
    root.mainloop()
