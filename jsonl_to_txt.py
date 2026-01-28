import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

class JsonlToTxtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JSONL to TXT Converter")
        self.root.geometry("600x400")
        
        self.selected_files = []
        self._setup_ui()

    def _setup_ui(self):
        # --- File Selection ---
        frame = ttk.LabelFrame(self.root, text="Input Files", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="Select JSONL Result Files", command=self.select_files).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear List", command=self.clear_list).pack(side="left", padx=5)

        self.listbox = tk.Listbox(frame, height=10)
        self.listbox.pack(fill="both", expand=True, pady=5)
        
        # --- Convert Button ---
        action_frame = ttk.Frame(self.root, padding=10)
        action_frame.pack(fill="x")
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(action_frame, textvariable=self.status_var).pack(side="left")
        
        ttk.Button(action_frame, text="CONVERT TO TXT", command=self.convert).pack(side="right")

    def select_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("JSONL files", "*.jsonl")])
        for path in paths:
            if path not in self.selected_files:
                self.selected_files.append(path)
                self.listbox.insert(tk.END, Path(path).name)

    def clear_list(self):
        self.selected_files = []
        self.listbox.delete(0, tk.END)

    def convert(self):
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select at least one JSONL file.")
            return
        
        # Ask where to save
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Save Output Text"
        )
        if not save_path:
            return

        self.status_var.set("Processing...")
        self.root.update()

        try:
            full_text = self._process_files(self.selected_files)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            
            self.status_var.set("Done!")
            messagebox.showinfo("Success", f"Text saved to:\n{save_path}")
            
        except Exception as e:
            self.status_var.set("Error")
            messagebox.showerror("Error", str(e))

    def _process_files(self, file_paths):
        """Reads JSONL, extracts content, and sorts by page ID."""
        records = []
        
        # 1. Read all files
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        records.append(data)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid line in {path}")

        # 2. Extract Content & IDs
        extracted = []
        for item in records:
            # OpenAI Batch structure
            custom_id = item.get("custom_id")
            
            # Navigate complex response structure
            try:
                # Standard Batch API response location
                response_body = item.get("response", {}).get("body", {})
                if not response_body: continue
                
                choices = response_body.get("choices", [])
                if not choices: continue
                
                content = choices[0].get("message", {}).get("content", "")
                
                if custom_id and content:
                    extracted.append((custom_id, content))
            except AttributeError:
                continue

        if not extracted:
            raise ValueError("No valid content found in these files. Are they OpenAI Batch results?")

        # 3. Sort by Custom ID (Crucial for page order)
        # This ensures page_0001 comes before page_0002 regardless of processing order
        extracted.sort(key=lambda x: x[0])

        # 4. Join text
        final_text = "\n\n".join([text for _, text in extracted])
        return final_text

if __name__ == "__main__":
    root = tk.Tk()
    app = JsonlToTxtApp(root)
    root.mainloop()