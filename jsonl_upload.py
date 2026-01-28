import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Try to import openai; warn if missing
try:
    from openai import OpenAI
except ImportError:
    messagebox.showerror(
        "Missing Library",
        "The 'openai' library is not installed.\nPlease run: pip install openai"
    )
    OpenAI = None

class BatchUploaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenAI Batch Uploader - Detailed View")
        self.root.geometry("900x600")
        self.root.configure(bg="#f0f0f0")

        self.api_key_var = tk.StringVar()
        self.is_processing = False
        
        # Determine an icon if available (optional)
        try:
            # Set a generic icon or ignore
            pass
        except Exception:
            pass

        self._setup_ui()

    def _setup_ui(self):
        # --- Top Section: API Key ---
        top_frame = ttk.LabelFrame(self.root, text="Configuration", padding=10)
        top_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(top_frame, text="OpenAI API Key:").pack(side="left")
        self.api_entry = ttk.Entry(top_frame, textvariable=self.api_key_var, show="*", width=50)
        self.api_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # --- Middle Section: File Table (Treeview) ---
        tree_frame = ttk.LabelFrame(self.root, text="File Queue", padding=10)
        tree_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Define columns
        columns = ("filename", "size", "status", "details")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended")
        
        # Define headings
        self.tree.heading("filename", text="Filename")
        self.tree.heading("size", text="Size (MB)")
        self.tree.heading("status", text="Status")
        self.tree.heading("details", text="Result / Error")

        # Define column widths
        self.tree.column("filename", width=250, anchor="w")
        self.tree.column("size", width=80, anchor="e")
        self.tree.column("status", width=150, anchor="center")
        self.tree.column("details", width=300, anchor="w")

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Bottom Section: Actions ---
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="Add Files...", command=self.add_files).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Add Folder...", command=self.add_folder).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear List", command=self.clear_list).pack(side="left", padx=5)
        
        self.upload_btn = ttk.Button(btn_frame, text="START UPLOAD", command=self.start_processing_thread)
        self.upload_btn.pack(side="right", padx=5)

        # Status Bar
        self.status_label = ttk.Label(self.root, text="Ready", relief="sunken", anchor="w")
        self.status_label.pack(fill="x")

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("JSONL files", "*.jsonl")])
        for f in files:
            self._insert_file(f)

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            path = Path(folder)
            for f in path.glob("*.jsonl"):
                self._insert_file(str(f))

    def _insert_file(self, filepath):
        # Check if already exists to avoid duplicates
        for item in self.tree.get_children():
            if self.tree.item(item)['values'][0] == os.path.basename(filepath):
                return

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        # We store the full path in 'tags' or a hidden mapping, 
        # but here we can just use the item ID as the map key or similar. 
        # Simplest: Insert and keep a reference.
        self.tree.insert(
            "", 
            "end", 
            iid=filepath,  # Use full path as the Item ID
            values=(os.path.basename(filepath), f"{size_mb:.2f}", "Pending", "")
        )

    def clear_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def start_processing_thread(self):
        if not self.api_key_var.get().strip():
            messagebox.showerror("Error", "Please enter your OpenAI API Key.")
            return
        
        children = self.tree.get_children()
        if not children:
            messagebox.showerror("Error", "No files in the queue.")
            return

        self.is_processing = True
        self.upload_btn.config(state="disabled")
        
        # Start the worker thread
        threading.Thread(target=self._process_queue, args=(children,), daemon=True).start()

    def _update_row(self, item_id, status, details=""):
        # Helper to update UI safely from thread
        self.root.after(0, lambda: self.tree.set(item_id, "status", status))
        self.root.after(0, lambda: self.tree.set(item_id, "details", details))

    def _update_status_bar(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))

    def _process_queue(self, item_ids):
        api_key = self.api_key_var.get().strip()
        try:
            client = OpenAI(api_key=api_key)
        except Exception as e:
            self._update_status_bar(f"Error initializing client: {e}")
            self.root.after(0, lambda: self.upload_btn.config(state="normal"))
            return

        total = len(item_ids)
        success_count = 0

        for index, item_id in enumerate(item_ids, start=1):
            filepath = item_id # We used filepath as iid
            filename = os.path.basename(filepath)
            
            self._update_status_bar(f"Processing {index}/{total}: {filename}")
            
            # 1. Uploading
            self._update_row(item_id, "Uploading...", "Sending file to OpenAI...")
            try:
                with open(filepath, "rb") as f:
                    file_response = client.files.create(file=f, purpose="batch")
                
                file_id = file_response.id
                self._update_row(item_id, "File Uploaded", f"ID: {file_id}")

                # 2. Creating Batch
                self._update_row(item_id, "Creating Batch...", f"File ID: {file_id}")
                
                batch_response = client.batches.create(
                    input_file_id=file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h"
                )
                
                batch_id = batch_response.id
                self._update_row(item_id, "DONE", f"Batch ID: {batch_id}")
                success_count += 1

            except Exception as e:
                error_msg = str(e)
                # Simplify error message for UI
                if "401" in error_msg: 
                    error_msg = "Invalid API Key"
                elif "429" in error_msg:
                    error_msg = "Rate Limit Exceeded"
                
                self._update_row(item_id, "FAILED", error_msg)

        self._update_status_bar(f"Completed. {success_count}/{total} successful.")
        self.root.after(0, lambda: self.upload_btn.config(state="normal"))
        self.root.after(0, lambda: messagebox.showinfo("Batch Upload", f"Finished processing {total} files."))

if __name__ == "__main__":
    root = tk.Tk()
    # Set window icon if it exists in standard paths (optional)
    # try: root.iconbitmap("app_icon.ico") 
    # except: pass
    app = BatchUploaderApp(root)
    root.mainloop()