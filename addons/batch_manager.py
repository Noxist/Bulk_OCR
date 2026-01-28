import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import datetime

# Try to import openai
try:
    from openai import OpenAI
except ImportError:
    messagebox.showerror("Error", "Please install openai: pip install openai")
    OpenAI = None

class BatchManagerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenAI Batch Manager (Bulk Download Support)")
        self.root.geometry("950x650")
        
        self.api_key_var = tk.StringVar()
        self.client = None

        self._setup_ui()

    def _setup_ui(self):
        # --- Top: API Key ---
        top_frame = ttk.LabelFrame(self.root, text="Connect", padding=10)
        top_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(top_frame, text="OpenAI API Key:").pack(side="left")
        self.api_entry = ttk.Entry(top_frame, textvariable=self.api_key_var, show="*", width=50)
        self.api_entry.pack(side="left", padx=5)
        ttk.Button(top_frame, text="Login / Refresh", command=self.init_client).pack(side="left")

        # --- Main List (Batches) ---
        list_frame = ttk.LabelFrame(self.root, text="Your Recent Batches (Last 50)", padding=10)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        cols = ("id", "created", "status", "output_file")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings", selectmode="extended")
        self.tree.heading("id", text="Batch ID")
        self.tree.heading("created", text="Created At")
        self.tree.heading("status", text="Status")
        self.tree.heading("output_file", text="Output File ID")
        
        self.tree.column("id", width=220)
        self.tree.column("created", width=140)
        self.tree.column("status", width=100)
        self.tree.column("output_file", width=220)
        
        # Bind Ctrl+A to select all
        self.tree.bind("<Control-a>", self.select_all)
        
        self.tree.pack(fill="both", expand=True)

        # --- Bottom: Actions ---
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill="x")
        
        ttk.Button(btn_frame, text="REFRESH LIST", command=self.refresh_batches).pack(side="left", padx=5)
        
        # Right side buttons
        ttk.Button(btn_frame, text="Download Selected", command=self.download_selected).pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Select All Completed", command=self.select_completed).pack(side="right", padx=5)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(fill="x")

    def init_client(self):
        key = self.api_key_var.get().strip()
        if not key:
            messagebox.showerror("Error", "Enter API Key")
            return
        try:
            self.client = OpenAI(api_key=key)
            self.refresh_batches()
            messagebox.showinfo("Success", "Found your batches!")
        except Exception as e:
            messagebox.showerror("Connection Failed", str(e))

    def refresh_batches(self):
        if not self.client: return
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.status_var.set("Fetching batches...")
        self.root.update_idletasks()
        
        try:
            # Increased limit to 50 to see more history
            batches = self.client.batches.list(limit=50)
            
            for b in batches:
                created_str = datetime.datetime.fromtimestamp(b.created_at).strftime('%Y-%m-%d %H:%M')
                out_id = b.output_file_id if b.output_file_id else ""
                self.tree.insert("", "end", values=(b.id, created_str, b.status, out_id))
            
            self.status_var.set(f"Loaded {len(list(batches))} batches.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not list batches: {e}")
            self.status_var.set("Error fetching batches.")

    def select_all(self, event=None):
        """Selects all items in the list."""
        self.tree.selection_set(self.tree.get_children())
        return "break"

    def select_completed(self):
        """Selects only items that are completed and have an output file."""
        self.tree.selection_remove(self.tree.selection())
        for item in self.tree.get_children():
            vals = self.tree.item(item)['values']
            status, out_id = vals[2], vals[3]
            if status == "completed" and out_id:
                self.tree.selection_add(item)

    def download_selected(self):
        selected = self.tree.selection()
        if not selected: 
            messagebox.showwarning("Select One", "Please select at least one batch row.")
            return
        
        # 1. Filter usable batches
        download_queue = []
        for item_id in selected:
            vals = self.tree.item(item_id)['values']
            batch_id, _, status, output_file_id = vals
            
            if status == "completed" and output_file_id:
                download_queue.append((batch_id, output_file_id))
        
        if not download_queue:
            messagebox.showwarning("Not Ready", "None of the selected batches are 'completed' with downloadable files.")
            return

        # 2. Determine Save Location
        if len(download_queue) == 1:
            # Single File: Ask for full path
            batch_id, file_id = download_queue[0]
            default_name = f"batch_output_{batch_id}.jsonl"
            save_path = filedialog.asksaveasfilename(defaultextension=".jsonl", initialfile=default_name)
            if not save_path: return
            
            threading.Thread(target=self._download_worker_single, args=(file_id, save_path), daemon=True).start()
            
        else:
            # Multiple Files: Ask for Directory
            target_dir = filedialog.askdirectory(title="Select Folder to Save Files")
            if not target_dir: return
            
            threading.Thread(target=self._download_worker_bulk, args=(download_queue, target_dir), daemon=True).start()

    def _download_worker_single(self, file_id, path):
        self.status_var.set("Downloading file...")
        try:
            content = self.client.files.content(file_id).text
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.root.after(0, lambda: messagebox.showinfo("Downloaded!", f"Saved file to:\n{path}"))
            self.status_var.set("Ready")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Download Error", str(e)))
            self.status_var.set("Error during download.")

    def _download_worker_bulk(self, queue, directory):
        total = len(queue)
        success = 0
        
        for i, (batch_id, file_id) in enumerate(queue, 1):
            msg = f"Downloading {i}/{total}: {batch_id}..."
            self.root.after(0, lambda m=msg: self.status_var.set(m))
            
            try:
                content = self.client.files.content(file_id).text
                filename = f"batch_output_{batch_id}.jsonl"
                path = os.path.join(directory, filename)
                
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                success += 1
            except Exception as e:
                print(f"Failed to download {batch_id}: {e}")

        final_msg = f"Bulk Download Complete. {success}/{total} files saved."
        self.root.after(0, lambda: self.status_var.set("Ready"))
        self.root.after(0, lambda: messagebox.showinfo("Complete", final_msg))

if __name__ == "__main__":
    root = tk.Tk()
    app = BatchManagerApp(root)
    root.mainloop()