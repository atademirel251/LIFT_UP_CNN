import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image
import os
import multiprocessing
import threading
import queue

class ConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Directory Converter")
        self.root.minsize(400, 200)
        self.root.resizable(True, True)

        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.input_dir.trace_add('write', self.check_paths)
        self.output_dir.trace_add('write', self.check_paths)
        self.overwrite_enabled = tk.BooleanVar(value=True)
        self.bg_color = ((82, 82, 82), '#525252')
        self.fg_color = ((255, 255, 0), '#ffff00')
        
        # Main container
        self.main_frame = ttk.Frame(self.root, padding=20)
        self.main_frame.pack(fill='both', expand=True)
        
        # Input directory
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.input_frame, text="Input Directory:").pack(side='left', padx=(0, 10))
        self.input_entry = ttk.Entry(self.input_frame, textvariable=self.input_dir, width=30)
        self.input_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(self.input_frame, text="Browse", command=self.select_input_dir).pack(side='left')
        
        # Output directory
        self.output_frame = ttk.Frame(self.main_frame)
        self.output_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.output_frame, text="Output Directory:").pack(side='left', padx=(0, 10))
        self.output_entry = ttk.Entry(self.output_frame, textvariable=self.output_dir, width=30)
        self.output_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        ttk.Button(self.output_frame, text="Browse", command=self.select_output_dir).pack(side='left')

        # Overwrite toggle
        self.overwrite_frame = ttk.Frame(self.main_frame)
        self.overwrite_frame.pack(fill='x', pady=5)
        self.overwrite_check = ttk.Checkbutton(self.overwrite_frame, text="Overwrite Existing Files", variable=self.overwrite_enabled)
        self.overwrite_check.pack(side="left")

        # Background Color Chooser
        self.bg_color_frame = ttk.Frame(self.main_frame)
        self.bg_color_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.bg_color_frame, text="Background Color:").pack(side='left', padx=(0, 10))
        self.bg_color_indicator = ttk.Label(self.bg_color_frame, text="█", foreground=self.bg_color[1])
        self.bg_color_indicator.pack(side='left', padx=(0, 10))
        ttk.Button(self.bg_color_frame, text="Choose", command=self.select_bg_color).pack(side='left')
        

        # Foreground Color Chooser
        self.fg_color_frame = ttk.Frame(self.main_frame)
        self.fg_color_frame.pack(fill='x', pady=5)
        
        ttk.Label(self.fg_color_frame, text="Foreground Color:").pack(side='left', padx=(0, 10))
        self.fg_color_indicator = ttk.Label(self.fg_color_frame, text="█", foreground=self.fg_color[1])
        self.fg_color_indicator.pack(side='left', padx=(0, 10))
        ttk.Button(self.fg_color_frame, text="Choose", command=self.select_fg_color).pack(side='left')

        # Progress bar
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(side="bottom", fill='x')
        self.progressbar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progressbar.pack(side="bottom", fill="x")
        
        # Convert button
        self.convert_btn = ttk.Button(
            self.main_frame,
            text="Convert",
            state='disabled',
            command=self.process_directory_parallel
        )
        self.convert_btn.pack(pady=10)

    def select_input_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.input_dir.set(path)
    
    def select_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)
    
    def check_paths(self, *args):
        if self.input_dir.get() and self.output_dir.get():
            self.convert_btn['state'] = 'normal'
        else:
            self.convert_btn['state'] = 'disabled'

    def select_bg_color(self):
        self.bg_color = colorchooser.askcolor()
        self.bg_color_indicator['foreground'] = self.bg_color[1]
        
    def select_fg_color(self):
        self.fg_color = colorchooser.askcolor()
        self.fg_color_indicator['foreground'] = self.fg_color[1]
    
    @staticmethod
    def convert(args):
        """Worker function for processing an image."""
        file_path, output_path, bg_color, fg_color, queue = args

        try:
            with Image.open(file_path) as img:
                # Convert image to HSV and manipulate channels
                red = img.convert("HSV").getchannel("S").point(lambda p: bg_color[0] if p <= 20 else fg_color[0])
                green = img.convert("HSV").getchannel("S").point(lambda p: bg_color[1] if p <= 20 else fg_color[1])
                blue = img.convert("HSV").getchannel("S").point(lambda p: bg_color[2] if p <= 20 else fg_color[2])
                alpha = img.convert("HSV").getchannel("S").point(lambda p: 128 if p <= 20 else 255)
                new_img = Image.merge(mode="RGBA", bands=(red, green, blue, alpha))

                new_img.save(output_path, 'PNG')

            queue.put(1)  # Notify main thread that one image is done
            return (file_path, None)  # No error

        except Exception as e:
            queue.put(1)  # Notify progress update even if failed
            return (file_path, str(e))  # Return error message

    def process_directory_parallel(self):
        """Process all images in a directory using multiple processes."""
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()

        os.makedirs(output_dir, exist_ok=True)

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
        tasks = []

        overwrite_enabled = self.overwrite_enabled.get()

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(valid_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename.replace("_TopView", ""))
                if not overwrite_enabled and os.path.exists(output_path):
                    continue  # Skip if already exists
                tasks.append((input_path, output_path))

        if not tasks:
            messagebox.showinfo("No Files", "No new images to process.")
            return

        # Set up progress bar
        self.progressbar["value"] = 0
        self.progressbar["maximum"] = len(tasks)

        # Get colors
        bg_color = self.bg_color[0]
        fg_color = self.fg_color[0]

        # Create queue to receive updates from worker processes
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()

        # Prepare tasks with queue reference
        task_data = [(task[0], task[1], bg_color, fg_color, progress_queue) for task in tasks]

        # Start multiprocessing
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        pool.map_async(self.convert, task_data)

        # Start a separate thread to monitor progress updates
        threading.Thread(target=self.update_progress, args=(progress_queue, len(tasks)), daemon=True).start()

    def update_progress(self, progress_queue, total_tasks):
        """Monitors the progress queue and updates the progress bar in real time."""
        completed_tasks = 0
        errors = []

        while completed_tasks < total_tasks:
            try:
                # Block until there's a new item in the queue
                result = progress_queue.get()
                if isinstance(result, tuple) and result[1]:  # If error
                    errors.append(f"{result[0]}: {result[1]}")

                completed_tasks += 1
                self.progressbar["value"] = completed_tasks
                self.root.update_idletasks()  # Force UI update

            except queue.Empty:
                pass

        # Show errors if any
        if errors:
            messagebox.showerror("Errors Occurred", "\n".join(errors))

        messagebox.showinfo("Conversion Complete", "All images processed successfully!")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()