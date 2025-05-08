import io
import os
import sys
import subprocess
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List scripts in order
scripts = [
    "EVERYTHING/area/25area.py",
    "EVERYTHING/area/30area.py", 
    "EVERYTHING/area/35area.py",
    "EVERYTHING/pressure/2atm.py",
    "EVERYTHING/pressure/2_5atm.py",
    "EVERYTHING/pressure/3atm.py"
]

# Create temp directory
os.makedirs("temp_figures", exist_ok=True)

# Create a wrapper script that will run the plot scripts and capture their output
wrapper_script = """
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# store original show
_original_show = plt.show

def capture_show(*args, **kwargs):
    fig = plt.gcf()
    fig.savefig(sys.argv[2], dpi=600)
    print(f"Captured and saved figure to {sys.argv[2]}")
    return _original_show(*args, **kwargs)

plt.show = capture_show

try:
    script_path = sys.argv[1]
    # add script's directory to PYTHONPATH so imports work
    script_dir = os.path.dirname(script_path)
    if script_dir:
        sys.path.insert(0, script_dir)

    # execute the script
    with open(script_path, 'r') as f:
        exec(f.read(), {'__file__': script_path, '__name__': '__main__'})

    # if the script never called plt.show(), still save its last figure
    if plt.get_fignums():
        plt.gcf().savefig(sys.argv[2], dpi=600)
        print(f"Saved figure to {sys.argv[2]}")
except Exception as e:
    print(f"Error executing {script_path}: {e}")
"""

# write wrapper to disk
wrapper_path = os.path.join("temp_figures", "wrapper.py")
with open(wrapper_path, 'w') as f:
    f.write(wrapper_script)

# Create the main figure with subplots (2 rows × 3 cols)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))

for i, script_path in enumerate(scripts):
    print(f"Processing {script_path}...")
    
    # define a high-res PNG for each subplot
    output_path = os.path.join("temp_figures", f"figure_{i}.png")
    
    # run the wrapper to produce that PNG
    result = subprocess.run(
        [sys.executable, wrapper_path, script_path, output_path],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    # if PNG was created, load & display it
    if os.path.exists(output_path):
        img = mpimg.imread(output_path)
        row, col = divmod(i, 3)
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
    else:
        print(f"Warning: No figure was created for {script_path}")

# Add your global title and footer
fig.suptitle("Tension vs Surface area", fontsize=16)
fig.text(0.5, 0.02, "Tension vs Internal Pressure", ha='center', fontsize=14)

# Lay out nicely, then save as a vector‐PDF
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("combined_plots.pdf", dpi=600, bbox_inches='tight')
print("Saved combined_plots.pdf")

# Clean up temporary files
for fn in os.listdir("temp_figures"):
    try:
        os.remove(os.path.join("temp_figures", fn))
    except OSError:
        pass
try:
    os.rmdir("temp_figures")
except OSError:
    pass

# (Optional) pop up the PDF in your default viewer if you like:
# os.system("open combined_plots.pdf")  # macOS
# os.system("xdg-open combined_plots.pdf")  # Linux

