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
    "area/25area.py",
    "area/30area.py", 
    "area/35area.py",
    "pressure/2atm.py",
    "pressure/2_5atm.py",
    "pressure/3atm.py"
]

# Create temp directory
os.makedirs("temp_figures", exist_ok=True)

# Create a wrapper script that will run the plot scripts and capture their output
wrapper_script = """
import sys
import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

# Store original show function
original_show = plt.show

# Replace plt.show() with our capturing function
def capture_show(*args, **kwargs):
    fig = plt.gcf() # Get current figure
    fig.savefig(sys.argv[2], dpi=600)
    print(f"Captured and saved figure to {sys.argv[2]}")
    return original_show(*args, **kwargs)

plt.show = capture_show

# Run the target script
try:
    script_path = sys.argv[1]
    
    # Add script directory to path
    script_dir = os.path.dirname(script_path)
    if script_dir:
        sys.path.insert(0, script_dir)
    
    # Execute the script
    with open(script_path, 'r') as f:
        exec(f.read(), {'__file__': script_path, '__name__': '__main__'})
    
    # If no figure was explicitly shown, check if any were created
    if plt.get_fignums():
        plt.gcf().savefig(sys.argv[2], dpi=600)
        print(f"Saved figure to {sys.argv[2]}")
except Exception as e:
    print(f"Error executing {script_path}: {e}")
"""

# Write the wrapper script to a file
wrapper_path = os.path.join("temp_figures", "wrapper.py")
with open(wrapper_path, 'w') as f:
    f.write(wrapper_script)

# Create the main figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))

# Process each script using the wrapper
for i, script_path in enumerate(scripts):
    print(f"Processing {script_path}...")
    
    # Define output path for this figure
    output_path = os.path.join("temp_figures", f"figure_{i}.png")
    
    # Run the wrapper script with the target script
    result = subprocess.run(
        [sys.executable, wrapper_path, script_path, output_path],
        capture_output=True,
        text=True
    )
    
    # Print output for debugging
    print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    # Load the saved figure if it exists
    if os.path.exists(output_path):
        img = mpimg.imread(output_path)
        
        # Place in the appropriate subplot
        row, col = divmod(i, 3)
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        
        # Removed the subplot title as requested
        
    else:
        print(f"Warning: No figure was created for {script_path}")

# Add global title and row labels
fig.suptitle("Tension vs Surface area", fontsize=16)
#fig.text(0.5, 0.52, "Area Analysis (25/30/35)", ha='center', fontsize=14)
fig.text(0.5, 0.02, "Tension vs Internal Pressure", ha='center', fontsize=14)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("combined_plots.png", dpi=600, bbox_inches='tight')
print("Saved combined_plots.png")

# Clean up temporary files
for file in os.listdir("temp_figures"):
    try:
        os.remove(os.path.join("temp_figures", file))
    except:
        pass
try:
    os.rmdir("temp_figures")
except:
    pass

# Display the result
plt.show()