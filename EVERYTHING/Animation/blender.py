import bpy
import math
import bmesh
from mathutils import Vector

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create new collection for our simulation
collection_name = "Bacterial_Cell_Simulation"
if collection_name not in bpy.data.collections:
    simulation_collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(simulation_collection)
else:
    simulation_collection = bpy.data.collections[collection_name]

# Set render engine to Cycles for better material rendering
bpy.context.scene.render.engine = 'CYCLES'

# Animation settings
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = 120
fps = 30
scene.render.fps = fps

# Set the active collection
layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]
bpy.context.view_layer.active_layer_collection = layer_collection

# Constants
initial_radius_pg = 5.0  # Initial radius of PG layer
pg_thickness = 0.4       # Thickness of PG layer
pm_thickness = 0.2       # Thickness of PM layer
max_expansion = 1.5      # Maximum expansion factor
pressure_threshold = 0.5  # Animation progress (0-1) when color change occurs - 2.5 atm equivalent

# Create materials
def create_materials():
    # PG layer material (green)
    pg_material = bpy.data.materials.new(name="PG_Material")
    pg_material.use_nodes = True
    nodes = pg_material.node_tree.nodes
    nodes.clear()
    
    # Create shader nodes
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    # Set PG material properties (green, semi-transparent)
    node_bsdf.inputs['Base Color'].default_value = (0.0, 0.8, 0.2, 1.0)
    node_bsdf.inputs['Roughness'].default_value = 0.1
    # Handle different Blender versions - try different specular input names
    if 'Specular' in node_bsdf.inputs:
        node_bsdf.inputs['Specular'].default_value = 0.5
    elif 'Specular IOR' in node_bsdf.inputs:
        node_bsdf.inputs['Specular IOR'].default_value = 0.5
    node_bsdf.inputs['Alpha'].default_value = 0.8
    
    # Connect nodes
    pg_material.node_tree.links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    
    # PM layer material (white to red with animation)
    pm_material = bpy.data.materials.new(name="PM_Material")
    pm_material.use_nodes = True
    nodes = pm_material.node_tree.nodes
    nodes.clear()
    
    # Create shader nodes for PM material
    node_output = nodes.new(type='ShaderNodeOutputMaterial')
    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_mix = nodes.new(type='ShaderNodeMixRGB')
    node_value = nodes.new(type='ShaderNodeValue')
    
    # Set PM material properties
    node_bsdf.inputs['Roughness'].default_value = 0.1
    # Handle different Blender versions - try different specular input names
    if 'Specular' in node_bsdf.inputs:
        node_bsdf.inputs['Specular'].default_value = 0.5
    elif 'Specular IOR' in node_bsdf.inputs:
        node_bsdf.inputs['Specular IOR'].default_value = 0.5
    node_bsdf.inputs['Alpha'].default_value = 0.9
    
    # Set up color mix (white to red)
    node_mix.blend_type = 'MIX'
    node_mix.inputs['Color1'].default_value = (1.0, 1.0, 1.0, 1.0)  # White
    node_mix.inputs['Color2'].default_value = (1.0, 0.0, 0.0, 1.0)  # Red
    
    # Set up connections
    pm_material.node_tree.links.new(node_mix.outputs['Color'], node_bsdf.inputs['Base Color'])
    pm_material.node_tree.links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
    pm_material.node_tree.links.new(node_value.outputs['Value'], node_mix.inputs['Fac'])
    
    # Store the value node for animation
    return pg_material, pm_material, node_value

# Create a hollow sphere with specified inner and outer radius
def create_hollow_sphere(name, outer_radius, thickness, material, collection):
    # Create outer sphere
    bpy.ops.mesh.primitive_uv_sphere_add(radius=outer_radius, segments=32, ring_count=16)
    outer_sphere = bpy.context.active_object
    outer_sphere.name = f"{name}_outer"
    
    # Create inner sphere
    inner_radius = outer_radius - thickness
    bpy.ops.mesh.primitive_uv_sphere_add(radius=inner_radius, segments=32, ring_count=16)
    inner_sphere = bpy.context.active_object
    inner_sphere.name = f"{name}_inner"
    
    # Select both spheres
    outer_sphere.select_set(True)
    inner_sphere.select_set(True)
    bpy.context.view_layer.objects.active = outer_sphere
    
    # Perform boolean difference
    bool_mod = outer_sphere.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = inner_sphere
    bpy.ops.object.modifier_apply(modifier="Boolean")
    
    # Remove inner sphere
    bpy.data.objects.remove(inner_sphere)
    
    # Create cutaway
    bm = bmesh.new()
    bpy.ops.object.mode_set(mode='EDIT')
    bm.from_mesh(outer_sphere.data)
    
    # Create a cube for the boolean cutaway - MODIFIED FOR 1/8 CUTAWAY
    bpy.ops.object.mode_set(mode='OBJECT')
    # We'll create a smaller cube and position it to cut only 1/8 of the sphere
    bpy.ops.mesh.primitive_cube_add(size=outer_radius*2)
    cube = bpy.context.active_object
    # Position the cube to cut only 1/8 of the sphere
    cube.location = Vector((outer_radius, -outer_radius, outer_radius))
    
    # Boolean operation for cutaway
    bool_mod = outer_sphere.modifiers.new(name="Cutaway", type='BOOLEAN')
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object = cube
    bpy.ops.object.select_all(action='DESELECT')
    outer_sphere.select_set(True)
    bpy.context.view_layer.objects.active = outer_sphere
    bpy.ops.object.modifier_apply(modifier="Cutaway")
    
    # Remove cube
    bpy.data.objects.remove(cube)
    
    # Apply material
    if outer_sphere.data.materials:
        outer_sphere.data.materials[0] = material
    else:
        outer_sphere.data.materials.append(material)
    
    # Move to collection
    if outer_sphere.users_collection[0] != collection:
        collection.objects.link(outer_sphere)
        outer_sphere.users_collection[0].objects.unlink(outer_sphere)
    
    return outer_sphere

# Function to animate expansion and color change
def animate_expansion(pg_layer, pm_layer, pm_color_node):
    # Animate expansion for both layers
    for frame in range(scene.frame_start, scene.frame_end + 1):
        # Calculate expansion factor based on frame
        progress = (frame - scene.frame_start) / (scene.frame_end - scene.frame_start)
        expansion_factor = 1.0 + (max_expansion - 1.0) * progress
        
        # Set scale keyframes for both layers
        current_frame = frame
        pg_layer.scale = (expansion_factor, expansion_factor, expansion_factor)
        pm_layer.scale = (expansion_factor, expansion_factor, expansion_factor)
        
        pg_layer.keyframe_insert(data_path="scale", frame=current_frame)
        pm_layer.keyframe_insert(data_path="scale", frame=current_frame)
        
        # Animate color change at threshold
        if progress < pressure_threshold:
            color_value = 0.0  # White
        else:
            # Gradually change from white to red after threshold
            transition_progress = (progress - pressure_threshold) / (1.0 - pressure_threshold)
            color_value = min(1.0, transition_progress * 2.0)  # Faster color change
            
        pm_color_node.outputs[0].default_value = color_value
        pm_color_node.outputs[0].keyframe_insert(data_path="default_value", frame=current_frame)

# Function to set up camera and lighting
def setup_scene():
    # Add camera
    bpy.ops.object.camera_add(location=(0, -15, 5))
    camera = bpy.context.active_object
    camera.name = "Main_Camera"
    camera.rotation_euler = (math.radians(75), 0, 0)
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    # Add key light
    bpy.ops.object.light_add(type='AREA', radius=5, location=(10, -10, 12))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light"
    key_light.data.energy = 500
    
    # Add fill light
    bpy.ops.object.light_add(type='AREA', radius=3, location=(-8, -12, 6))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = 300
    
    # Add backlight
    bpy.ops.object.light_add(type='AREA', radius=2, location=(0, 5, 8))
    back_light = bpy.context.active_object
    back_light.name = "Back_Light"
    back_light.data.energy = 200
    
    # Add world ambient
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg_node = world.node_tree.nodes['Background']
    bg_node.inputs[0].default_value = (0.05, 0.05, 0.08, 1.0)  # Dark blue background
    bg_node.inputs[1].default_value = 1.0  # Strength

# Main execution
def main():
    # Create materials
    pg_material, pm_material, pm_color_node = create_materials()
    
    # Calculate initial radii to ensure layers are touching
    initial_radius_pm = initial_radius_pg - pg_thickness
    
    # Create layers
    pg_layer = create_hollow_sphere("PG_Layer", initial_radius_pg, pg_thickness, pg_material, simulation_collection)
    pm_layer = create_hollow_sphere("PM_Layer", initial_radius_pm, pm_thickness, pm_material, simulation_collection)
    
    # Setup scene
    setup_scene()
    
    # Animate layers
    animate_expansion(pg_layer, pm_layer, pm_color_node)
    
    print("Animation setup complete!")

if __name__ == "__main__":
    main()