
# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


# # Render a textured mesh
# 
# This tutorial shows how to:
# - load a mesh and textures from an `.obj` file. 
# - set up a renderer 
# - render the mesh 
# - vary the rendering settings such as lighting and camera position
# - use the batching features of the pytorch3d API to render the mesh from different viewpoints

# ## 0. Install and Import modules
# In[1]:


import os
import torch
import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturedSoftPhongShader
)

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))


# If using **Google Colab**, fetch the utils file for plotting image grids:

# In[2]:

from plot_image_grid import image_grid


# OR if running **locally** uncomment and run the following cell:

# In[13]:


# from utils import image_grid


# ### 1. Load a mesh and texture file
# 
# Load an `.obj` file and it's associated `.mtl` file and create a **Textures** and **Meshes** object. 
# 
# **Meshes** is a unique datastructure provided in PyTorch3D for working with batches of meshes of different sizes. 
# 
# **Textures** is an auxillary datastructure for storing texture information about meshes. 
# 
# **Meshes** has several class methods which are used throughout the rendering pipeline.

# If running this notebook using **Google Colab**, run the following cell to fetch the mesh obj and texture files and save it at the path `data/cow_mesh`:
# If running locally, the data is already available at the correct path. 



# Setup
device = torch.device("cuda:2")
torch.cuda.set_device(device)

# Set paths
DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)
texture_image=mesh.textures.maps_padded()
print(mesh.textures.verts_uvs_padded())


# #### Let's visualize the texture map

# In[3]:


plt.figure(figsize=(7,7))
print("saving figure!")
plt.imsave('./data/1.png',texture_image.squeeze().cpu().numpy())
plt.grid("off");
plt.axis('off');


# ## 2. Create a renderer
# 
# A renderer in PyTorch3D is composed of a **rasterizer** and a **shader** which each have a number of subcomponents such as a **camera** (orthographic/perspective). Here we initialize some of these components and use default values for the rest.
# 
# In this example we will first create a **renderer** which uses a **perspective camera**, a **point light** and applies **phong shading**. Then we learn how to vary different components using the modular API.  

# In[4]:


# Initialize an OpenGL perspective camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
R, T = look_at_view_transform(2.7, 0, 180) 
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    bin_size = None,  # this setting controls whether naive or coarse-to-fine rasterization is used
    max_faces_per_bin = None  # this setting is for coarse rasterization
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=TexturedSoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)


# ## 3. Render the mesh

# The light is in front of the object so it is bright and the image has specular highlights.

# In[5]:


images = renderer(mesh)
plt.figure(figsize=(10, 10))
print("saving figure 2!")
plt.imsave('./data/2.png',images[0, ..., :3].cpu().numpy())
plt.grid("off");
plt.axis("off");


# ## 4. Move the light behind the object and re-render
# 
# We can pass arbirary keyword arguments to the `rasterizer`/`shader` via the call to the `renderer` so the renderer does not need to be reinitialized if any of the settings change/
# 
# In this case, we can simply update the location of the lights and pass them into the call to the renderer. 
# 
# The image is now dark as there is only ambient lighting, and there are no specular highlights.

# In[6]:


# Now move the light so it is on the +Z axis which will be behind the cow. 
lights.location = torch.tensor([0.0, 0.0, +1.0], device=device)[None]
images = renderer(mesh, lights=lights)


# In[7]:


plt.figure(figsize=(10, 10))
print("saving figure 3!")
plt.imsave('./data/3.png',images[0, ..., :3].cpu().numpy())
plt.grid("off");
plt.axis("off");


# ## 5. Rotate the object, modify the material properties or light properties
# 
# We can also change many other settings in the rendering pipeline. Here we:
# 
# - change the **viewing angle** of the camera
# - change the **position** of the point light
# - change the **material reflectance** properties of the mesh

# In[8]:


# Rotate the object by increasing the elevation and azimuth angles
R, T = look_at_view_transform(dist=2.7, elev=10, azim=-150)
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

# Move the light location so the light is shining on the cow's face.  
lights.location = torch.tensor([[2.0, 2.0, -2.0]], device=device)

# Change specular color to green and change material shininess 
materials = Materials(
    device=device,
    specular_color=[[0.0, 1.0, 0.0]],
    shininess=10.0
)

# Re render the mesh, passing in keyword arguments for the modified components.
images = renderer(mesh, lights=lights, materials=materials, cameras=cameras)


# In[9]:


plt.figure(figsize=(10, 10))
print("saving figure 4!")
plt.imsave('./data/4.png',images[0, ..., :3].cpu().numpy())
plt.grid("off");
plt.axis("off");


# ## 6. Batched Rendering
# 
# One of the core design choices of the PyTorch3D API is to support **batched inputs for all components**. 
# The renderer and associated components can take batched inputs and **render a batch of output images in one forward pass**. We will now use this feature to render the mesh from many different viewpoints.
# 

# In[10]:


# Set batch size - this is the number of different viewpoints from which we want to render the mesh.
batch_size = 20

# Create a batch of meshes by repeating the cow mesh and associated textures. 
# Meshes has a useful `extend` method which allows us do this very easily. 
# This also extends the textures. 
meshes = mesh.extend(batch_size)

# Get a batch of viewing angles. 
elev = torch.linspace(0, 180, batch_size)
azim = torch.linspace(-180, 180, batch_size)

# All the cameras helper methods support mixed type inputs and broadcasting. So we can 
# view the camera from the same distance and specify dist=2.7 as a float,
# and then specify elevation and azimuth angles for each viewpoint as tensors. 
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

# Move the light back in front of the cow which is facing the -z direction.
lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)


# In[11]:


# We can pass arbirary keyword arguments to the rasterizer/shader via the renderer
# so the renderer does not need to be reinitialized if any of the settings change.
images = renderer(meshes, cameras=cameras, lights=lights)


# In[14]:


image_grid(images.cpu().numpy(), rows=4, cols=5, rgb=True)


# ## 7. Conclusion
# In this tutorial we learnt how to **load** a textured mesh from an obj file, initialize a PyTorch3D datastructure called **Meshes**, set up an **Renderer** consisting of a **Rasterizer** and a **Shader**, and modify several components of the rendering pipeline. 
