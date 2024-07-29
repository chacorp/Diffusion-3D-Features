import torch
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
import numpy as np
from diffusion import add_texture_to_render
from dino import get_dino_features
from render import batch_render
from pytorch3d.ops import ball_query
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV

from tqdm import tqdm
from time import time
import random


FEATURE_DIMS = 1280+768 # diffusion unet + dino
VERTEX_GPU_LIMIT = 10000


def arange_pixels(
    resolution=(128, 128),
    batch_size=1,
    subsample_to=None,
    invert_y_axis=False,
    margin=0,
    corner_aligned=True,
    jitter=None,
):
    h, w = resolution
    n_points = resolution[0] * resolution[1]
    uh = 1 if corner_aligned else 1 - (1 / h)
    uw = 1 if corner_aligned else 1 - (1 / w)
    if margin > 0:
        uh = uh + (2 / h) * margin
        uw = uw + (2 / w) * margin
        w, h = w + margin * 2, h + margin * 2

    x, y = torch.linspace(-uw, uw, w), torch.linspace(-uh, uh, h)
    if jitter is not None:
        dx = (torch.ones_like(x).uniform_() - 0.5) * 2 / w * jitter
        dy = (torch.ones_like(y).uniform_() - 0.5) * 2 / h * jitter
        x, y = x + dx, y + dy
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )

    if subsample_to is not None and subsample_to > 0 and subsample_to < n_points:
        idx = np.random.choice(
            pixel_scaled.shape[1], size=(subsample_to,), replace=False
        )
        pixel_scaled = pixel_scaled[:, idx]

    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0

    return pixel_scaled


def Mesh_from_obj(obj_file, tex_file, device='cpu', create_texture_atlas=False):
    verts, faces, aux = load_obj(obj_file)

    tex = None
    if create_texture_atlas:
        # TexturesAtlas type
        tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
    else:
        # TexturesUV type
        verts_uvs = aux.verts_uvs.to(device)      # (V, 2)
        faces_uvs = faces.textures_idx.to(device) # (F, 3)

        texture = np.array(Image.open(tex_file))     # (H, W, 3)

        denom = 1/255
        # import pdb;pdb.set_trace()
        tex_map  = torch.tensor(texture[..., :3]).unsqueeze(0) * denom
        tex_map  = tex_map.to(device)

        # pytorch3D texture attribute
        tex      = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=tex_map)

    ## set 'disp' as a mesh texture 
    mesh = Meshes(
        verts    = [verts.to(device)], 
        faces    = [faces.verts_idx.to(device)],
        textures = tex
    )
    return mesh

def compute_features_from_mesh(
    device, 
    pipe, 
    dino_model, 
    obj_file, 
    tex_file, 
    prompt,
    num_views = 100,
    H=512,
    W=512,
    num_images_per_prompt = 1,
    tolerance = 0.004,
    random_seed = 42,
    use_normal_map = True,
    ):
    features = get_features_per_vertex_w_tex(
        device=device,
        pipe=pipe, 
        dino_model=dino_model,
        obj_file=obj_file,
        tex_file=tex_file,
        prompt=prompt,
        num_views=num_views,
        H=H,
        W=W,
        tolerance=tolerance,
        num_images_per_prompt=num_images_per_prompt,
        use_normal_map=use_normal_map,
    )
    return features.cpu()

def get_features_per_vertex_w_tex(
    device,
    pipe,
    dino_model,
    prompt,
    obj_file, 
    tex_file,
    mesh=None,
    num_views=100,
    H=512,
    W=512,
    tolerance=0.01,
    use_latent=False,
    use_normal_map=True,
    num_images_per_prompt=1,
    mesh_vertices=None,
    return_image=True,
    bq=True,
    prompts_list=None,
):
    t1 = time()
    
    if mesh is None:
        mesh = Mesh_from_obj(obj_file, tex_file, device=device)
    if mesh_vertices is None:
        mesh_vertices = mesh.verts_list()[0]
        print("num mesh_vertices: ",mesh_vertices.shape[0])
        print(VERTEX_GPU_LIMIT)
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = random.sample(range(len(mesh_vertices)), VERTEX_GPU_LIMIT)
        maximal_distance = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        maximal_distance = torch.cdist(mesh_vertices, mesh_vertices).max()  # .cpu()
    ball_drop_radius = maximal_distance * tolerance
    torch.cuda.empty_cache()
    #  [100, 512 512 4] [100, 512, 512, 1, 3] ## [100, 512, 512, 1]
    # print(batched_renderings.shape, normal_batched_renderings.shape, depth.shape)
    batched_renderings, normal_batched_renderings, camera, depth = batch_render(
        device, mesh, mesh.verts_list()[0], num_views, H, W, use_normal_map
    )
    #import pdb;pdb.set_trace()
    print("Rendering complete")
    if use_normal_map:
        normal_batched_renderings = normal_batched_renderings.cpu()
    batched_renderings = batched_renderings.cpu()
    pixel_coords = arange_pixels((H, W), invert_y_axis=True)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False)[0].to(device).reshape(1, H, W, 2).half()
    camera = camera.cpu()
    normal_map_input = None
    depth = depth.cpu()
    torch.cuda.empty_cache()
    ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS)).half()  # .to(device)
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1)).half()  # .to(device)
    for idx in tqdm(range(len(batched_renderings))):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        indices = xy_depth[:, 2] != -1
        xy_depth = xy_depth[indices]
        world_coords = (
            camera[idx].unproject_points(
                xy_depth, world_coordinates=True, from_ndc=True
            )  # .cpu()
        ).to(device)
        diffusion_input_img = (
            batched_renderings[idx, :, :, :3].cpu().numpy() * 255
        ).astype(np.uint8)
        if use_normal_map:
            normal_map_input = normal_batched_renderings[idx]
        depth_map = depth[idx, :, :, 0].unsqueeze(0).to(device)
        if prompts_list is not None:
            prompt = random.choice(prompts_list)
        #import pdb;pdb.set_trace()
        diffusion_output = add_texture_to_render(
            pipe,
            diffusion_input_img,
            depth_map,
            prompt,
            normal_map_input=normal_map_input,
            use_latent=use_latent,
            num_images_per_prompt=num_images_per_prompt,
            return_image=return_image
        )
        #import pdb;pdb.set_trace()
        # use
        pil_b_rnd = ToPILImage()(batched_renderings[idx, :, :, :3].permute(2,0,1))
        
        aligned_dino_features = get_dino_features(device, dino_model, pil_b_rnd, grid)
        aligned_features = None
        with torch.no_grad():
            ft = torch.nn.Upsample(size=(H,W), mode="bilinear")(diffusion_output[0].unsqueeze(0)).to(device)
            ft_dim = ft.size(1)
            aligned_features = torch.nn.functional.grid_sample(
                ft, grid, align_corners=False
            ).reshape(1, ft_dim, -1)
            aligned_features = torch.nn.functional.normalize(aligned_features, dim=1)
        # this is feature per pixel in the grid
        aligned_features = torch.hstack([aligned_features*0.5, aligned_dino_features*0.5])
        features_per_pixel = aligned_features[0, :, indices].cpu()
        # map pixel to vertex on mesh
        if bq:
            queried_indices = (
                ball_query(
                    world_coords.unsqueeze(0),
                    mesh_vertices.unsqueeze(0),
                    K=100,
                    radius=ball_drop_radius,
                    return_nn=False,
                )
                .idx[0]
                .cpu()
            )
            mask = queried_indices != -1
            repeat = mask.sum(dim=1)
            ft_per_vertex_count[queried_indices[mask]] += 1
            ft_per_vertex[queried_indices[mask]] += features_per_pixel.repeat_interleave(
                repeat, dim=1
            ).T
        else:
            distances = torch.cdist(
            world_coords, mesh_vertices, p=2
            )
            closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
            ft_per_vertex[closest_vertex_indices] += features_per_pixel.T
            ft_per_vertex_count[closest_vertex_indices] += 1

    idxs = (ft_per_vertex_count != 0)[:, 0]
    ft_per_vertex[idxs, :] = ft_per_vertex[idxs, :] / ft_per_vertex_count[idxs, :]
    missing_features = len(ft_per_vertex_count[ft_per_vertex_count == 0])
    print("Number of missing features: ", missing_features)
    print("Copied features from nearest vertices")

    if missing_features > 0:
        filled_indices = ft_per_vertex_count[:, 0] != 0
        missing_indices = ft_per_vertex_count[:, 0] == 0
        distances = torch.cdist(
            mesh_vertices[missing_indices], mesh_vertices[filled_indices], p=2
        )
        closest_vertex_indices = torch.argmin(distances, dim=1).cpu()
        ft_per_vertex[missing_indices, :] = ft_per_vertex[filled_indices][
            closest_vertex_indices, :
        ]
    t2 = time() - t1
    t2 = t2 / 60
    print("Time taken in mins: ", t2)
    return ft_per_vertex
