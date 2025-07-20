from typing import Optional, Tuple

import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image

from .point_cloud import PointCloud

def plot_point_cloud(
    pc: PointCloud,
    color: bool = True,
    grid_size: int = 1,
    fixed_bounds: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = (
        (-0.75, -0.75, -0.75),
        (0.75, 0.75, 0.75),
    ),
):
    """
    Render a point cloud as a plot to the given image path.

    :param pc: the PointCloud to plot.
    :param image_path: the path to save the image, with a file extension.
    :param color: if True, show the RGB colors from the point cloud.
    :param grid_size: the number of random rotations to render.
    """
    fig = plt.figure(figsize=(8, 8))

    for i in range(grid_size):
        for j in range(grid_size):
            ax = fig.add_subplot(grid_size, grid_size, 1 + j + i * grid_size, projection="3d")
            color_args = {}
            if color:
                color_args["c"] = np.stack(
                    [pc.channels["R"], pc.channels["G"], pc.channels["B"]], axis=-1
                )
            c = pc.coords

            if grid_size > 1:
                theta = np.pi * 2 * (i * grid_size + j) / (grid_size**2)
                rotation = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                c = c @ rotation

            ax.scatter(c[:, 0], c[:, 1], c[:, 2], **color_args)

            if fixed_bounds is None:
                min_point = c.min(0)
                max_point = c.max(0)
                size = (max_point - min_point).max() / 2
                center = (min_point + max_point) / 2
                ax.set_xlim3d(center[0] - size, center[0] + size)
                ax.set_ylim3d(center[1] - size, center[1] + size)
                ax.set_zlim3d(center[2] - size, center[2] + size)
            else:
                ax.set_xlim3d(fixed_bounds[0][0], fixed_bounds[1][0])
                ax.set_ylim3d(fixed_bounds[0][1], fixed_bounds[1][1])
                ax.set_zlim3d(fixed_bounds[0][2], fixed_bounds[1][2])

    return fig


# source: https://github.com/hasancaslan/BeautifulPointCloud
class XMLTemplates:
    HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="2,2,2" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="540"/> <!-- Set to 540 for square aspect ratio -->
            <integer name="height" value="540"/> <!-- Set to 540 for square aspect ratio -->
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>

"""

    HEAD_NO_SHADOW = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="2,2,3" target="0,0,1" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="540"/> <!-- Set to 540 for square aspect ratio -->
            <integer name="height" value="540"/> <!-- Set to 540 for square aspect ratio -->
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""

    HEAD_HIGH_QUALITY = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="1.7,1.7,1.7" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="1536"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="740"/> <!-- Set to 540 for square aspect ratio -->
            <integer name="height" value="740"/> <!-- Set to 540 for square aspect ratio -->
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>

"""

    HEAD_VIDEO_QUALITY = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="1.7,1.7,1.7" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="1000"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="740"/> <!-- Set to 540 for square aspect ratio -->
            <integer name="height" value="740"/> <!-- Set to 540 for square aspect ratio -->
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>

"""

    HEAD_HIGH_QUALITY_NO_SHADOW = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="1.7,1.7,2.7" target="0,0,1.0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="independent">
            <integer name="sampleCount" value="1536"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="740"/> <!-- Set to 540 for square aspect ratio -->
            <integer name="height" value="740"/> <!-- Set to 540 for square aspect ratio -->
            <rfilter type="gaussian"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
"""

    BALL_SEGMENT = """
    <shape type="sphere">
        <float name="radius" value="0.015"/>
        <transform name="toWorld">
            <translate x="{}" y="{}" z="{}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{},{},{}"/>
        </bsdf>
    </shape>
"""

    TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6.3,6.3,6.3"/>
        </emitter>
    </shape>
</scene>
"""

def render_point_cloud(
    pc: PointCloud,
    output_path: str = None,
    with_labels: bool = False,
    high_quality: bool = False,
    no_shadow: bool = False,
    red_tint: bool = False,
    green_tint: bool = False,
    mask_indices: Optional[np.ndarray] = None,
    mask_color: Optional[np.ndarray] = np.array([0.65, 0.01, 0.15]),
    mask_color_2: Optional[np.ndarray] = None,
    dark_gray_points: bool = False,  # New flag for dark gray points
):
    c = pc.coords
    xml_segments = [XMLTemplates.HEAD]
    if no_shadow:
        xml_segments = [XMLTemplates.HEAD_NO_SHADOW]
    if high_quality:
        xml_segments = [XMLTemplates.HEAD_HIGH_QUALITY]
    if high_quality and no_shadow:
        xml_segments = [XMLTemplates.HEAD_HIGH_QUALITY_NO_SHADOW]
    for i, point in enumerate(c):
        # Apply dark gray color if the flag is set
        if dark_gray_points:
            color = np.array([0.2, 0.2, 0.2])  # Dark gray
            color /= np.linalg.norm(color)  # Normalize color
        elif with_labels:
            if pc.labels[i] == 0:
                color = np.clip(np.array([0.65, 0.1, 0.1]), 0.001, 1.0)
            elif pc.labels[i] == 1:
                color = np.clip(np.array([0.1, 0.65, 0.1]), 0.001, 1.0)
            elif pc.labels[i] == 2:
                color = np.clip(np.array([0.1, 0.1, 0.65]), 0.001, 1.0)
            else:
                color = np.clip(np.array([0.1, 0.65, 0.65]), 0.001, 1.0)
            color /= np.linalg.norm(color)
        else:
        
            rgb = np.array([point[0] + 0.5, point[1] + 0.5, point[2] + 0.5 - 0.0125])
            color = np.clip(rgb, 0.001, 1.0)
            color /= np.linalg.norm(color)
            if mask_indices is not None:
                if i in mask_indices:
                    rgb = mask_color
                    color = np.clip(
                        rgb,
                        0.001,
                        1.0,
                    )
                    #color = np.clip(
                    #    np.array([point[0] + 0.65, point[1] * 0.6 + 0.2, point[2] * 0.6 + 0.2 - 0.0125]),
                    #    0.001,
                    #    1.0,
                    #)
                elif mask_color_2 is not None:
                    rgb = mask_color_2
                    color = np.clip(
                        rgb,
                        0.001,
                        1.0,
                    )
                else:
                    color = np.clip(
                        np.array([point[0] * 0.7 + 0.2, point[1] + 0.45, point[2] * 0.7 + 0.2 - 0.0125]),
                        0.001,
                        1.0,
                    )
            if red_tint:
                color = np.clip(
                        np.array([point[0] + 0.65, point[1] * 0.6 + 0.2, point[2] * 0.6 + 0.2 - 0.0125]),
                        0.001,
                        1.0,
                    )
            if green_tint:
                color = np.clip(
                        np.array([point[0] * 0.7 + 0.2, point[1] + 0.45, point[2] * 0.7 + 0.2 - 0.0125]),
                        0.001,
                        1.0,
                    )

        z_addition = 0.0
        if no_shadow:
            z_addition = 1.0
        xml_segments.append(
            XMLTemplates.BALL_SEGMENT.format(point[0], point[1], point[2] + z_addition, *color)
        )


    xml_segments.append(XMLTemplates.TAIL)
    xml_content = "".join(xml_segments)

    mi.set_variant("scalar_rgb")
    scene = mi.load_string(xml_content)
    img = mi.render(scene)

    if output_path is not None:
        mi.util.write_bitmap(output_path, img)

    img = np.array(img)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def render_point_cloud_video(
    pc_path: str,
    output_path: str,
    num_frames: int = 100,
):
    file_name = os.path.basename(pc_path).split(".")[0]
    print(f"Rendering video for {file_name}")
    if file_name == "changeit":
        try:
            c = np.load(pc_path)["coords"]
            #c = np.load(pc_path)["tensor"]
        except:
            pc = PointCloud.load(pc_path)
            c = pc.coords
        #c = np.load(pc_path)['coords']
        #theta = np.pi / 2
        #rotation = np.array(
        #    [
        #        [np.cos(theta), -np.sin(theta), 0.0],
        #        [np.sin(theta), np.cos(theta), 0.0],
        #        [0.0, 0.0, 1.0],
        #    ]
        #)
        #c = c @ rotation
        #rotation = np.array([
        #    [np.cos(theta), 0, np.sin(theta)],
        #    [0, 1, 0],
        #    [-np.sin(theta), 0, np.cos(theta)],
        #])
        #c = c @ rotation
    elif file_name == "spice":
        c = np.load(pc_path)["coords"]
        #theta = np.pi / 2
        #rotation = np.array(
        #    [
        #        [np.cos(theta), -np.sin(theta), 0.0],
        #        [np.sin(theta), np.cos(theta), 0.0],
        #        [0.0, 0.0, 1.0],
        #    ]
        #)
        #c = c @ rotation
        c[:, 0] = c[:, 0] / 1.5
        c[:, 1] = c[:, 1] / 1.5
        c[:, 2] = c[:, 2] / 1.5
    else:
        pc = PointCloud.load(pc_path)
        c = pc.coords

    images_dir = os.path.join(output_path, f"images")
    os.makedirs(images_dir, exist_ok=True)

    # add tqdm progress bar
    for i in tqdm(range(num_frames)):
        xml_segments = [XMLTemplates.HEAD_VIDEO_QUALITY]
        theta = (2 * np.pi / num_frames) * i
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta), np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        for point in c:
            rgb = np.array([point[0] + 0.5, point[1] + 0.5, point[2] + 0.5 - 0.0125])
            point = point @ rot
            color = np.clip(rgb, 0.001, 1.0)
            color /= np.linalg.norm(color)
            xml_segments.append(
                XMLTemplates.BALL_SEGMENT.format(point[0], point[1], point[2], *color)
            )

        xml_segments.append(XMLTemplates.TAIL)
        xml_content = "".join(xml_segments)
    
        mi.set_variant("scalar_rgb")
        scene = mi.load_string(xml_content)
        img = mi.render(scene)
    
        if output_path is not None:
            mi.util.write_bitmap(output_path, img)
    
        img = np.array(img)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        Image.fromarray(img).save(f"{images_dir}/frame_{i:05}.png")

    # Create video from all the frames in the output directory
    video_path = os.path.join(output_path, f"video.mp4")
    out = cv2.VideoWriter(video_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          20, 
                          (400, 400))

    for i in range(num_frames):
        img = cv2.imread(f"{images_dir}/frame_{i:05}.png", cv2.IMREAD_UNCHANGED)
        out.write(img)

    out.release()
