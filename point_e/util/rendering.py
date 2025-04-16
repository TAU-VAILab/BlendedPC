import mitsuba as mi
import numpy as np

from .point_cloud import PointCloud


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
    output_path: str = None
):
    xml_segments = [XMLTemplates.HEAD]
    for point in pc.coords:
        rgb = np.array([point[0] + 0.5, point[1] +
                       0.5, point[2] + 0.5 - 0.0125])
        color = np.clip(rgb, 0.001, 1.0)
        color /= np.linalg.norm(color)
        xml_segments.append(
            XMLTemplates.BALL_SEGMENT.format(
                point[0], point[1], point[2], *color)
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
