import pygame
import numpy as np
import moderngl
import pandas as pd
from pyrr import Matrix44, Vector3
from pygame.locals import *

# ================= SETTINGS =================
WINDOW_SIZE = (1200, 800)
SCALE = 0.02
POINT_SIZE = 5.0
CAMERA_SPEED = 2.0
MOUSE_SENSITIVITY = 0.3
ZOOM_SPEED = 5.0
FPS = 60

# ================= INIT =================
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Starfield Viewer")

ctx = moderngl.create_context()
# program point size not needed for instanced triangle fans, but harmless:
ctx.enable(moderngl.PROGRAM_POINT_SIZE)
# enable depth test so stars/axes depth correctly
ctx.enable(moderngl.DEPTH_TEST)

clock = pygame.time.Clock()

# ========== LOAD DATA ==========
print("Loading CSV data...")
systems = pd.read_csv("DIM_KnownSystems.csv", delimiter=",")
types = pd.read_csv("DIM_StellarType.csv", delimiter=",")
names = list(systems["Name"].apply(lambda x: x.encode('ascii', errors='ignore').decode('ascii')))
xyz = systems[["X", "Y", "Z"]].to_numpy(dtype=np.float32) * SCALE
distances = systems["Distance"].to_numpy(dtype=np.float32)
N = len(systems)

# Map Component1ID to RGB color
type_color    = types.set_index("StellarTypeID")[["Red", "Green", "Blue"]]
type_lum      = types.set_index("StellarTypeID")[["Luminosity"]]
colors        = np.zeros((N, 3), dtype=np.float32)
luminosities  = np.zeros((N, 1), dtype=np.float32)
point_sizes   = np.ones((N, 1), dtype=np.float32)

for i, cid in enumerate(systems["Component1ID"]):
    luminosities[i] = type_lum.loc[cid]
    if cid in type_color.index:
        rgb = type_color.loc[cid].to_numpy() / 255.0
    else:
        rgb = np.array([1.0, 1.0, 1.0])
    colors[i] = rgb

for i, cid in enumerate(systems["Component2ID"]):
    if (cid != 0):
        temp_lum = type_lum.loc[cid]['Luminosity']
        if (temp_lum > luminosities[i]):
            luminosities[i] = temp_lum
            if cid in type_color.index:
                rgb = type_color.loc[cid].to_numpy() / 255.0
            else:
                rgb = np.array([1.0, 1.0, 1.0])
            colors[i] = rgb

for i, cid in enumerate(systems["Component3ID"]):
    if (cid != 0):
        temp_lum = type_lum.loc[cid]['Luminosity']
        if (temp_lum > luminosities[i]):
            luminosities[i] = temp_lum
            if cid in type_color.index:
                rgb = type_color.loc[cid].to_numpy() / 255.0
            else:
                rgb = np.array([1.0, 1.0, 1.0])
            colors[i] = rgb

for i, cid in enumerate(systems["Component4ID"]):
    if (cid != 0):
        temp_lum = type_lum.loc[cid]['Luminosity']
        if (temp_lum > luminosities[i]):
            luminosities[i] = temp_lum
            if cid in type_color.index:
                rgb = type_color.loc[cid].to_numpy() / 255.0
            else:
                rgb = np.array([1.0, 1.0, 1.0])
            colors[i] = rgb

luminosities = luminosities.reshape(-1)  # make it (N,)

# Avoid divide-by-zero by clamping a tiny minimum
# we ignore the 4*Pie denominator here
# values are between 0.00000001 and 1
brightness = np.clip(luminosities / (distances**2 + 1e-12), 0.0, 1.0)

# apparent magnitude can blow up for very small brightness; keep things stable
# values roughly between -20 and 0
# in reality low magnitude would be +20 and high magnitude would be negative, but this is easier to process down the road
magnitude = -2.5 * np.log10(np.clip(1.0 / np.maximum(brightness, 1e-12), 1e-12, 1e12))

# convert apparent magnitude into float from 0 to 1
# play around with exp_steepness inside to show more or less stars, lower numbers mean flatter curve and more stars visible
exp_steepness = 0.2
mag_bytes = np.clip(np.exp(magnitude * exp_steepness), 0.0, 1.0)

# brighter points get larger circles
point_sizes = np.clip(1.5 * brightness, 0.25, 1.5)

print(f"Loaded {len(xyz)} stars.")

with open('stars.csv','w') as outfile: 
  outfile.write('name,dist,luminosity,app_brightness,app_magnitude,mag_percentage,radius_px\n') 
  for i in range(len(systems)): 
    outfile.write('%s,%3.7f,%3.7f,%3.7f,%3.2f,%3.2f,%3.2f\n'%(names[i], distances[i], luminosities[i], brightness[i],magnitude[i],mag_bytes[i],point_sizes[i]))

# ---------- CREATE CIRCLE MESH ----------
segments = 8
theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
unit_circle = np.zeros((segments + 2, 2), dtype='f4')
unit_circle[0] = [0.0, 0.0]  # center
unit_circle[1:-1, 0] = np.cos(theta)
unit_circle[1:-1, 1] = np.sin(theta)
unit_circle[-1] = unit_circle[1]  # close the fan
vbo_circle = ctx.buffer(unit_circle.tobytes())

# make sure instance scalars are shaped (N,1)
point_sizes = point_sizes.reshape(-1, 1)
mag_bytes = mag_bytes.reshape(-1, 1)

# ---------- INSTANCE DATA ----------
instance_data = np.hstack([xyz, point_sizes, colors, mag_bytes]).astype('f4')
vbo_instances = ctx.buffer(instance_data.tobytes())

prog = ctx.program(
    vertex_shader='''
    #version 330
    in vec2 in_position;      // unit circle vertex (2D)
    in vec3 instance_pos;     // circle center in 3D
    in float instance_radius; // circle radius (desired pixel size)
    in vec3 instance_color;   // per-instance color
    in float instance_brightness;   // per-instance brightness

    uniform mat4 mvp;   // model-view-projection
    uniform mat4 view;  // view matrix
    uniform float screen_height; // window height in pixels

    out vec3 frag_color;
    out float frag_brightness;

    void main() {
        // Camera right and up vectors for billboarding
        vec3 right = vec3(view[0][0], view[1][0], view[2][0]);
        vec3 up    = vec3(view[0][1], view[1][1], view[2][1]);

        // Compute position in clip space
        vec4 clip_pos = mvp * vec4(instance_pos, 1.0);

        // Perspective divide to get NDC
        vec3 ndc = clip_pos.xyz / clip_pos.w;

        // Compute scale so that 'instance_radius' pixels corresponds to NDC units
        float scale = instance_radius * 2.0 / screen_height * clip_pos.w;

        // Billboard offset
        vec3 offset = (right * in_position.x + up * in_position.y) * scale;

        // Final world position
        vec4 final_pos = mvp * vec4(instance_pos + offset, 1.0);
        gl_Position = final_pos;

        frag_color = instance_color;
        frag_brightness = instance_brightness;
    }
    ''',

    fragment_shader='''
    #version 330
    in vec3 frag_color;
    in float frag_brightness;
    out vec4 color;
    void main() {
        color = vec4(frag_color * frag_brightness, 1.0);
    }
    ''',
)

prog['screen_height'].value = WINDOW_SIZE[1]

# ---------- VERTEX ARRAY ----------
vao = ctx.vertex_array(
    prog,
    [
        (vbo_circle, '2f', 'in_position'),
        # the '/i' at the end marks these instance attributes
        (vbo_instances, '3f 1f 3f 1f/i', 'instance_pos', 'instance_radius', 'instance_color', 'instance_brightness'),
    ],
)

# --- Axes ---
axis_length = 0.25
axis_vertices = np.array([
    [0, 0, 0], [axis_length, 0, 0],  # X axis
    [0, 0, 0], [0, axis_length, 0],  # Y axis
    [0, 0, 0], [0, 0, axis_length],  # Z axis
], dtype=np.float32)
axis_colors = np.array([
    [1, 0, 0], [1, 0, 0],
    [0, 1, 0], [0, 1, 0],
    [0, 0, 1], [0, 0, 1],
], dtype=np.float32)

vbo_axis = ctx.buffer(axis_vertices.tobytes())
cbo_axis = ctx.buffer(axis_colors.tobytes())
vao_axis = ctx.vertex_array(
    ctx.program(
        vertex_shader='''
        #version 330
        in vec3 in_position;
        in vec3 in_color;
        uniform mat4 mvp;
        out vec3 color;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            color = in_color;
        }
        ''',
        fragment_shader='''
        #version 330
        in vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
        ''',
    ),
    [(vbo_axis, '3f', 'in_position'),
     (cbo_axis, '3f', 'in_color')]
)
prog_axis = vao_axis.program

print('shadering done')

# ================= CAMERA =================
# start the camera backed away from origin so you see stars in front of you
cam_pos = Vector3([0.0, 0.0, 0.0])
yaw, pitch = 0.0, 0.0

def get_view_matrix():
    forward = Vector3([
        np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)),
        np.sin(np.radians(pitch)),
        np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    ])
    return Matrix44.look_at(cam_pos, cam_pos + forward, Vector3([0.0, 1.0, 0.0]))

# ================= MAIN LOOP =================
running = True
right_held = False

# optionally hide mouse while looking
pygame.event.set_grab(False)
pygame.mouse.set_visible(True)

while running:
    dt = clock.tick(FPS) / 1000.0

    # --- EVENTS ---
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            running = False
        elif event.type == MOUSEBUTTONDOWN and event.button == 3:
            right_held = True
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)
        elif event.type == MOUSEBUTTONUP and event.button == 3:
            right_held = False
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
        elif event.type == MOUSEWHEEL:
            cam_pos.z -= event.y * ZOOM_SPEED * dt

    keys = pygame.key.get_pressed()
    forward = Vector3([np.cos(np.radians(yaw)), 0, np.sin(np.radians(yaw))])
    right_vec = Vector3([np.sin(np.radians(yaw - 90)), 0, np.cos(np.radians(yaw - 90))])

    if keys[K_w]:
        cam_pos += forward * CAMERA_SPEED * dt
    if keys[K_s]:
        cam_pos -= forward * CAMERA_SPEED * dt
    if keys[K_a]:
        cam_pos -= right_vec * CAMERA_SPEED * dt
    if keys[K_d]:
        cam_pos += right_vec * CAMERA_SPEED * dt

    if right_held:
        mx, my = pygame.mouse.get_rel()
        yaw += mx * MOUSE_SENSITIVITY
        pitch -= my * MOUSE_SENSITIVITY
        pitch = np.clip(pitch, -89, 89)
    else:
        pygame.mouse.get_rel()  # reset delta

    # ========== RENDER ==========
    ctx.clear(0.0, 0.0, 0.0)
    proj = Matrix44.perspective_projection(45.0, WINDOW_SIZE[0] / WINDOW_SIZE[1], 0.1, 1000.0)
    view = get_view_matrix()
    mvp = proj * view
    mvp_bytes = mvp.astype('f4').tobytes()
    view_bytes = view.astype('f4').tobytes()

    # Draw axes
    prog_axis['mvp'].write(mvp_bytes)
    vao_axis.render(moderngl.LINES)

    # Draw stars: write both mvp and view (view used for billboard orientation)
    prog['mvp'].write(mvp_bytes)
    prog['view'].write(view_bytes)

    # render circle fan once per instance
    vao.render(mode=moderngl.TRIANGLE_FAN, instances=N)

    pygame.display.flip()

pygame.quit()
