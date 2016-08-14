from page_dewarp import *

def make_params(rvec, tvec, cubic_slopes):
    return np.hstack((rvec.flatten(),
                      tvec.flatten(),
                      cubic_slopes.flatten()))

def lerp(a, b, u):
    return a + u * (b-a)

def cspline(x):
    return 3*x**2 - 2*x**3

def subdivide(points, is_closed, tol=None):

    output = []
    
    if tol is None:
        tol = 0.02

    for i, point_i in enumerate(points):
        j = i + 1
        if j >= len(points):
            if is_closed:
               j = 0
            else:
                output.append(point_j)
                break
        point_j = points[j]
        dist = np.linalg.norm(point_j - point_i)
        count = int(np.ceil(dist / tol))
        for k in range(count):
            u = float(k)/count
            output.append(lerp(point_i, point_j, u))

    output = np.array(output)

    return output

def add_paragraph(lc, starty, startx, width, count,
                  indent=0.0, spacing=None, lastwidth=None, tol=None):

    if starty is None:
        starty = line_coords[-1][-1,1] + spacing

    if lastwidth is None:
        lastwidth = width

    if spacing is None:
        spacing = 0.1

    y = starty

    for i in range(count):
        if i+1 == count:
            w = lastwidth
        else:
            w = width
        if i == 0:
            x = startx + indent
            w -= indent
        else:
            x = startx

        segment = np.array([
            [ x, y ],
            [ x+w, y ]])

        lc.append(subdivide(segment, False))
        y += spacing

height = 512
width = height*3/4

display = np.zeros((height, width), dtype=np.uint8)

aspect = float(width)/height

page_width = aspect
page_height = 1.0

# top left norm should be (-1.0, -a)
# bottom right norm should be (1, a)

z = 0.5 * FOCAL_LENGTH

# rvec, tvec, cubic_slopes, ycoords, xcoords
param_vecs = [
    (np.array([0.0, 0.0, 0.0]),
     np.array([-0.5*page_width, -0.5*page_height, 1.02*z]),
     np.array([0.0, 0.0])),

    (np.array([-0.12, 0.0, 0.0]),
     np.array([-0.5*page_width, -0.5*page_height-0.03, 1.35*z]),
     np.array([-1.0, -0.5])),

    (np.array([-0.14, -0.1, 0.08]),
     np.array([-0.5*page_width+0.04, -0.5*page_height-0.05, 1.5*z]),
     np.array([-1.5, 0.0])),

    (np.array([-0.18, -0.2, 0.0]),
     np.array([-0.5*page_width-0.05, -0.5*page_height-0.03, 1.45*z]),
     np.array([-1.0, 1.0])),

    (np.array([0.1, 0.1, 0.0]),
     np.array([-0.5*page_width-0.05, -0.5*page_height+0.03, 1.2*z]),
     np.array([-1.0, -1.0]))
]

param_vecs = [ np.hstack(i) for i in param_vecs ]

outline_coords = np.array([
    [0, 0],
    [page_width, 0],
    [page_width, page_height],
    [0, page_height]])

outline_coords = subdivide(outline_coords, True)

line_coords = []

add_paragraph(line_coords, 0.2, 0.3, page_width-0.5, 2, -0.1, 0.03, page_width-0.6)
add_paragraph(line_coords, 0.3, 0.1, page_width-0.2, 5, 0.05, 0.03, 0.3)
add_paragraph(line_coords, None, 0.1, page_width-0.2, 7, 0.05, 0.03, 0.5)
add_paragraph(line_coords, None, 0.1, page_width-0.2, 3, 0.05, 0.03, 0.35)
add_paragraph(line_coords, None, 0.1, page_width-0.2, 4, 0.05, 0.03, 0.4)

window = 'Visualize'
cv2.namedWindow(window)

frame_count = 0
frames_per_step = 16

nvecs = len(param_vecs)

while 1:

    i = (frame_count / frames_per_step)
    if i >= nvecs:
        break
    
    j = (i + 1) % nvecs
    q = frame_count % frames_per_step
    q = float(q) / frames_per_step

    params = lerp(param_vecs[i], param_vecs[j], cspline(q))
    image_points = project_xy(outline_coords, params)
    image_points = norm2pix(display.shape, image_points, True)
    display[:] = 0
    cv2.fillPoly(display, [image_points], (255, 255, 255), cv2.LINE_AA)

    for line in line_coords:
        image_points = project_xy(line, params)
        image_points = norm2pix(display.shape, image_points, True)
        cv2.polylines(display, [image_points], False, (0, 0, 0), 2, cv2.LINE_AA)

    small = cv2.resize(display, (width/2, height/2), None, -1, -1, cv2.INTER_AREA)
    pil_image = Image.fromarray(small, 'L')
    pil_image.save('frame{:02d}.gif'.format(frame_count))
    
    cv2.imshow(window, small)
    cv2.waitKey(5)

    frame_count += 1
