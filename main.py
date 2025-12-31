import cv2
import numpy as np
import math

SAUSAGE = (170, 180, 230)  # B, G, R
BLACK = (0, 0, 0)


def draw_rabbit(canvas, x, y, phase=0, scale=1.0, angle=0):
    sprite_size = 260
    rabbit = np.zeros((sprite_size, sprite_size, 3), dtype=np.uint8)

    cx, cy = sprite_size // 2, sprite_size // 2

    SAUSAGE = (170, 180, 230)
    INNER   = (190, 200, 245)
    BLACK   = (0, 0, 0)

    body_center = (cx - int(15*scale), cy + int(40*scale))
    body_axes   = (int(95*scale), int(55*scale))

    head_center = (cx + int(75*scale), cy + int(5*scale))
    head_axes   = (int(45*scale), int(38*scale))


    back_leg_move  = int(18 * math.sin(phase))
    front_leg_move = int(14 * math.sin(phase + math.pi))

    back_joint = (cx - int(55*scale), cy + int(75*scale))
    front_joint = (cx + int(35*scale), cy + int(80*scale))

    cv2.ellipse(rabbit,
                (back_joint[0], back_joint[1] + back_leg_move),
                (int(22*scale), int(14*scale)),
                15, 0, 360, SAUSAGE, -1)

    cv2.ellipse(rabbit,
                (cx - int(70*scale), cy + int(120*scale) + back_leg_move),
                (int(40*scale), int(16*scale)),
                10, 0, 360, SAUSAGE, -1)

    cv2.ellipse(rabbit,
                (front_joint[0], front_joint[1] + front_leg_move),
                (int(18*scale), int(12*scale)),
                -10, 0, 360, SAUSAGE, -1)

    cv2.ellipse(rabbit,
                (cx + int(55*scale), cy + int(125*scale) + front_leg_move),
                (int(32*scale), int(14*scale)),
                -10, 0, 360, SAUSAGE, -1)

    cv2.circle(rabbit, back_joint, int(10*scale), SAUSAGE, -1)
    cv2.circle(rabbit, front_joint, int(8*scale), SAUSAGE, -1)


    cv2.ellipse(rabbit, body_center, body_axes, -10, 0, 360, SAUSAGE, -1)


    tail_center = (cx - int(110*scale), cy + int(35*scale))
    cv2.circle(rabbit, tail_center, int(22*scale), SAUSAGE, -1)

    cv2.ellipse(rabbit, head_center, head_axes, 0, 0, 360, SAUSAGE, -1)

    ear1_center = (cx + int(95*scale), cy - int(85*scale))
    ear2_center = (cx + int(65*scale), cy - int(90*scale))

    cv2.ellipse(rabbit, ear1_center, (int(20*scale), int(75*scale)), 15, 0, 360, SAUSAGE, -1)
    cv2.ellipse(rabbit, ear2_center, (int(18*scale), int(70*scale)), -10, 0, 360, SAUSAGE, -1)

    cv2.ellipse(rabbit, ear1_center, (int(10*scale), int(55*scale)), 15, 0, 360, INNER, -1)
    cv2.ellipse(rabbit, ear2_center, (int(9*scale), int(50*scale)), -10, 0, 360, INNER, -1)

  
    arm_move = int(10 * math.sin(phase))
    cv2.ellipse(rabbit,
                (cx + int(55*scale), cy + int(85*scale) + arm_move),
                (int(18*scale), int(10*scale)),
                20, 0, 360, SAUSAGE, -1)

    cv2.circle(rabbit, (head_center[0]+int(10*scale), head_center[1]-int(8*scale)),
               int(5*scale), BLACK, -1)

    cv2.circle(rabbit, (head_center[0]+int(35*scale), head_center[1]+int(5*scale)),
               int(4*scale), BLACK, -1)

    cv2.ellipse(rabbit, (head_center[0]+int(28*scale), head_center[1]+int(18*scale)),
                (int(12*scale), int(8*scale)), 0, 0, 180, BLACK, 2)

    bob = int(5 * math.sin(phase * 0.6))
    rabbit = np.roll(rabbit, bob, axis=0)


    if angle != 0:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rabbit = cv2.warpAffine(rabbit, M, (sprite_size, sprite_size),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=BLACK)

    h, w = rabbit.shape[:2]
    x1, y1 = x - w//2, y - h//2
    x2, y2 = x1 + w, y1 + h

    if x1 < 0 or y1 < 0 or x2 > canvas.shape[1] or y2 > canvas.shape[0]:
        return canvas

    mask = cv2.cvtColor(rabbit, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    roi = canvas[y1:y2, x1:x2]
    roi[mask > 0] = rabbit[mask > 0]
    canvas[y1:y2, x1:x2] = roi

    return canvas

def make_video():
    width, height = 900, 500
    fps = 30
    total_seconds = 10
    total_frames = fps * total_seconds

    out = cv2.VideoWriter(
        "rabbit_animation.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    rabbit_scale = 0.65 

    for frame in range(total_frames):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if frame < fps * 5:
            t = frame / (fps * 5)
            x = int(120 + t * 700)
            y = height - 140

            phase = frame * 0.30
            canvas = draw_rabbit(canvas, x, y, phase=phase, scale=rabbit_scale, angle=0)

            cv2.putText(canvas, "RABBIT RUNNING", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, SAUSAGE, 2)


        else:
            f = frame - fps * 5
            t = f / (fps * 5)

            x = int(120 + t * 700)
            jump_height = 240
            y = int((height - 140) - jump_height * (4 * t * (1 - t)))

            angle = int(720 * t)
            scale = rabbit_scale + 0.15 * math.sin(math.pi * t)

            canvas = draw_rabbit(canvas, x, y, phase=0, scale=scale, angle=angle)

            cv2.putText(canvas, "RABBIT FLIP", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, SAUSAGE, 2)

        out.write(canvas)

    out.release()
    print("✅ Video saved as rabbit_animation.mp4")


if __name__ == "__main__":
    make_video()
