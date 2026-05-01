"""VLM-based contact assignment (Sec. 3.1, Appendix A).

At inference time, we sample frames at 3 fps, overlay colored segmentation
masks for hands and candidate objects, and ask a VLM to label which hand
touches which object.  The prompt is reproduced verbatim from the paper's
supplementary material.

Pipeline:
  1. Subsample clip to 3 fps keyframes
  2. Render: green dot = left hand center, red dot = right hand center,
     colored filled mask per object
  3. Call GPT-4V (or compatible VLM) with the spatial prompt
  4. Parse JSON output; enforce the one-out-of-k constraint

The output is a (T, 2) binary tensor in the original clip's time axis,
with 1 indicating contact for (left, right) hands.
"""

from __future__ import annotations

import base64
import json
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Prompt (from Appendix A, Table in paper)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise visual classifier for hand-object contact detection in cluttered scenes.

CRITICAL CONSTRAINTS:
1. Each hand (left/right) can be in contact with AT MOST ONE object at a time.
2. "In contact" means direct physical touch: grasping, holding, pressing, or any visible contact.
3. If a hand is not clearly touching any object, you must mark all objects as 0 for that hand."""

USER_PROMPT_TEMPLATE = """Analyze this image for hand-object contact (actual touching, not just reaching).
VISUAL GUIDANCE:
The image has been annotated with colored masks:
- GREEN dot = Left hand
- RED dot = Right hand
- Other COLORED masks = Candidate objects (each object has a unique color)
CANDIDATE OBJECTS (in order):
{object_list}
STRICT DEFINITION OF CONTACT:
For this task, contact means clear physical touching in this frame only.
Contact (label = 1) requires BOTH:
1. Mask intersection:
   - The hand mask and the object mask share some pixels or directly overlap at the boundary (no visible gap).
2. Touching region:
   - The overlap is at a plausible touching area (finger tips, fingers, palm, side of hand) on the visible surface of the object.
NO Contact (label = 0) in all of these cases:
- The hand is reaching toward, hovering above, or very close to an object with a visible gap between masks.
- The hand is aligned in depth (e.g., above or behind the object) but the masks do not intersect.
- The hand is in a pose that suggests future contact, but there is no current touching in this frame.
- There is only a tiny, ambiguous intersection (1-2 pixels) that could be noise or occlusion. In such uncertain cases, choose 0 (no contact).
IMPORTANT:
- **Reaching or hovering is NOT contact.**
- **If you are unsure whether contact is happening, choose 0 (no contact).**
CONSTRAINTS (VALIDATION CHECK):
- Each hand can touch AT MOST ONE object.
  - Sum of left across all objects must be <= 1.
  - Sum of right across all objects must be <= 1.
- If a hand is not clearly touching any object, it should have 0 for all objects.
OUTPUT FORMAT:
Return only a JSON object in this exact format (no extra text):
{{
{json_format}
}}
Where:
- 1 = the specified hand is clearly touching that object in this frame."""


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

# Colors for up to 8 objects (BGR for cv2, then converted to RGB for display)
_OBJECT_COLORS_BGR = [
    (255, 128, 0),    # blue
    (0, 255, 128),    # green-cyan
    (128, 0, 255),    # purple
    (0, 128, 255),    # orange
    (255, 0, 128),    # pink
    (0, 255, 255),    # yellow
    (255, 255, 0),    # cyan
    (128, 255, 0),    # lime
]

_LEFT_COLOR_BGR  = (0, 255, 0)    # green
_RIGHT_COLOR_BGR = (0, 0, 255)    # red
_DOT_RADIUS      = 12


def _render_annotated_frame(
    frame_rgb:   np.ndarray,        # (H, W, 3) uint8 RGB
    left_box:    Optional[np.ndarray],   # (4,) amodal [x1,y1,x2,y2] or None
    right_box:   Optional[np.ndarray],   # (4,) amodal [x1,y1,x2,y2] or None
    obj_masks:   list[np.ndarray],       # list of (H, W) bool masks
    obj_names:   list[str],
) -> np.ndarray:
    """Overlay colored annotations on the frame for VLM input.

    Returns:
        (H, W, 3) uint8 RGB annotated image
    """
    canvas = frame_rgb.copy()

    # Overlay object masks with semi-transparency
    for i, (mask, _) in enumerate(zip(obj_masks, obj_names)):
        color_bgr = _OBJECT_COLORS_BGR[i % len(_OBJECT_COLORS_BGR)]
        color_rgb = color_bgr[::-1]
        overlay = canvas.copy()
        overlay[mask > 0] = (
            0.4 * np.array(color_rgb) +
            0.6 * canvas[mask > 0]
        ).astype(np.uint8)
        canvas = overlay

    # Hand dots at bounding-box centers
    for box, color in [(left_box, _LEFT_COLOR_BGR[::-1]),
                       (right_box, _RIGHT_COLOR_BGR[::-1])]:
        if box is not None and not np.all(box == 0):
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            cx = np.clip(cx, 0, frame_rgb.shape[1] - 1)
            cy = np.clip(cy, 0, frame_rgb.shape[0] - 1)
            cv2.circle(canvas, (cx, cy), _DOT_RADIUS, color, -1)

    return canvas


def _frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Encode RGB frame as JPEG base64 string for OpenAI API."""
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode('utf-8')


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def _call_vlm(
    annotated_frame: np.ndarray,
    obj_names:       list[str],
    api_key:         str,
    model:           str = 'gpt-4o',
    max_retries:     int = 3,
) -> dict[str, dict[str, int]]:
    """Call GPT-4V and return parsed contact JSON.

    Args:
        annotated_frame: (H, W, 3) annotated RGB image
        obj_names:       list of object labels e.g. ['obj1', 'obj2']
        api_key:         OpenAI API key
        model:           VLM model name
    Returns:
        {obj_name: {'left': 0/1, 'right': 0/1}, ...}
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    object_list = '\n'.join(f'{i+1}. {n}' for i, n in enumerate(obj_names))
    json_fmt    = '\n'.join(
        f'  "{n}": {{"left": 0, "right": 0}}' for n in obj_names
    )
    user_prompt = USER_PROMPT_TEMPLATE.format(
        object_list=object_list, json_format=json_fmt
    )

    b64 = _frame_to_base64(annotated_frame)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model   = model,
                messages = [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': [
                        {'type': 'text',      'text': user_prompt},
                        {'type': 'image_url', 'image_url': {
                            'url': f'data:image/jpeg;base64,{b64}',
                            'detail': 'high',
                        }},
                    ]},
                ],
                max_tokens      = 256,
                temperature     = 0.0,
                response_format = {'type': 'json_object'},
            )
            raw = resp.choices[0].message.content
            result = json.loads(raw)
            return _validate_contact_json(result, obj_names)
        except Exception as e:
            if attempt == max_retries - 1:
                warnings.warn(f'VLM call failed after {max_retries} attempts: {e}')
                return {n: {'left': 0, 'right': 0} for n in obj_names}


def _validate_contact_json(
    raw:       dict,
    obj_names: list[str],
) -> dict[str, dict[str, int]]:
    """Enforce one-out-of-k constraint and clamp to {0, 1}."""
    result = {}
    for n in obj_names:
        entry = raw.get(n, {})
        result[n] = {
            'left':  int(bool(entry.get('left',  0))),
            'right': int(bool(entry.get('right', 0))),
        }

    # Each hand contacts at most one object
    for side in ('left', 'right'):
        total = sum(result[n][side] for n in obj_names)
        if total > 1:
            # Keep only the first positive
            found = False
            for n in obj_names:
                if result[n][side] == 1:
                    if found:
                        result[n][side] = 0
                    else:
                        found = True
    return result


# ---------------------------------------------------------------------------
# Main contact labeler
# ---------------------------------------------------------------------------

class VLMContactLabeler:
    """Assigns binary contact labels to a clip at 3 fps using a VLM.

    Args:
        api_key:    OpenAI API key (or set OPENAI_API_KEY env var)
        model:      VLM model name (default 'gpt-4o')
        contact_fps: subsampling rate (default 3 fps, matching paper)
        clip_fps:   original clip framerate (default 30 fps)
        n_examples: number of in-context calibration examples (paper: 5)
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str = 'gpt-4o',
        contact_fps: int = 3,
        clip_fps:    int = 30,
    ):
        import os
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.model   = model
        # Subsample every N frames to hit contact_fps
        self.subsample = max(1, clip_fps // contact_fps)

    def label_clip(
        self,
        frames:      np.ndarray,          # (T, H, W, 3) uint8 RGB
        left_boxes:  np.ndarray,          # (T, 4) amodal boxes or zeros
        right_boxes: np.ndarray,          # (T, 4) amodal boxes or zeros
        obj_masks:   list[np.ndarray],    # list of (T, H, W) bool
        obj_names:   list[str],
    ) -> torch.Tensor:
        """Return (T, 2) binary contact tensor interpolated to full frame rate.

        Args:
            frames:      RGB video frames
            left_boxes:  left hand bounding boxes per frame
            right_boxes: right hand bounding boxes per frame
            obj_masks:   per-object segmentation masks
            obj_names:   object labels (e.g. ['holder_black'])
        Returns:
            (T, 2) float32 tensor of contact labels [left, right]
        """
        T = frames.shape[0]
        # Sample keyframes
        keyframe_indices = list(range(0, T, self.subsample))
        contacts = {}   # frame_idx -> {obj: {left, right}}

        for idx in keyframe_indices:
            frame_masks = [m[idx] for m in obj_masks] if obj_masks else []
            annotated = _render_annotated_frame(
                frames[idx],
                left_boxes[idx],
                right_boxes[idx],
                frame_masks,
                obj_names,
            )
            if self.api_key:
                contacts[idx] = _call_vlm(annotated, obj_names,
                                          self.api_key, self.model)
            else:
                # No API key — return zeros (for testing without VLM)
                contacts[idx] = {n: {'left': 0, 'right': 0} for n in obj_names}

        return self._interpolate_to_full(contacts, keyframe_indices, T, obj_names)

    @staticmethod
    def _interpolate_to_full(
        contacts:         dict,
        keyframe_indices: list[int],
        T:                int,
        obj_names:        list[str],
    ) -> torch.Tensor:
        """Forward-fill contact labels from keyframes to all frames."""
        result = torch.zeros(T, 2)   # [left, right]
        prev   = {n: {'left': 0, 'right': 0} for n in obj_names}

        ki_set = set(keyframe_indices)
        for t in range(T):
            if t in ki_set:
                prev = contacts[t]
            # Aggregate over all objects (OR — any object contacted → contact=1)
            left  = int(any(prev[n]['left']  for n in obj_names))
            right = int(any(prev[n]['right'] for n in obj_names))
            result[t] = torch.tensor([left, right], dtype=torch.float32)
        return result
