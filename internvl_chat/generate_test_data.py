import json
import os
from tqdm import tqdm
from decord import VideoReader, cpu
import av
import numpy as np
import random
import math
import torch
from PIL import Image

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
    local_num_frames = 4        
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments
        
    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)

    return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    width = height = 0
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        new_width, new_height = img.size
        if width == 0 and height == 0:
            width, height = new_width, new_height
        elif width != new_width or height != new_height:
            raise ValueError(f"Frame size mismatch: expected ({width}, {height}), got ({new_width}, {new_height})")
    return frame_indices, width, height

def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    normalized_box = [
        min(max(round((x1 / image_width) * 1000), 0), 1000),
        min(max(round((y1 / image_height) * 1000), 0), 1000),
        max(min(round((x2 / image_width) * 1000), 1000), 0),
        max(min(round((y2 / image_height) * 1000), 1000), 0)
    ]
    return normalized_box


if __name__ == "__main__":
    # import debugpy
    # # 在main函数开头添加debugpy配置
    # debugpy.listen(("0.0.0.0", 5678))
    # print("⏳ 等待调试器连接...")
    # debugpy.wait_for_client()
    # print("🔗 调试器已连接!")
    dataset_root = '/data/mm2025/IntentVC_Enhanced_Annotated'
    mode = "public"
    source_path = f"/data2/LLaMA-Factory/sample_result_{mode}.json"
    with open(source_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
        f.close()
    target_transfer = {	
        "electricfan":"electric fan",
        "racing":"racing car",
        "rubicCube":"rubik's cube",
        "pool":"pool ball",
        "gametarget":"game character",
    }
    
    final_data = []
    annos = source_data['captions']
    for k, v in tqdm(annos.items()):
        # video item
        video = k+'/'+k+'.mp4'
        # id item
        video_id = k
        name = video_id.split('-')[0]
            
        bbox_path = video.replace(video.split('/')[-1], 'object_bboxes.txt')
        bbox_path = os.path.join(dataset_root, f'{name}/{bbox_path}')
        video_path = bbox_path.replace('object_bboxes.txt', k+'.mp4')

        num_segments = 32
        frame_indices, width, height = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
        # print(width, height)
        sample_indices = [int(x) for x in frame_indices]

        with open(bbox_path, 'r') as f:
            bboxes = [list(map(int, line.strip().split(','))) for line in f]
            bboxes = [bbox for index, bbox in enumerate(bboxes) if index in sample_indices]

        normlized_bboxes = []
        for index, bbox in enumerate(bboxes):
            # x, y, w, h
            nbbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            nbbox = normalize_coordinates(nbbox, width, height)
            normlized_bboxes.append(nbbox)
            
        bboxes_str = ""
        if target_transfer.get(name, None) is not None:
            print(f"{video.split('/')[-1]} -> {target_transfer[name]}")
            name = target_transfer[name]
        for index, bbox in enumerate(normlized_bboxes):
            # <ref>class name</ref><box>[[x1, y1, x2, y2], ...]</box>
            bboxes_str += f'{index+1}. ' + "<ref>" + f'{name}' + "</ref><box>[" + f'{bbox}' + "]</box>\n"
            
        # conversations item
        conversations = []
        q1 = {
            "from": "human",
            "value": f"Describe this video in one sentence, and focus on the {name} in the red coordinate box, whose coordinates at each frame are:\n{bboxes_str}."
            # "value": f"Describe this video in one sentence, and focus on {name} located in the coordinate list. The coordinates in each frame are:\n{bboxes_str}."
            # "value": f"<image>Describe this video in one sentence, and focus on {name} in the red box of each frame."
            }
        a1 = {
            "from": "gpt",
            "value": k
            }
        conversations.append(q1)
        conversations.append(a1)

        # data item
        data_item = {
            "id": f'{video_id}',
            "video": video,
            "conversations": conversations
        }
        final_data.append(data_item)
            
            
    
    # format_data = read_jsonl(file_path='data/annotaions/ft_data_example.jsonl')
    # new_data = []
    # for i in range(len(format_data)):
    #     item = format_data[i]
    #     try:
    #         item["video"] = video_name[i]
    #     except:
    #         break
    #     new_data.append(item)
    write_jsonl(final_data, file_path=f"shell/data/eval_{mode}_data_f32-redbox.jsonl")
    # # print(format_data)

