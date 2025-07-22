import json
import os
from tqdm import tqdm
from decord import VideoReader
import av
import numpy as np
import random
import math
# import torch
# from PIL import Image

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

def get_frame_indices(num_frames, vlen, sample='middle', fix_start=None, input_fps=1, min_num_frames=1, max_num_frames=-1, local_num_frames=8):
    if min_num_frames > vlen:
        if sample == 'dynamic_fps1':
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen

    if sample == 'dynamic_fps1':
        duration = float(vlen) / input_fps
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        if max_num_frames > 0:
            num_frames = min(num_frames, max_num_frames)
        sample = "middle" # NOTE

        # logger.info(f"? is OK (img), duation={duration} frames={num_frames}!!!!")

    num_frames = max(min_num_frames, num_frames)

    # print(f"\033[0;31m vlen={vlen}, input_fps={input_fps} num_frames={num_frames} \033[0m")
        
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError(f"Not support sample type: {sample}")
    
    return frame_indices

def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, min_num_frames=1,
        max_num_frames=-1, local_num_frames=8
    ):
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, min_num_frames=min_num_frames, max_num_frames=max_num_frames, local_num_frames=local_num_frames
    )

    # print(fps, frame_indices)
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), torch.uint8
    # https://github.com/dmlc/decord/issues/208
    video_reader.seek(0)
    # frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, float(fps), duration

class TCSLoader(object):
    def __init__(self, sc_config_key='sensecore'):
        self.client = None
        self.sc_config_key = sc_config_key
        self.time_msg = 'short' 

    def __call__(self, fn, max_num_frames=-1, min_num_frames=4, sample='rand', clip=None, local_num_frames=-1):
        frames, frame_indices, fps, duration = read_frames_decord(video_path=fn, num_frames=max_num_frames, sample=sample, fix_start=None, min_num_frames=min_num_frames, max_num_frames=max_num_frames, local_num_frames=local_num_frames)
        sec = [str(round(f / fps, 1)) for f in frame_indices]
        msg = f"\nThe video lasts for {duration:.2f} seconds, and {len(sec)} frames are uniformly sampled from it. "            
        # frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
        return frames, frame_indices

def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    normalized_box = [
        min(max(round((x1 / image_width) * 1000), 0), 1000),
        min(max(round((y1 / image_height) * 1000), 0), 1000),
        max(min(round((x2 / image_width) * 1000), 1000), 0),
        max(min(round((y2 / image_height) * 1000), 1000), 0)
    ]
    return normalized_box

def get_num_frames_by_duration(duration):
    num_segments = int(duration // 8)
    num_frames = 8 * num_segments

    num_frames = min(256, num_frames)
    num_frames = max(64, num_frames)

    return num_frames

def video_get_item(video_path):
    clip = None

    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == 'video')
    # duration item
    duration = float(video_stream.duration * video_stream.time_base)
    num_frames = get_num_frames_by_duration(duration)


    tcs_loader = TCSLoader()
    frames, frame_indices = tcs_loader(
        video_path,
        max_num_frames=32,
        min_num_frames=8,
        sample='middle',
        local_num_frames=8,
        clip=clip)
    
    # assert len(frame_indices) == num_frames
    # assert frames.shape[0] == num_frames, f"Expected {num_frames} frames, but got {frames.shape[0]} frames."
    return frames, frame_indices, duration

#     '''
#     msg = f"\nThe video lasts for {duration:.2f} seconds, and {num_frames} frames are uniformly sampled from it. "
#     # Generate special tokens for each video frame
#     special_tokens = msg.strip() + '\n' + '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        
#     if '<image>\n' in data_item['conversations'][0]['value']:
#         data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
#             '<image>\n', special_tokens)
#     else:
#         data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
#             '<image>', special_tokens)
#     '''

#     # Select the appropriate preprocessing function based on the template name
#     preprocess_function = self._get_preprocess_function()

#     # Transform each frame image and stack them into a tensor
#     pixel_values = [transform(image) for image in image_list]
#     pixel_values = torch.stack(pixel_values)
#     num_patches = pixel_values.size(0)

#     # Preprocess the conversations and generate the return dictionary
#     num_image_tokens = [self.num_video_token] * num_patches
#     ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
#                                 self.tokenizer, num_image_tokens, group_by_length=True,
#                                 ds_name=self.data_name, num_image=num_patches, model_max_length=self.model_max_length)
#     ret = dict(
#         input_ids=ret['input_ids'][0],
#         labels=ret['labels'][0],
#         pixel_values=pixel_values,
#         image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
#         num_tokens=[len(ret['input_ids'][0])],
#         num_img_tokens=num_image_tokens,
#         num_imgs=[num_patches]
#     )
#     return ret

if __name__ == "__main__":
    # IntentVC_Enhanced_Annotated
    dataset_root = '/data/mm2025/IntentVC'
    source_path = "/data3/LLaMA-Factory/train.json"
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

    max_frame = 0
    min_frame = 100000
    for k, v in tqdm(annos.items()):
        for vid, vi in enumerate(v):
            # video item
            video = k+'/'+k+'.mp4'
            # id item
            video_id = k
            name = video_id.split('-')[0]
            
            bbox_path = video.replace(video.split('/')[-1], 'object_bboxes.txt')
            bbox_path = os.path.join(dataset_root, f'{name}/{bbox_path}')
            # bbox_path = source_path.replace('train.json', 'IntentVC'+'/'+name+'/') + bbox_path
            video_path = bbox_path.replace('object_bboxes.txt', k+'.mp4')

            # duration item
            frames, frame_indices, duration = video_get_item(video_path)
            _, height, width, _ = frames.shape
            max_frame = max(frames.shape[0], max_frame)
            min_frame = min(frames.shape[0], min_frame)

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
            bboxe_array = np.asarray(normlized_bboxes)
            np.save(bbox_path.replace('object_bboxes.txt', 'f32_object_bboxes.npy'), bboxe_array)
            
