import os
import random

from glob import glob
from tqdm import tqdm
from PIL import Image
import json
import argparse

if __name__ == "__main__":
    # write me an argparse that takes seq, start_frame, end_frame, data_root, n_images, and out_dir as arguemnts
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/nas4_dataset/3D/KITTI-360")
    parser.add_argument("--n_images", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="/home/nas4_user/sungwonhwang/ws/diffusers/data/kitti360")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--use_blip", action="store_true")

    args = parser.parse_args() 

    seqs = [f"2013_05_28_drive_{str(i).zfill(4)}_sync" for i in [0, 2, 3, 4, 5, 6, 7, 9, 10]]
    text_prompt = "a photography of a suburban street"
    
    for seq in tqdm(seqs, total=len(seqs)):
        frame_segments = glob(os.path.join(args.data_root, "data_3d_semantics", "train", seq, "static", "*.ply"))
        frame_segments = [os.path.basename(frame).split(".")[0] for frame in frame_segments]
        frame_segments = [(f.split("_")[0], f.split("_")[1]) for f in frame_segments]
        for frame_segment in tqdm(frame_segments, total=len(frame_segments)):
            start_frame, end_frame = frame_segment
            out_dir = os.path.join(args.out_dir, seq, f"{str(start_frame).zfill(10)}_{str(end_frame).zfill(10)}", "train")

            os.makedirs(out_dir, exist_ok=True)

            image_root = os.path.join(args.data_root, "data_2d_raw", seq, "**/*.png")

            img_fs = glob(image_root, recursive=True)
            random.shuffle(img_fs)


            n_count = 0
            captions = []
            while n_count < args.n_images:
                img_f = img_fs.pop()
                frame = int(img_f.split("/")[-1].split(".")[0])
                if len(img_fs) == 0:
                    img_fs = glob(image_root, recursive=True)
                    random.shuffle(img_fs)
                if frame >= int(start_frame) and frame <= int(end_frame):
                    img = Image.open(img_f)
                    w, h = img.size
                    l = random.randrange(0, w-h)
                    img = img.crop((l, 0, l + h, h))
                    img = img.resize((args.img_size, args.img_size))
                    img.save(os.path.join(out_dir, f"{str(n_count).zfill(3)}.png"))
                    n_count += 1
            # save metatdata & text prompt 
            metadata = [{"file_name": f"{str(j).zfill(3)}.png", "additional_feature": text_prompt} for j in range(args.n_images)]
            with open(os.path.join(out_dir, "metadata.jsonl"), encoding="utf=8", mode="w") as f:
                for i in metadata:
                    f.write(json.dumps(i) + "\n")
            print(f"Save complete at {out_dir}")