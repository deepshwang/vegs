import numpy as np 
import pathlib
import argparse
from glob import glob
import os
from kitti360scripts.helpers.project import CameraPerspective as KITTICameraPerspective
import shutil
from scene.colmap_loader import rotmat2qvec
from PIL import Image
from plyfile import PlyData, PlyElement
import open3d as o3d
from tqdm import tqdm

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="/home/nas4_dataset/3D/KITTI-360")
    args = args.parse_args()
    seqs = ["2013_05_28_drive_0005_sync", 
            "2013_05_28_drive_0007_sync",
            "2013_05_28_drive_0003_sync",
            "2013_05_28_drive_0004_sync",
            "2013_05_28_drive_0000_sync",
            "2013_05_28_drive_0006_sync",
            "2013_05_28_drive_0002_sync",
            "2013_05_28_drive_0009_sync",
            "2013_05_28_drive_0010_sync"]

    ok_list = [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 34, 35, 36, 37, 38, 39, 40, 41, 42] # static semantics
    
    # Get a list of frame segments of static scenes
    data_dir = pathlib.Path(args.data_dir)
    for seq in seqs:
        seg_seq_dir = str(data_dir / "data_3d_semantics" / "train" / seq / "static")
        segs = sorted(glob(f"{seg_seq_dir}/*.ply"))
        segs = [os.path.basename(seg).split(".")[0] for seg in segs]
        f = f"{data_dir / 'data_2d_raw' / seq}/**/*.png"
        cams = [KITTICameraPerspective(data_dir, seq=seq, cam_id=0), KITTICameraPerspective(data_dir, seq=seq, cam_id=1)] 
        frames = np.array(sorted(list(cams[0].frames)))
        for seg in tqdm(segs, total=len(segs)):
            cache_dir = ".cache/colmap_ws"
            cache_img_dir = os.path.join(cache_dir, "images")
            os.makedirs(cache_img_dir, exist_ok=True)
            cache_mask_dir = os.path.join(cache_dir, "masks")
            os.makedirs(cache_mask_dir, exist_ok=True)
            
            start_frame = int(seg.split("_")[0])
            end_frame = int(seg.split("_")[1])
            cur_frames = frames.copy()
            
            out = (frames >= start_frame).astype(np.int32) * (frames < end_frame).astype(np.int32)
            filter_idx = np.where(out == 1)[0]
            cur_frames = frames[filter_idx]

            save_idx = 1
            # save cameras.txt
            for cam_id in [0, 1]:
                K = cams[cam_id].K
                f = open(os.path.join(cache_dir, "cameras.txt"), "a")
                # TODO: change Hard-coded width/height for KITTI-360
                f.write(f"{cam_id+1} PINHOLE 1408 376 {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n")
                f.close()
            for frame in cur_frames:
                for cam_id in [0, 1]:
                    img = data_dir / "data_2d_raw" / seq / f"image_{str(cam_id).zfill(2)}" / "data_rect" / f"{str(int(frame)).zfill(10)}.png"
                    if not os.path.exists(img):
                        print(f"{img} doesn't exist!")
                        continue
                    # save image file to cache
                    save_img_name = f"{str(save_idx).zfill(6)}.png"
                    shutil.copyfile(img, os.path.join(cache_img_dir, save_img_name))

                    # save mask
                    semantic_img = data_dir / "data_2d_semantics" / "train" / seq / f"image_{str(cam_id).zfill(2)}" / "semantic" / f"{str(int(frame)).zfill(10)}.png"
                    if os.path.exists(semantic_img):
                        semantic_img = Image.open(semantic_img)
                        semantic_img = np.array(semantic_img)
                        mask = (np.sum([semantic_img == c for c in ok_list], axis=0)>0) + 0.
                        mim = Image.fromarray(np.uint8(mask * 255), 'L')
                        mim.save(os.path.join(cache_mask_dir, save_img_name + ".png"))


                    # save images.txt
                    w2c = np.linalg.inv(cams[cam_id].cam2world[frame])
                    R = rotmat2qvec(w2c[:3, :3])
                    T = w2c[:3, 3].squeeze()
                    f = open(os.path.join(cache_dir, "images.txt"), "a")
                    f.write(f"{save_idx} {R[0]} {R[1]} {R[2]} {R[3]} {T[0]} {T[1]} {T[2]} {cam_id+1} {save_img_name}\n")
                    f.write("\n")
                    f.close()
                    
                    save_idx += 1
            # # create a blank points3d.txt file
            file = open(os.path.join(cache_dir, "points3D.txt"), "w")
            file.close() 

            # # Start COLMAP
            print("Starting COLMAP..")
            database_path = os.path.join(cache_dir, "database.db")
            
            # Feature extraction            
            os.system(f"colmap feature_extractor --database_path {database_path} --image_path {cache_img_dir} --ImageReader.mask_path {cache_mask_dir}")

            # Feature matching
            os.system(f"colmap exhaustive_matcher --database_path {database_path}")

            # Triangulation with known camera parameters
            dest_dir = str(data_dir / "data_3d_colmap" / "train" / seq / "static" / seg)
            os.makedirs(dest_dir, exist_ok=True)
            os.system(f"colmap point_triangulator --database_path {database_path} --image_path {cache_img_dir} --input_path {cache_dir} --output_path {dest_dir}")
            
            # Convert to PLY 
            os.system(f"colmap model_converter --input_path {dest_dir} --output_path {dest_dir} --output_type TXT")
            os.system(f"colmap model_converter --input_path {dest_dir} --output_path {dest_dir}/points3D.ply --output_type PLY")

            # remove cache
            shutil.rmtree(cache_dir)
            
            dest_dir = str(data_dir / "data_3d_colmap" / "train" / seq / "static" / seg)
            process_dir = str(data_dir / "data_3d_colmap_processed" / "train" / seq / "static" / seg)
            os.makedirs(process_dir, exist_ok=True)
            try:
                plydata = PlyData.read(dest_dir + "/points3D.ply")
            except:
                continue
            vertices = plydata['vertex']
            xyz = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            rgb = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
            
            # initialize points with open3d object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            
            # outlier removal
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.0)
            xyz = np.asarray(pcd.points)
            rgb = rgb[ind]

            # save ply
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            elements = np.empty(xyz.shape[0], dtype=dtype)
            elements[:] = list(map(tuple, np.concatenate([xyz, rgb], axis=1)))
            vertex_element = PlyElement.describe(elements, 'vertex')
            ply_data = PlyData([vertex_element])

            ply_data.write(process_dir + "/points3D.ply")


