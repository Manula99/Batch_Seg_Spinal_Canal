import os

def spinal_cord_segmentation(img_path, output_im):
    #run SCT deeplearning for spinal canal segmentation
    from spinalcordtoolbox.scripts import sct_deepseg
    sct_deepseg.main(argv=['sc_canal_t2', '-i', img_path,
                     '-o', output_im])

def vis_seg(img_path, seg_path):
    import nibabel as nib
    import numpy as np
    from matplotlib import pyplot as plt

    img = nib.load(img_path)
    arr = np.asanyarray(img.dataobj)

    seg = nib.load(seg_path)
    seg_arr = np.asanyarray(seg.dataobj)

    plt.figure(figsize=(8, 8))
    #adjust orientation of slices
    plt.imshow(np.rot90(arr), cmap="Greys_r", clim=(0, 150))
    #adjust segmentation opacity to overlay mask over image
    plt.imshow(np.rot90(seg_arr), cmap="Reds", clim=(0.9, 1), alpha=0.5)
    plt.axis('off')
    seg_dir = os.path.dirname(seg_path)
    plt.savefig(os.path.join(seg_dir, "seg_qc.png"))

def canal_seg_slurm(input_img, seg_path, job_name, python_env, node_list, partition):
	from subprocess import Popen, PIPE
	output_dir = os.path.dirname(seg_path)
	sbatch_script = ("#!/bin/bash \n"
                      f"#SBATCH --job-name={job_name} \n"
                      f"#SBATCH --output={os.path.join(output_dir, f'{job_name}.out')} \n"
                      f"#SBATCH --error={os.path.join(output_dir, f'{job_name}.err')} \n"
                      "#SBATCH --time=0:20:00 \n"
                      "#SBATCH -N 1 \n"
                      "#SBATCH --cpus-per-task=1 \n"
                      "#SBATCH --mem=11G \n"
					  f"#SBATCH --nodelist={','.join(node_list)} \n"
					  f"#SBATCH --partition={partition} \n"
                      f"conda activate {python_env} \n"
					  f"python run_canal_seg.py spawn {input_img} {seg_path} \n"
					  f"python run_canal_seg.py vis {input_img} {seg_path} \n")
	#suffix = datetime.datetime.now().isoformat()
	with open(f"canal_seg.slurm", "w") as f:
		f.write(sbatch_script)
	sbatch_command = (f"sbatch canal_seg.slurm")
	#print(sbatch_command)
	process = Popen(sbatch_command, stdout=PIPE, stderr=PIPE, env=os.environ, shell=True)
	stdout, stderr = process.communicate()
	print(stdout)
	print(stderr)

if __name__ == "__main__":
    import sys

    mode = sys.argv[1]
    if mode == 'vis':
        img_path = sys.argv[2]
        seg_path = sys.argv[3]
        vis_seg(img_path, seg_path)
    if mode == 'spawn':
        img_path = sys.argv[2]
        seg_path = sys.argv[3]
        spinal_cord_segmentation(img_path, seg_path)