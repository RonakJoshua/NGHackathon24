# Important files 
`requirements.txt`
`final_output.ipynb`(`finalRevised.ipynb`)
`finetune.ipynb` 
`extract.sh`(`extractWindows.sh`)

# Getting Started
Please download the model weights [here](https://drive.google.com/file/d/1v9GcJYwForenvSveiC1kfHYkwQxF79Hk/view) (it is a Google Drive link, the .pth file is too large to commit to a Github repository). Add the .pth model directly to the root directory of the repository.

Once you've downloaded and added the .pth file to the directory, set up and activate a python virtual environment (virtualenv or pyenv are great options). Next, run `pip install -r requirements.txt`. Now, you can run the `final_output.ipynb` file.

If you are utilizing a Nvidia GPU, using the `finalRevised.ipynb` file may run smoother 

To test an image, input the path of that image into section that states "# CHANGE THIS PATH TO CHANGE THE INPUT IMAGE" in the middle of the last module. NOTE: Python dash lines go the opposite way so remember to change it. 



# Training the finetune Model
Run `finetune.ipynb`
To run the `finetune.ipynb` file, you must download the Imagenet mini dataset first.

# Download The Imagenet Mini Dataset
The test and validation files of the Imagenet mini dataset can be found on Kaggle [here](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data). Place the fully extracted `tran` and `val` directories inside of a `imagenet/images` directory.

The bounding boxes and annotation files can be found on the Imagenet website [here](https://image-net.org/download-bboxes). The bounding box files are all compressed in .tar.gz files. To extract each folder, utilize the `extract.sh` file. Place the fully extracted `bboxes_annotations` directory inside of the `imagenet` directory.

If you are on WINDOW use the `extractWindows.sh` file in gitbash in your downloads folder. To do this, type "nano extract_bboxes.sh" into gitbash. copy and paste the code onto the file. press CTRL + X, then Y, and Enter to save the file, then type "chmod +x extract_bboxes.sh", lastly type "./extract_bboxes.sh" to run script. Place the fully extracted `bboxes_annotations` directory inside of the `imagenet` directory.

ImageNet mini images and annotations are not included since they take up >4GB of storage. 


# Team 10 ---
Team members
  Lam Pham ​- Senior(cmpsc)
  Ravi Patel​ ​- Sophmore(cmpsc)
  Ronak Singh​ ​- Senior(cmpsc)
  Brandon Widjaja ​- Sophmore(cmpsc)

# About
Team 10 is competing against 13 other teams in the N.G.-Hackathon 2024.
Objective: Develop an object detection program that involves finding and locating everyday     objects within an image​

Code structure consists of a faster R-CNN pretrained COCO model, an ImageNet mini dataset. 
Extract imageNet data set, apply random transformation for training, make a faster R-CNN based on imageNet dataset, merge evaluations from both COCO and ImageNet models. 

