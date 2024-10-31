# Getting Started
Please download the model weights [here]https://drive.google.com/file/d/1v9GcJYwForenvSveiC1kfHYkwQxF79Hk/view (it is a Google Drive link, the .pth file is too large to commit to a Github repository). Add the .pth model directly to the root directory of the repository.

Once you've downloaded and added the .pth file to the directory, set up and activate a python virtual environment (virtualenv or pyenv are great options). Next, run `pip install -r requirements.txt`. Now, you can run the `final_output.ipynb` file.

To run the `finetune.ipynb` file, you must download the Imagenet mini dataset first.

# Download The Imagenet Mini Dataset
The test and validation files of the Imagenet mini dataset can be found on Kaggle [here]https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/data. Place the fully extracted `tran` and `val` directories inside of a `imagenet/images` directory.

The bounding boxes and annotation files can be found on the Imagenet website [here]https://image-net.org/download-bboxes. The bounding box files are all compressed in .tar.gz files. To extract each folder, utilize the `extract.sh` file. Place the fully extracted `bboxes_annotations` directory inside of the `imagenet` directory.

ImageNet mini images and annotations are not included since they take up >4GB of storage. 
