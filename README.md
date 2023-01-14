## Alzheimers Classification using Brain MRI's

### Usage

#### Dependencies
` pip install -r requirements.txt`

#### Training 
Navigate to `src/`

`python train.py --model model_name --epochs no_of_epochs`

Parameters: <br>
--epochs number of epochs <br>
--lr learning rate <br>
--batch batch size <br>
--model model_name <br>
--num_classes number of classes <br>
--gpu to use gpu <br>
--optimizer type of optimizer, default:adamax
--logger tensorboard(tb) or wandb(eb) <br>


### Dataset
Download the dataset from the link below to 'data/' directory
Kaggle : [Alzheimers Brain MRI](https://www.kaggle.com/datasets/basheersaeed/alzheimers-brain-mri)

#### Description
    -   Number of Classes : 3
    -   Image Size: 2048 x 2048
    -   Total Images : 600
        -   AD : 200
        -   MCI : 200
        -   CN : 200

### Models
 -  EfficientNet
 -  ResNet50
 -  ViT
 -  ConvNeXt


