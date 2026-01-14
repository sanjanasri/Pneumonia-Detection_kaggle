# Pneumonia-detection


This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 


The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal). 

+---------+--------+-----------+-------+
|  Split  | Normal | Pneumonia | Total |
+---------+--------+-----------+-------+
| Train   |  1342  |    3876   |  5218 |
| Val     |    9   |      9    |   18  |
| Test    |  235   |    391    |  626  |
+---------+--------+-----------+-------+


License: CC BY 4.0

Citation : http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

 ### Downloading the Kaggle dataset


You can download it using the Kaggle CLI:

1. Install and configure the Kaggle CLI (once):

   ```bash
   pip install kaggle

   mkdir -p data

   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia \
    -p data/ \
    --unzip
    
 After this, you should have:   
    data/
  chest_xray/
    train/
      NORMAL/
      PNEUMONIA/
    val/
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/
      
      If you ever use a dataset that has only train/ and test/ and you want to create a val/ split, you can run a small Python script once.
      We’ll create a 10% validation split from the training set.
      python scripts/split_train_val.py --data_dir data/my_xray_dataset --val_fraction 0.1
