# Pneumonia Classification
Group project for 50.039 Theory and Practice of Deep Learning, SUTD Term 6
<br>
## Introduction
For this project we receive grayscale X-ray images of the chest area and attempt to classify whether the patient is healthy (class 0), or has pneumonia (class 1). Childhood pneumonia is the number one cause of childhood mortality due to infectious diseases (Rudan et al. ,2008). A quick diagnosis and urgent treatment is needed in order to save the child. A Deep learning model that can detect signs of pneumonia can help to provide rapid diagnosis and referrals for these children (Kermany et al, 2018).
<br>
<br> In the repository, there are notebooks numbered from 0 to 7. Although they have been seeded, their outcomes may not be deterministic due to batch normalization and dropout techniques. Hence, while notebooks 0 to 5 can be run, they may not produce the necessary files to run all notebooks numbered 6. Therefore, we advise against running them and have included the notebook numbered 7 to verify our best performing models which can be downloaded [here](https://drive.google.com/drive/folders/1zcXmKO0L9nvmTLk23JpvpBmgLlQouqLa?usp=sharing).
<br>
## Further Considerations
1. Continued training of CNN architecture with higher patience value for EarlyStopper, as validation loss could end up converging much later.
<br> 2. While random horizontal flip theoretically introduces variance, it may not be the objectively correct thing to do. This is because the heart would end up on the right side of the chest, and consequently the right lung would appear smaller than it should.
<br> 3. Using the recall or f1 score as a regularization term could improve the model's performance while minimising false negatives.
<br> 4. Integration of Resnet34 (or other architecture) weights with our other models might yield better performance.
<br>
