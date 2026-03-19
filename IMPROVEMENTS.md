Things found day 1:

- Postprocessing: The path loss exponent decreases with higher height. Maybe doing some checking algorithm.
![alt text](vinogradov_height_formula.png)

- Preprocessing: City detection, because the path loss formula depends on type of cities. Maybe with the number or height of buildings. And distribution, percentage of it. Train different models depending on type of city and classify the image before processing it.

- Train a BIGGER model, with more parameters.

- Giving me the confidence of the path loss based on the error by groundtruth.. So calculate it not with deep learning but algorithms if the confidence is low. Where the model fails, or gets a very wrong value, put low confidence (after getting the results from the model). And then we train again with the ground truth of confidence (based on errors), predicting the confidence, and if the confidence is low, path loss is calculated algorithmically.

- Path loss is data missing because sometimes it was too high (so no signal). This affected delay spread and angular spread. But Line of Sight will always exist.

- I'm normalizing the values with the highest and minimum of the dataset. Maybe not doing it that way, if the value goes higher than the highest of the dataset can give problems (and lowest). And also maybe predict the path loss linearly, converting it to linear before training the model, because that way the loss will be higher, it will be more varied data.

Things found 2:

- Train one model for LoS and one for not LoS (it's an input matrix).

- (DON'T DO THIS ONE Right Now) Normalize the height now that we have the antenna height parameter. Train one model with height normalized and one with NOT normalized (the normalized one with the drone height as the max).

- Be careful with very high path loss (infinite or NaN) in the dataset.

- Look at the data, don't rely on bigger models or more data.

