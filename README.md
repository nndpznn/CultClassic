# CultClassic - A Movie Buff's Best Friend
"Top X Movies to Watch" lists are no fun anymore. Everyone's seen 'em. But you'd be crazy to think that those are all the good movies left in the world... but how to go about finding them? Welcome to Cult Classic, a pet project from Nolan Nguyen and Allen Ho, designed to help you find your next film fixation. Harnessing the power of other users with "underground" tastes, our engine aims to deliver the best non-mainstream movies right to your feet. Let's find something new together!

## HOW TO USE:
- Use 'python3 -m scripts.train' to call the training function.
- Use 'python3 -m scripts.test' to call the evaluation function.
- Use 'python3 -m scripts.testNuser user-id top-k-items' to call the recommendation function, to get user n's top k recommended movies by the model.

## TODO:
- TRAINING: Create scripts to train the model on real data, cost/loss function, adjusting weights, etc.
- EVALUATION: Create scripts to "use" the model on new data and present accuracy statistics. RMSE, NDCG, Precision@k, Recall@k? 
- OUTPUT: For output later - find a way to map movie IDs to movie names so that the output movies actually make sense.

## Credits
- Thank you to the [Codemy.com](https://www.youtube.com/@Codemycom) YouTube channel for beginner's guides on the basics of neural networks through PyTorch.
