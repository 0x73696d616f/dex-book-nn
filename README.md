# GRU neural network 

Neural network development for on chain Bitcoin price prediction.

## Prediction Results

Using a look back parameter of 10, which means it looks at the average prices of each of the last 10 days, it achieves the results beloww. A dataset with Bitcoin/USD prices was fetched from [Kaggle](https://www.kaggle.com/) and the data split into 64% training, 16% validation and 20% testing. 

![prediction results](results.png)

## Storing in the blockchain

The model is split into chunks which are then stored in a smart contract using SSTORE2 from solmate's library.

## Apothem address

https://apothem.blocksscan.io/address/0xD87D11f9832e39E4394D4118766ed9e76188e51A#transactions