- Try 31, first calculating with physical path loss and then predicting error. Use better formulas, formulas like Okumura-Hata, or better. Research which ones and implement them.
Of course first looking at the topology and then implementing one formula or another, also taking into account the height and LoS!

- Also feeding the "regularly computed" map as an input to the network, because this way it can know if its LoS or kind of city.

- Not predicting where there is NO DATA (just ignoring it, NOT putting 180), which is where are buildings, buildings are not 180dBs pathloss, they shouldn't be used for anything, for the map, for anything. Nor for the error. (But yes where is not Line of Sight). Dont predict where topology is not =0, but yes where there is no LoS but topology=0

- For the error calculation, only take into account where we originally have measured path loss (related to the last one!)

- For the formulas check "2 ray model" (for LoS it will be good), it will help with the radial thing. Also receiver height is not 0m, but 1.5m