# Shakhtar-Stats-with-Regression
Analysis of Shakhtar Donetsk statistics (season 2015-16) using regression and ML.NET

Problem: using the labeled data of football statistics the machine is to determine the quantity of scored goals in the next (unlabeled) game.

Means: Visual Studio, ML.NET (C#/.NET).

ML-algorithm: regression.

The example of code and data analysis using the chosen ML-algorithm:

https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/predict-prices

Data (source):
-	Football team (web-site): «Shakhtar Donetsk» (https://shakhtar.com/)
-	Season: 2015-16.

Data of csv-files decoding:
Features:
-	location: home (1) or guest (0) game;
-	possession: percentage of ball possession;
-	passes: quantity of passes;
-	shots_on_target: quantity of shots on target;
-	corners: quantity of corners;
-	fouls: quantity of fouls;
-	offsides: quantity of offsides;
-	won_challenges: quantity of won challenges;
-	win_loss_draw: the result of the game – victory (3), draw (1) or defeat (0).

Label:
-	goals_scored – quantity of scored scores.

Part of the training sample data: about 85%.

Results:

Training sample to testing sample: 21:3 (87.5% to 12.5%).

RSquared Score: -3.4; Root Mean Squared Error: 0.99;

Predicted goals scored: 2.9; actual goals scored: 3.

Conclusions:

The prediction is quite accurate.

Links:

-	Github-repository: https://github.com/SergiyMogiley/Shakhtar-Stats-with-Regression

-	LinkedIn: https://www.linkedin.com/in/sergiy-mogiley-5348a615b/
