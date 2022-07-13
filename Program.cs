using Microsoft.ML;

string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Shakhtar_15_16_train.csv");
string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Shakhtar_15_16_test.csv");
string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model_Shakhtar.zip");

MLContext mlContext = new MLContext(seed: 0);

var model = Train(mlContext, _trainDataPath);

ITransformer Train(MLContext mlContext, string dataPath)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<GameStats>(dataPath, hasHeader: true, separatorChar: ',');

    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "GoalsScored")

        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "LocationEncoded", inputColumnName: "Location"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PossessionEncoded", inputColumnName: "Possession"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PassesEncoded", inputColumnName: "Passes"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ShotsOnTargetEncoded", inputColumnName: "ShotsOnTarget"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "CornersEncoded", inputColumnName: "Corners"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "FoulsEncoded", inputColumnName: "Fouls"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "OffsidesEncoded", inputColumnName: "Offsides"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "WonChallengesEncoded", inputColumnName: "WonChallenges"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ResultEncoded", inputColumnName: "Result"))

        .Append(mlContext.Transforms.Concatenate("Features", "LocationEncoded", "PossessionEncoded", "PassesEncoded", "ShotsOnTargetEncoded",
                                                 "CornersEncoded", "FoulsEncoded", "OffsidesEncoded", "WonChallengesEncoded", "ResultEncoded"))

    .Append(mlContext.Regression.Trainers.FastTree());

    var model = pipeline.Fit(dataView);

    mlContext.Model.Save(model, dataView.Schema, _modelPath);

    return model;
}

Evaluate(mlContext, model);

void Evaluate(MLContext mlContext, ITransformer model)
{
    IDataView dataView = mlContext.Data.LoadFromTextFile<GameStats>(_testDataPath, hasHeader: true, separatorChar: ',');

    var predictions = model.Transform(dataView);

    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");

    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");

    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
}

TestSinglePrediction(mlContext, model);

void TestSinglePrediction(MLContext mlContext, ITransformer model)
{
    var predictionFunction = mlContext.Model.CreatePredictionEngine<GameStats, GameResultPrediction>(model);

    var GameStatsSample = new GameStats()
    {
        Location = 1,
        Possession = 20,
        Passes = 45.04,
        ShotsOnTarget = 4,
        Corners = 1,
        Fouls = 20,
        Offsides = 3,
        WonChallenges = 15,
        Result = 3,
        GoalsScored = 0
    };

    var prediction = predictionFunction.Predict(GameStatsSample);

    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted Goals Scored: {prediction.GoalsScored:0.####}, actual Goals Scored: 3");
    Console.WriteLine($"**********************************************************************");
}