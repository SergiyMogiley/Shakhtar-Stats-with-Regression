using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
public class GameStats
{
    [LoadColumn(0)]
    public short Location;

    [LoadColumn(1)]
    public short Possession;

    [LoadColumn(2)]
    public double Passes;

    [LoadColumn(3)]
    public short ShotsOnTarget;

    [LoadColumn(4)]
    public short Corners;

    [LoadColumn(5)]
    public short Fouls;

    [LoadColumn(6)]
    public short Offsides;

    [LoadColumn(7)]
    public short WonChallenges;

    [LoadColumn(8)]
    public short Result;

    [LoadColumn(9)]
    public float GoalsScored;
}

public class GameResultPrediction
{
    [ColumnName("Score")]
    public float GoalsScored;
}
