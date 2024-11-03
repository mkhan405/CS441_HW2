package com.khan.spark.Training

import com.typesafe.config.{Config, ConfigFactory}

case class SparkTrainingConfig(numWorkers: Int, batchSize: Int, avgFreq: Int, numPrefetchBatches: Int,
       numEpochs: Int, learningRate: Double, hiddenLayerSize: Int, seed: Int)
case class FileConfig(inputTextPath: String, embeddingFilePath: String, baseDir: String, outputFilePath: String)
case class AppConfig(windowSize: Int, stride: Int, embeddingDim: Int, padToken: String, vocabSize: Long,
   sparkTrainingConfig: SparkTrainingConfig, fileConfig: FileConfig)

object AppConfig {
  private val config: Config = ConfigFactory.load()

  def load(): AppConfig = {
    val sparkTrainingConfig = SparkTrainingConfig(
      numWorkers = config.getInt("app.sparkModelConfig.numWorkers"),
      batchSize = config.getInt("app.sparkModelConfig.batchSize"),
      avgFreq = config.getInt("app.sparkModelConfig.avgFreq"),
      numPrefetchBatches = config.getInt("app.sparkModelConfig.numPrefetchBatches"),
      numEpochs = config.getInt("app.sparkModelConfig.numEpochs"),
      learningRate = config.getDouble("app.sparkModelConfig.learningRate"),
      hiddenLayerSize = config.getInt("app.sparkModelConfig.hiddenLayerSize"),
      seed = config.getInt("app.sparkModelConfig.seed")
    )

    val fileConfig = FileConfig(
      inputTextPath = config.getString("app.fileConfig.inputTextPath"),
      embeddingFilePath = config.getString("app.fileConfig.embeddingFilePath"),
      baseDir = config.getString("app.fileConfig.baseDir"),
      outputFilePath = config.getString("app.fileConfig.outputFilePath")
    )

    AppConfig(
      windowSize = config.getInt("app.windowSize"),
      stride = config.getInt("app.stride"),
      padToken = config.getString("app.padToken"),
      embeddingDim = config.getInt("app.embeddingDim"),
      vocabSize = config.getLong("app.vocabSize"),
      sparkTrainingConfig = sparkTrainingConfig,
      fileConfig = fileConfig)
  }
}