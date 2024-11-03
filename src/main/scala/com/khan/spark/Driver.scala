package com.khan.spark

import com.khan.spark.Training.AppConfig
import com.khan.spark.SlidingWindows.Compute._
import com.khan.spark.Embeddings.EmbeddingLoader._
import com.khan.spark.Training.Transformer._
import com.khan.spark.utils.io.configureFileSystem
import com.typesafe.config.ConfigFactory
import org.apache.spark._
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}

import org.apache.hadoop.fs.FileSystem

object Driver {
  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf().setAppName("CS 411 HW2").setMaster("local[*]")
    val config = AppConfig.load()

    // Set Training Config
    val sparkContext: SparkContext = new SparkContext(conf)
    val fs: FileSystem = configureFileSystem(sparkContext, config.padToken)

    val logger: Logger = LoggerFactory.getLogger("Driver")
    logger.info("Driver Started...")

    val (tokenToEmbeddingRDD, embeddingToTokenRDD, indexToIndexRDD, tokenToIndexRDD) =
      loadEmbeddings(sparkContext, config)

    logger.info("Embedding RDD Computed")
    // Report Spark Statistics
    logger.info(s"Number of Executors: ${sparkContext.getExecutorMemoryStatus.size}")

    val tokenToEmbeddingBroadcast = sparkContext.broadcast(tokenToEmbeddingRDD.collectAsMap())
    val indexToTokenBroadcast = sparkContext.broadcast(indexToIndexRDD.collectAsMap())
    val dataset = computeSlidingWindows(sparkContext, config, tokenToIndexRDD, tokenToEmbeddingBroadcast)
    val model = train(sparkContext, fs, config, dataset)

    val output = predict(model, "serve power", config, tokenToEmbeddingBroadcast, indexToTokenBroadcast)
    logger.debug(s"Prediction Output: ${output}")
    sparkContext.stop()
  }
}