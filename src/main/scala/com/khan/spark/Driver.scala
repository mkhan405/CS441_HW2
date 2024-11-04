package com.khan.spark

import com.khan.spark.Training.AppConfig
import com.khan.spark.SlidingWindows.Compute._
import com.khan.spark.Embeddings.EmbeddingLoader._
import com.khan.spark.Training.Transformer._
import com.khan.spark.utils.io.configureFileSystem
import org.apache.spark._
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

    val (tokenToEmbeddingRDD, indexToIndexRDD, tokenToIndexRDD) =
      loadEmbeddings(sparkContext, config)

    logger.info("Embedding RDD Computed")

    val tokenToEmbeddingBroadcast = sparkContext.broadcast(tokenToEmbeddingRDD.collectAsMap())
    val indexToTokenBroadcast = sparkContext.broadcast(indexToIndexRDD.collectAsMap())
    val dataset = computeSlidingWindows(sparkContext, config, tokenToIndexRDD, tokenToEmbeddingBroadcast)
    val model = train(sparkContext, fs, config, dataset)

    val attempts = Array("express power", "since war", "free situation")
    attempts.foreach(input => {
      val output = predict(model, input, config, tokenToEmbeddingBroadcast, indexToTokenBroadcast)
      logger.debug(s"Input: ${input}, Prediction Output: ${output}")
    })

    sparkContext.stop()
  }
}