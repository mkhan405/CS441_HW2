package com.khan.spark.SlidingWindows

import com.khan.spark.Embeddings.EmbeddingsCompute._

import com.khan.spark.Training.AppConfig
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}
import org.apache.hadoop.fs.{FileSystem, Path}
import java.net.URI

import scala.collection.Map

object Compute {
  val logger: Logger = LoggerFactory.getLogger("SlidingWindowCompute")

  private def generateSlidingWindows(sc: SparkContext, tokenizedRDD: RDD[String], tokenToEmbeddingBroadcast:
    Broadcast[Map[String, Array[Float]]], windowSize: Int, stride: Int, embeddingDim: Int, padToken: String,
    vocabSize: Long, tokenToIndexBroadcast: Broadcast[Map[String, Long]]) = {

    // Generate Input/Target Pairs for each tokenized line of text
    val samples = tokenizedRDD.map(line => {
      // If input length is less than the window size, pad the remaining space
      val windows = line.split(" ").sliding(windowSize + 1, stride).map(s => (s.slice(0, s.length - 1) ++ Array.fill(windowSize
        - s.length + 1)(padToken), s.last)).toArray
      logger.debug(s"Total window for line: ${windows.length}")
      windows
    })

    logger.info(s"Total sliding windows: ${samples.map(_.length).sum()}")

    // Extract list of inputs and targets
    val inputs = samples.map(S => S.map(s => s._1))
    val outputs = samples.map(S => S.map(s => s._2))

    // For each input, compute the positional and token embeddings
    val inputsWithPositionalEmbedding = inputs.map(in => {
      in.map(window => {
        val posEmbedding: INDArray = generatePositionalEmbedding(window, windowSize, embeddingDim)
        val embedding: INDArray = generateEmbedding(window, tokenToEmbeddingBroadcast, embeddingDim)
        embedding.add(posEmbedding)
      })
    })

    // For each target, generate one-hot encodings
    val outputEmbeddings = outputs.map(_.map(o => {
      // Retrieve the index of the token ID from the vocabulary
      val l: Long = tokenToIndexBroadcast.value.getOrElse(o, -1)

      // If not found, set label to list of zeros
      // Otherwise, set a 1 at the corresponding token index
      if (l == -1) {
        Nd4j.zeros(1, vocabSize)
      } else {
        val result = Nd4j.zeros(1, vocabSize)
        result.putScalar(0, l, 1.0)
        result
      }
    }))

    // Return dataset object with the
    inputsWithPositionalEmbedding.zip(outputEmbeddings).flatMap(T => {
      T._1.zip(T._2).map(t => new DataSet(t._1.reshape(1, embeddingDim * windowSize), t._2))
    })
  }


  def computeSlidingWindows(sc: SparkContext, config: AppConfig, tokenToIndexRDD: RDD[(String, Long)],
                            tokenToEmbeddingBroadcast:  Broadcast[Map[String, Array[Float]]]): RDD[DataSet] = {

    // Read Input Text from File
    // val textInputPath = getFilePath(sc.getConf.get("inputFilename", "inputs/input.txt"))
    val textInputPath = new Path(config.fileConfig.baseDir, config.fileConfig.inputTextPath).toString

    // Convert Text File to RDD
    val linesPerGroup = 10  // Number of lines to combine into one string
    val inputRDD: RDD[String] = sc.textFile(textInputPath)

    val groupedRDD: RDD[String] = inputRDD
      .zipWithIndex()  // Add indices to track line numbers
      .map { case (line, idx) => (idx / linesPerGroup, line) }  // Group by every N lines
      .groupByKey()  // Group the lines together
      .map { case (_, lines) => lines.mkString("\n") }  // Combine lines with newline separator

    val tokenizedRDD: RDD[String] = groupedRDD.map(s => splitAndTokenize(s))
    logger.info("Tokenized Output Computed")

    // logger.debug(s"Tokenized Output: ${tokenizedRDD.collect().mkString(",")}")
    // Token to Index Broadcast
    val tokenToIndexBroadcast = sc.broadcast(tokenToIndexRDD.collectAsMap())

    val dataset = generateSlidingWindows(sc, tokenizedRDD, tokenToEmbeddingBroadcast, config.windowSize, config.stride,
      config.embeddingDim, config.padToken, config.vocabSize, tokenToIndexBroadcast)
    val collectedDataset = dataset.collect()

    logger.debug(s"DataSet Head Features: ${collectedDataset.head.getFeatures.shape.mkString(",")}")
    logger.debug(s"DataSet Head Label: ${collectedDataset.head.getLabels.shape().mkString(",")}")

    logger.debug(s"First sliding window label: ${collectedDataset.head.getFeatures.getRow(0).toDoubleVector.mkString("Array(", ", ", ")")}")
    logger.debug(s"Total number processed: ${collectedDataset.length}")
    dataset
  }
}
