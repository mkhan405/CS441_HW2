package com.khan.spark.SlidingWindows


import com.khan.spark.Embeddings.EmbeddingsCompute._
import com.khan.spark.Training.AppConfig
import com.khan.spark.utils.EncodingData.encoding

import com.knuddels.jtokkit.api.IntArrayList
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

  /**
   * Generates input/target pairs for training from an RDD of space-delimited BPE tokens, creating sliding windows of specified size.
   *
   * @param sc The SparkContext used for parallel processing.
   * @param tokenizedRDD An RDD[String] where each string represents a sequence of BPE tokens, space-delimited.
   * @param tokenToEmbeddingBroadcast A Broadcast variable containing a mapping from token strings to their corresponding float array embeddings.
   * @param windowSize The size of the sliding window used to create input sequences.
   * @param stride The number of tokens to shift the window after each extraction, determining the overlap between consecutive windows.
   * @param embeddingDim The dimensionality of the token embeddings.
   * @param padToken The token used to pad sequences that are shorter than the window size.
   * @param vocabSize The total number of unique tokens in the vocabulary.
   * @param tokenToIndexBroadcast A Broadcast variable mapping token strings to their corresponding indices in the vocabulary.
   *
   * @return An RDD[DataSet] containing the generated input and target pairs, where each input is a concatenated array of token embeddings (with positional information)
   *         and each target is a one-hot encoded representation of the last token in the sliding window.
   *
   * This function processes each line of tokenized text to create overlapping input sequences and corresponding target tokens
   * using a sliding window approach. Each input is augmented with positional embeddings and the targets are transformed into
   * one-hot encodings. If a target token is not found in the vocabulary, a zero vector is returned for that target.
   *
   * Note: The output DataSet for each input/target pair is reshaped to ensure compatibility with the neural network input requirements.
   */
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
    val posEmbedding: INDArray = generatePositionalEmbedding(windowSize, embeddingDim)

    // For each input, compute the positional and token embeddings
    val inputsWithPositionalEmbedding = inputs.map(in => {
      in.map(window => {
        val embedding: INDArray = generateEmbedding(window, tokenToEmbeddingBroadcast, embeddingDim)
        embedding.add(posEmbedding)
      })
    })

    // For each target, generate one-hot encodings
    val outputEmbeddings = outputs.map(_.map(o => {
      // Get textual representation of token ID
      val tokenList = new IntArrayList()
      tokenList.add(o.toInt)
      val token = encoding.decode(tokenList)
      // Retrieve the index of the token ID from the vocabulary
      val l: Long = tokenToIndexBroadcast.value.getOrElse(token, -1)

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


  /**
   * Computes sliding windows of tokenized text input, converting it into an RDD of DataSet objects for training.
   *
   * @param sc The SparkContext used for parallel processing.
   * @param config An instance of AppConfig containing configuration parameters such as file paths.
   * @param tokenToIndexRDD An RDD containing pairs of tokens and their corresponding indices in the vocabulary.
   * @param tokenToEmbeddingBroadcast A Broadcast variable containing a mapping from token strings to their corresponding
   *                                  float array embeddings.
   *
   * @return An RDD[DataSet] where each DataSet object contains input features as token embeddings with positional information
   *         and corresponding target labels for training a neural network.
   *
   * This function reads input text from a specified file, groups the lines according to the configuration, and processes them
   * into tokenized strings. It then generates sliding windows of inputs and targets, encapsulated as DataSet objects suitable for
   * use in machine learning workflows. The function also logs various stages of processing for debugging purposes, including
   * the shapes of the dataset features and labels.
   *
   * Note: The function assumes that the input text is formatted appropriately and that the tokenization method is defined in
   * the `splitAndTokenize` function.
   */
  def computeSlidingWindows(sc: SparkContext, config: AppConfig, tokenToIndexRDD: RDD[(String, Long)],
                            tokenToEmbeddingBroadcast:  Broadcast[Map[String, Array[Float]]]): RDD[DataSet] = {

    // Read Input Text from File
    // val textInputPath = getFilePath(sc.getConf.get("inputFilename", "inputs/input.txt"))
    val textInputPath = new Path(config.fileConfig.baseDir, config.fileConfig.inputTextPath).toString

    // Convert Text File to RDD
    val linesPerGroup = config.lineGrouping  // Number of lines to combine into one string
    val inputRDD: RDD[String] = sc.textFile(textInputPath)

    val groupedRDD: RDD[String] = inputRDD
      .zipWithIndex()  // Add indices to track line numbers
      .map { case (line, idx) => (idx / linesPerGroup, line) }  // Group by every N lines
      .groupByKey()  // Group the lines together
      .map { case (_, lines) => lines.mkString("\n") }  // Combine lines with newline separator

    val tokenizedRDD: RDD[String] = groupedRDD.map(s => splitAndTokenize(s))
    logger.info("Tokenized Output Computed")

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
