package com.khan.spark.SlidingWindows

import com.khan.spark.Embeddings.EmbeddingInstance
import com.khan.spark.utils.io.getResourcePath
import com.khan.spark.utils.EncodingData._
import com.knuddels.jtokkit.api.IntArrayList
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.Map

import scala.collection.mutable.ArrayBuffer

object Compute {

  private val embeddingSchema = StructType(
    Array(
      StructField("name", StringType),
      StructField("embedding", ArrayType(StringType)) // Assuming embeddings are float arrays
    )
  )

  val logger: Logger = LoggerFactory.getLogger("SlidingWindowCompute")

  private def splitAndTokenize(str: String): String = {
    str.split(" ").map(t => encoding.encode(t)).map(bpe => bpe.toArray).map(_.mkString(" ")).mkString(" ")
  }

  private def initializeEmbeddings(embeddingPath: String): (RDD[(String, Array[Float])], RDD[(Array[Float], String)]) = {
    val spark = SparkSession.builder.appName("Simple Application")
      .config("spark.master", "local")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .master("local[*]")
      .getOrCreate()


      val dfRdd = spark.read.format("csv")
        //.schema(embeddingSchema)
        .option("header", "true") //first line in file has headers
        .option("inferSchema", "true") //first line in file has headers
        .load(embeddingPath)
        .rdd

      // Parse Embedding CSV Data
      val embeddingRdd = dfRdd.map(r => new EmbeddingInstance(r.get(0).toString, r.get(1).
        toString.split(";").map(_.toFloat)))

      // Generate token -> embedding and embedding -> token maps
      val tokenToEmbeddingRDD = embeddingRdd.map(i => (i.token, i.vector))
      val embeddingToTokenRDD = embeddingRdd.map(i => (i.vector, i.token))

      (tokenToEmbeddingRDD, embeddingToTokenRDD)
  }

  private def generatePositionalEmbedding(tokens: Array[String], windowSize: Int, embeddingDim: Int): INDArray = {
    val result: INDArray = Nd4j.zeros(windowSize, embeddingDim)
    (0 until windowSize).foreach(pos => {
      (0 until embeddingDim by 2).foreach(i => {
        val angle: Double = pos / Math.pow(10000, (2.0 * i) / embeddingDim)
        result.putScalar(Array(pos, i), Math.sin(angle))
        if (i + 1 < embeddingDim)
          result.putScalar(Array(pos, i + 1), Math.cos(angle))
      })
    })
    result
  }

  private def generateEmbedding(tokens: Array[String], tokenToEmbeddingRDD:Broadcast[Map[String, Array[Float]]],
                                embeddingDim: Int): INDArray = {

    // Convert string of Integer Token IDs -> IntArrayList
    val intTokens = new IntArrayList()
    tokens.foreach(t => intTokens.add(t.toInt))

    // For each decoded integer ID, generate embedding
    val embeddings: Array[Float] = encoding.decode(intTokens).split(" ").flatMap(w => {
      tokenToEmbeddingRDD.value.getOrElse(w, Array.fill(embeddingDim)(0.0f))
    })

    Nd4j.create(embeddings)
  }

  private def generateSlidingWindows(sc: SparkContext, tokenizedRDD: RDD[String], tokenToEmbeddingRDD: RDD[(String, Array[Float])],
         windowSize: Int, stride: Int, embeddingDim: Int, padToken: String) = {

    val tokenToEmbeddingBroadcast = sc.broadcast(tokenToEmbeddingRDD.collectAsMap())

    val samples = tokenizedRDD.map(line => {
      logger.debug(s"Computing Line: ${line}")
      line.split(" ").sliding(windowSize + 1, stride).map(s => (s.slice(0, s.length - 1) ++ Array.fill(windowSize
        - s.length + 1)(padToken), s.last)).toArray
    })

    val inputs = samples.map(S => S.map(s => s._1))
    val outputs = samples.map(S => S.map(s => s._2))

    val inputsWithPositionalEmbedding = inputs.map(in => {
      in.map(window => {
        val posEmbedding: INDArray = generatePositionalEmbedding(window, windowSize, embeddingDim)
        val embedding: INDArray = generateEmbedding(window, tokenToEmbeddingBroadcast, embeddingDim)
        embedding.add(posEmbedding)
      })
    })

    val outputEmbeddings = outputs.map(_.map(o => generateEmbedding(Array(o), tokenToEmbeddingBroadcast, embeddingDim)))
    inputsWithPositionalEmbedding.zip(outputEmbeddings).flatMap(T => {
      T._1.zip(T._2).map(t => new DataSet(t._1, t._2))
    })
  }


  def computeSlidingWindows(sc: SparkContext): Unit = {
    // Read Input Text from File
    val inputFilename = s"/${sc.getConf.get("inputFilename", "inputs/input.txt")}"
    val inputFilePath = getResourcePath(inputFilename)
    // Read Embeddings from CSV
    val embeddingsFilename = s"/${sc.getConf.get("embeddingFilename", "inputs/embeddings.csv")}"
    val embeddingPath = getResourcePath(embeddingsFilename)

    if (inputFilePath == null || embeddingPath == null) {
      val missingFilename = if (inputFilePath.equals(null)) inputFilename else embeddingsFilename
      logger.error(s"File: ${missingFilename} cannot be opened")
      return
    }

    // Convert Text File to RDD
    val inputRDD: RDD[String] = sc.textFile(inputFilePath)
    val tokenizedRDD: RDD[String] = inputRDD.map(s => splitAndTokenize(s))
    logger.info("Tokenized Output Computed")
    // logger.debug(s"Tokenized Output: ${tokenizedRDD.collect().mkString(",")}")
    
    // Load Embeddings and generate maps
    val (tokenToEmbeddingRDD, embeddingToTokenRDD) = initializeEmbeddings(embeddingPath)
    logger.info("Embedding RDD Computed")

    // Loading Training Configuration Parameters
    val windowSize: Int = sc.getConf.get("windowSize", "20").toInt
    val stride: Int = sc.getConf.get("stride", "3").toInt
    val padToken: String = sc.getConf.get("pad_token", "0")
    val embeddingDim: Int = sc.getConf.get("embeddingDim", "128").toInt

    val dataset = generateSlidingWindows(sc, tokenizedRDD, tokenToEmbeddingRDD, windowSize, stride, embeddingDim, padToken)
    val collectedDataset = dataset.collect()

    logger.debug(s"DataSet Head Features: ${collectedDataset.head.getFeatures.length()}")
    logger.debug(s"DataSet Head Label: ${collectedDataset.head.getLabels.length()}")

    logger.info(s"If head features are of correct length: ${collectedDataset.forall(d => d.getFeatures.length() == 2560)}")
    logger.info(s"If head labels are of correct length: ${collectedDataset.forall(d => d.getLabels.length() == 128)}")
  }
}
