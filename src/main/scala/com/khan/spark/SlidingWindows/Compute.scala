package com.khan.spark.SlidingWindows

import com.khan.spark.Embeddings.EmbeddingInstance
import com.khan.spark.utils.io.getResourcePath
import com.khan.spark.utils.EncodingData._
import com.knuddels.jtokkit.api.IntArrayList
import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, StructType}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer

object Compute {

  private val embeddingSchema = StructType(
    Array(
      StructField("name", StringType),
      StructField("embedding", ArrayType(StringType)) // Assuming embeddings are float arrays
    )
  )

  val logger: Logger = LoggerFactory.getLogger("SlidingWindowCompute")

  private def splitAndTokenize(str: String): Array[String] = {
    str.split(" ").map(t => encoding.encode(t)).flatMap(bpe => bpe.toArray).map(_.toString)
  }

  private def initializeEmbeddings(embeddingPath: String): (RDD[(String, Array[Float])], RDD[(Array[Float], String)]) = {
    val spark = SparkSession.builder.appName("Simple Application")
      .config("spark.master", "local")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .master("local[*]")
      .getOrCreate()

    try {
      // Convert Embeddings to Dataframe
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
    } finally {
      spark.stop()
    }
  }

  private def generatePositionalEmbedding(tokens: Array[String], windowSize: Int, embeddingDim: Int): INDArray = {
    val result: INDArray = Nd4j.zeros(windowSize, embeddingDim)
    (0 until windowSize).foreach(pos => {
      (0 until embeddingDim).foreach(i => {
        val angle: Double = pos / Math.pow(10000, (2.0 * i) / embeddingDim)
        result.putScalar(Array(pos, i), Math.sin(angle))
        result.putScalar(Array(pos, i + 1), Math.cos(angle))
      })
    })
    result
  }

  private def generateEmbedding(tokens: Array[String], tokenToEmbeddingRDD: RDD[(String, Array[Float])],
      embeddingDim: Int): INDArray = {
//    val embeddings: Array[Float] = tokens.flatMap(t => {
//      tokenToEmbeddingRDD.lookup(t) match {
//        case Seq() => Array.fill(embeddingDim)(0.0f)
//        case values => values.head
//      }
//    })

    // Convert string of Integer Token IDs -> IntArrayList
    val intTokens = new IntArrayList()
    tokens.foreach(t => intTokens.add(t.toInt))

    // For each decoded integer ID, generate embedding
    val embeddings: Array[Float] = encoding.decode(intTokens).split(" ").flatMap(w => {
      tokenToEmbeddingRDD.lookup(w) match {
        case Seq() => Array.fill(embeddingDim)(0.0f)
        case values => values.head
      }
    })

    Nd4j.create(embeddings)
  }

  private def generateSlidingWindows(tokenizedRDD: RDD[String], tokenToEmbeddingRDD: RDD[(String, Array[Float])],
         windowSize: Int, stride: Int) = {
    val samples = tokenizedRDD.map(line => {
      line.split(" ").sliding(windowSize + 1, stride).map(s => (s.slice(0, s.length - 1), s.last)).toArray
    })

    val inputs = samples.map(S => S.map(s => s._1))
    val outputs = samples.map(S => S.map(s => s._2))

    val inputsWithPositionalEmbedding = inputs.map(in => {
      in.map(window => {
        // TODO: Add embeddingDim config param
        val posEmbedding: INDArray = generatePositionalEmbedding(window, windowSize, 128)
        val embedding: INDArray = generateEmbedding(window, tokenToEmbeddingRDD, 128)
        embedding.add(posEmbedding)
      })
    })

    val outputEmbeddings = outputs.map(_.map(o => generateEmbedding(Array(o), tokenToEmbeddingRDD, 128)))
    inputsWithPositionalEmbedding.zip(outputEmbeddings).flatMap(T => {
      T._1.zip(T._2).map(t => new DataSet(t._1, t._2))
    })
  }


  def computeSlidingWindows(sc: SparkContext, inputFileName: String): Unit = {
    // Read Input Text from File
    val inputFilePath = getResourcePath(inputFileName)
    // Read Embeddings from CSV
    val embeddingsFilename = s"/${sc.getConf.get("embeddingFilename", "inputs/embeddings.csv")}"
    val embeddingPath = getResourcePath(embeddingsFilename)

    if (inputFilePath == null || embeddingPath == null) {
      val missingFilename = if (inputFilePath.equals(null)) inputFileName else embeddingsFilename
      logger.error(s"File: ${missingFilename} cannot be opened")
      return
    }

    // Convert Text File to RDD
    val inputRDD: RDD[String] = sc.textFile(inputFilePath)
    val tokenizedRDD: RDD[String] = inputRDD.flatMap(s => splitAndTokenize(s))
    logger.info("Tokenized Output Computed")
    // logger.debug(s"Tokenized Output: ${tokenizedRDD.collect().mkString(",")}")
    
    // Load Embeddings and generate maps
    val (tokenToEmbeddingRDD, embeddingToTokenRDD) = initializeEmbeddings(embeddingPath)
    logger.info("Embedding RDD Computed")

    // Mutable collection to add sliding windows
    // val datasetList: ArrayBuffer[DataSet] = new ArrayBuffer[DataSet]()

  }
}
