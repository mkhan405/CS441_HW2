package com.khan.spark.Embeddings

import com.khan.spark.Training.AppConfig
import com.khan.spark.utils.io._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}
import org.apache.hadoop.fs.{FileSystem, Path}

import java.io.FileNotFoundException

object EmbeddingLoader {
  val logger: Logger = LoggerFactory.getLogger("EmbeddingLoader")

  /**
   * Initializes embeddings from a specified CSV file and generates mapping RDDs for tokens and embeddings.
   *
   * @param embeddingPath The path to the CSV file containing token embeddings.
   * @return A tuple containing three RDDs:
   *         - RDD[(String, Array[Float])]: A mapping of token strings to their corresponding float array embeddings.
   *         - RDD[(Long, String)]: A mapping of token indices (as Long) to their corresponding token strings.
   *         - RDD[(String, Long)]: A mapping of token strings to their corresponding token indices (as Long).
   *
   * This method performs the following tasks:
   * - Creates a SparkSession configured for local execution.
   * - Reads the embeddings from the specified CSV file, inferring the schema and treating the first line as headers.
   * - Converts the DataFrame into an RDD and parses the embedding data into instances of the `EmbeddingInstance` class.
   * - Generates mappings for token to embedding, index to token, and token to index from the parsed embedding data.
   *
   * It is assumed that the CSV file is formatted correctly, with the first column containing tokens and the second column
   * containing embeddings represented as semicolon-separated float values.
   */
  private def initializeEmbeddings(embeddingPath: String) = {
    val spark = SparkSession.builder.appName("Simple Application")
      .config("spark.master", "local")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .master("local[*]")
      .getOrCreate()


    val df = spark.read.format("csv")
      .option("header", "true") //first line in file has headers
      .option("inferSchema", "true") //first line in file has headers
      .load(embeddingPath)

    val dfRdd = df.rdd

    // Parse Embedding CSV Data
    val embeddingRdd = dfRdd.map(r => new EmbeddingInstance(r.get(0).toString, r.get(1).
      toString.split(";").map(_.toFloat)))

    // Generate token -> embedding and embedding -> token maps
    val tokenToEmbeddingRDD = embeddingRdd.map(i => (i.token, i.vector))

    // Map index to token and token to index
    val indexToIndexRDD = embeddingRdd.zipWithIndex().map(i => (i._2, i._1.token))
    val tokenToIndexRDD = indexToIndexRDD.map((i) => (i._2, i._1))

    (tokenToEmbeddingRDD, indexToIndexRDD, tokenToIndexRDD)
  }

  /**
   * Loads embeddings from a specified CSV file and returns three RDDs for further processing.
   *
   * @param sc The SparkContext used for distributed data processing.
   * @param config The configuration object containing file paths and other relevant parameters.
   * @return A tuple containing three RDDs:
   *         - RDD[(String, Array[Float])]: A mapping of token strings to their corresponding float array embeddings.
   *         - RDD[(Long, String)]: A mapping of token to their index in the vocabulary
   *         - RDD[(String, Long)]: A mapping of index to their corresponding token in the vocabulary
   *
   * This method performs the following tasks:
   * - Constructs the full path to the embedding CSV file using the base directory and embedding file path from the configuration.
   * - Initializes the loading of embeddings from the specified file path.
   *
   * It is expected that the embeddings are formatted correctly in the CSV to ensure successful loading.
   */
  def loadEmbeddings(sc: SparkContext, config: AppConfig): (RDD[(String, Array[Float])], RDD[(Long, String)],
    RDD[(String, Long)]) = {
    val embeddingInputPath = new Path(config.fileConfig.baseDir, config.fileConfig.embeddingFilePath).toString
    // Read Embeddings from CSV
    initializeEmbeddings(embeddingInputPath)
  }
}
