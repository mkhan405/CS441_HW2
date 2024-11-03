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

  private def initializeEmbeddings(embeddingPath: String) = {
    val spark = SparkSession.builder.appName("Simple Application")
      .config("spark.master", "local")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .master("local[*]")
      .getOrCreate()


    val df = spark.read.format("csv")
      //.schema(embeddingSchema)
      .option("header", "true") //first line in file has headers
      .option("inferSchema", "true") //first line in file has headers
      .load(embeddingPath)

    val dfRdd = df.rdd

    // Parse Embedding CSV Data
    val embeddingRdd = dfRdd.map(r => new EmbeddingInstance(r.get(0).toString, r.get(1).
      toString.split(";").map(_.toFloat)))

    // Generate token -> embedding and embedding -> token maps
    val tokenToEmbeddingRDD = embeddingRdd.map(i => (i.token, i.vector))
    val embeddingToTokenRDD = embeddingRdd.map(i => (i.vector, i.token))

    // Map index to token and token to index
    val indexToIndexRDD = embeddingRdd.zipWithIndex().map(i => (i._2, i._1.token))
    val tokenToIndexRDD = indexToIndexRDD.map((i) => (i._2, i._1))

    (tokenToEmbeddingRDD, embeddingToTokenRDD, indexToIndexRDD, tokenToIndexRDD)
  }

  def loadEmbeddings(sc: SparkContext, config: AppConfig): (RDD[(String, Array[Float])], RDD[(Array[Float], String)],
    RDD[(Long, String)], RDD[(String, Long)]) = {
    val embeddingInputPath = new Path(config.fileConfig.baseDir, config.fileConfig.embeddingFilePath).toString
    // Read Embeddings from CSV
    initializeEmbeddings(embeddingInputPath)
  }

}
