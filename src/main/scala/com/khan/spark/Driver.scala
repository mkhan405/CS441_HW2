package com.khan.spark

import org.apache.spark.sql.SparkSession
import com.khan.spark.SlidingWindows.Compute._
import com.typesafe.config.ConfigFactory
import org.apache.spark._
import org.slf4j.{Logger, LoggerFactory}

object Driver {
  def main(args:Array[String] ): Unit = {
    val conf: SparkConf = new SparkConf().setAppName("CS 411 HW2").setMaster("local[*]")
    val applicationConfig = ConfigFactory.load().resolve()

    // Setting Input File Paths
    conf.set("inputFilename", s"${applicationConfig.getString("training-conf.inputFilePath")}")
    conf.set("embeddingFilename", s"${applicationConfig.getString("training-conf.embeddingFilePath")}")
    // Setting sliding window parameters
    conf.set("embeddingDim", s"${applicationConfig.getInt("training-conf.embeddingDim")}")
    conf.set("windowSize", s"${applicationConfig.getInt("window-conf.size")}")
    conf.set("stride", s"${applicationConfig.getInt("window-conf.stride")}")
    conf.set("pad_token", s"${applicationConfig.getInt("window-conf.pad_token")}")

    val sparkContext: SparkContext = new SparkContext(conf)
    val logger: Logger = LoggerFactory.getLogger("Driver")
    logger.info("Driver Started...")

    computeSlidingWindows(sparkContext)

    sparkContext.stop()
  }
}


//    val spark = SparkSession.builder.appName("Simple Application")
//      .config("spark.master", "local").getOrCreate()
//
//    val logger: Logger = LoggerFactory.getLogger("Main Program")
//
//    val resourcePath = getResourcePath("/README.md")
//    val logData = spark.read.textFile(resourcePath).cache()
//    val numAs = logData.filter(line => line.contains("a")).count()
//    val numBs = logData.filter(line => line.contains("b")).count()
//    logger.info(s"Lines with a: $numAs, Lines with b: $numBs")
//    spark.stop()