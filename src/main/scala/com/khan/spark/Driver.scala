package com.khan.spark

import org.apache.spark.sql.SparkSession

import com.khan.spark.SlidingWindows.Compute._

import org.apache.spark._
import org.slf4j.{Logger, LoggerFactory}

object Driver {
  def main(args:Array[String] ): Unit = {
    val conf: SparkConf = new SparkConf().setAppName("CS 411 HW2").setMaster("local[*]")

    conf.set("embeddingFilename", "input/embeddings.csv")

    val sparkContext: SparkContext = new SparkContext(conf)
    val logger: Logger = LoggerFactory.getLogger("Driver")
    logger.info("Driver Started...")

    computeSlidingWindows(sparkContext, "/input/test.txt")

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