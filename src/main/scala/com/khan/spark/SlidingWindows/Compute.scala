package com.khan.spark.SlidingWindows

import com.khan.spark.utils.io.getResourcePath
import com.khan.spark.utils.EncodingData._
import org.apache.spark._
import org.apache.spark.rdd._
import org.slf4j.{Logger, LoggerFactory}

object Compute {

  val logger: Logger = LoggerFactory.getLogger("SlidingWindowCompute")

  private def splitAndTokenize(str: String): Array[String] = {
    str.split(" ").map(t => encoding.encode(t)).flatMap(bpe => bpe.toArray).map(_.toString)
  }


  def computeSlidingWindows(sc: SparkContext, inputFileName: String) = {
    val inputFilePath = getResourcePath(inputFileName)
    val inputRDD: RDD[String] = sc.textFile(inputFilePath)
    val tokenizedRDD: RDD[String] = inputRDD.flatMap(s => splitAndTokenize(s))
    logger.debug(s"Tokenized Text Computed: ${tokenizedRDD.collect().mkString(",")}")
  }
}
