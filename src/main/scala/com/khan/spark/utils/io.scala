package com.khan.spark.utils

import com.khan.spark.SlidingWindows.Compute.logger
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.SparkContext

import java.io.FileNotFoundException
import java.net.URI
import javax.annotation.Nullable

object io {
  private val config: Config = ConfigFactory.load().resolve()
  val isProd: Boolean = sys.props.getOrElse("prod", "false").toBoolean

  @Nullable
  def getResourcePath(path: String): String = {
    try {
      if (!isProd) {
        getClass.getResource(path).toString
      } else {
        val s3URI = config.getString("resources.s3URI")
        s3URI + path
      }
    } catch {
      case e: NullPointerException => null
    }
  }

  def getFilePath(filename: String): String = {
    // Read Input Text from File
    val inputFilename = s"/${filename}"
    val inputFilePath = getResourcePath(inputFilename)

    if (inputFilePath == null) {
      logger.error(s"File: ${inputFilePath} cannot be opened")
      throw new FileNotFoundException(s"File not found: ${inputFilePath}")
    }
    inputFilePath
  }

  def configureFileSystem(sc: SparkContext, baseDir: String): FileSystem = {
    // Get Hadoop configuration from Spark context
    val hadoopConfig = sc.hadoopConfiguration

    if (isProd) {
      // Configure S3 filesystem settings
      hadoopConfig.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
      hadoopConfig.set("fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")

      val s3Uri = new URI(baseDir)
      FileSystem.get(s3Uri, hadoopConfig)
    } else {
      FileSystem.get(hadoopConfig)
    }
  }
}